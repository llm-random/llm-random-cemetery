import torch
import torch.nn as nn
import torch.nn.functional as F

from lizrd.core.llm import Attention, LLM, Residual


def muP_attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    causal: bool,
    mode: bool,
):
    if mode == "dhead":
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead)
        if causal:
            a.masked_fill_(
                torch.tril(torch.ones_like(a)) == 0, float("-inf")
            )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)
    elif mode == "key":
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2) / (dhead**0.5)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        a = a * (1 / dhead**0.5)
        if causal:
            a.masked_fill_(
                torch.tril(torch.ones_like(a)) == 0, float("-inf")
            )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)
    elif mode == "key_flash":
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            key = key / (dhead**0.5)
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                is_causal=causal,
            )
    return output


class muP_AttentionMechanism(nn.Module):
    def __init__(self, mode: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dhead: int,
        causal: bool,
        *args,
        **kwargs,
    ):
        return muP_attention_mechanism(
            query=query,
            key=key,
            value=value,
            dhead=dhead,
            causal=causal,
            flash=self.mode,
        )


# if key ,or key_flash works as good as dhead mode, implement changes in forward only
class muP_Attention(Attention):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
        mode=False,
    ):
        super(muP_Attention, self).__init__(
            self,
            dmodel,
            heads,
            causal,
            init_type,
            init_scale,
            dhead=dhead,
            flash=flash,
            mode=False,
        )
        self.attention_mechanism = muP_AttentionMechanism(mode=mode)

    def forward(self, x):
        # You can reuse the parent forward method or implement a completely new one
        # For example, calling the parent forward method:
        # output = super(muP_Attention, self).forward(x)

        # Modify or override parts of the forward logic here if needed
        pass  # Replace this with your implementation

class muP_LLM(LLM):
    def __init__(self, embedding_layer, encoder_tower, head, mup_config: dict = None):
        super(muP_LLM, self).__init__(embedding_layer, encoder_tower, head)

        self.mup = False
        if mup_config is not None:
            self.mup = True
            # Register alpha_in and alpha_out as buffers to make them non-trainable
            self.register_buffer("alpha_in", torch.tensor(mup_config["alpha_in"], dtype=torch.float32))
            self.register_buffer("alpha_out", torch.tensor(mup_config["alpha_out"], dtype=torch.float32))
            self.register_buffer("m_d", torch.tensor(mup_config["m_d"], dtype=torch.float32))

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        if self.mup:
            x *= self.alpha_in  # Use the buffer value
        x = self.encoder(x)
        x = self.head(x)
        if self.mup:
            x *= (self.alpha_out / self.m_d)  # Use the buffer value
        return x
