import math
from typing import Optional
import torch
import numpy as np

from lizrd.core.misc import (
    Linear,
    LoggingLayer,
    time_measured,
)
from lizrd.core.llm import RoPE
from research.attention_moe.moe_layers_cc.moe_gating import TokenGating


class TokenChoiceMoMQA(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_heads: int,
        length: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        zloss_weight: float = 0,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        get_router_values_from: str = "weights",
        moe_values_exp: Optional[int] = 1,
        detach_gate: bool = False,
        use_dropped_tokens_head: bool = False,
        use_qk_norm: bool = False,
        scale_attention_inside: bool = False,
        scale_attention_outside: bool = False,
        **_,
    ):
        """
        Args:
            dmodel: dimension of the input
            doutput: dimension of the output (default: dmodel)
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()
        self.dmodel = dmodel
        self.n_heads = n_heads
        assert self.dmodel % self.n_heads == 0
        self.dhead = self.dmodel // self.n_heads
        self.gating = TokenGating(
            dmodel=dmodel,
            n_experts=n_heads,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
            get_router_values_from=get_router_values_from,
            detach_gate=detach_gate,
            moe_values_exp=moe_values_exp,
            zloss_weight=zloss_weight,
        )
        experts = []
        for _ in range(self.n_heads):
            expert = Linear(
                self.dmodel,
                2 * self.dhead,
                init_type=init_type,
                init_scale=init_scale,
            )
            experts.append(expert)
        self.expert_weights = torch.nn.Parameter(
            torch.stack([e.weight.T for e in experts], dim=0)
        )
        self.dropped_k = torch.nn.Parameter(
            torch.randn(self.dhead, dtype=torch.float32) * init_scale
        )
        self.dropped_v = torch.nn.Parameter(
            torch.randn(self.dhead, dtype=torch.float32) * init_scale
        )
        self.use_dropped_tokens_head = use_dropped_tokens_head
        if self.use_dropped_tokens_head:
            self.dropped_kv_head = Linear(
                self.dmodel,
                2 * self.dhead,
                init_type=init_type,
                init_scale=init_scale,
            )
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.rope = RoPE(self.dhead, length)
        self.use_qk_norm = use_qk_norm
        self.scale_attention_inside = scale_attention_inside
        self.scale_attention_outside = scale_attention_outside
        # self.qk_norm_scalar = torch.nn.Parameter(torch.tensor(np.log2(length**2 - length)))
        self.attention_scalar_inside = torch.nn.Parameter(torch.tensor(1 / math.sqrt(self.dhead)))
        self.attention_scalar_outside = torch.nn.Parameter(torch.tensor(1.0))
        self.q_norm = torch.nn.LayerNorm(self.dhead)
        self.k_norm = torch.nn.LayerNorm(self.dhead)
        assert self.expert_weights.shape == (
            self.n_heads,
            self.dmodel,
            2 * self.dhead,
        )

    @time_measured("assign_tokens_to_input")
    def extract(self, x, token_indicies):
        capacity = token_indicies.shape[0]
        token_indicies = token_indicies.T.reshape(self.n_heads * capacity)
        experts_input = x[token_indicies, :]
        experts_input = experts_input.reshape(self.n_heads, capacity, self.dmodel)
        return experts_input

    @time_measured("assign_tokens_to_output")
    def merge(
        self,
        experts_output,
        token_expert_values,
        token_expert_indices,
        batch_size,
        seq_len,
        x,
    ):
        output = torch.zeros(
            batch_size * seq_len,
            self.dhead,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        experts_output = experts_output * token_expert_values.T.unsqueeze(-1)
        output.index_add_(
            dim=0,
            index=token_expert_indices.T.flatten(),
            source=experts_output.reshape(
                self.n_heads * experts_output.shape[1], self.dhead
            ),
        )
        output = output.reshape(batch_size, seq_len, self.dhead)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        token_expert_indices, token_expert_values = self.gating(x)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.dhead)

        x = x.flatten(start_dim=0, end_dim=1)
        experts_input = self.extract(x, token_expert_indices)
        kv = torch.einsum("nab,n...a->n...b", self.expert_weights, experts_input)
        k, v = kv.split(self.dhead, dim=-1)
        is_token_dropped = torch.zeros(
            batch_size * seq_len,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        assert token_expert_indices.shape == token_expert_values.shape
        is_token_dropped.index_add_(
            dim=0,
            index=token_expert_indices.flatten(),
            source=token_expert_values.flatten(),
        )
        is_token_dropped = (is_token_dropped == 0.0).reshape(batch_size, seq_len)
        k = self.merge(
            k,
            1 - (token_expert_values == 0.0).to(k.dtype),
            token_expert_indices,
            batch_size,
            seq_len,
            x,
        )

        v = self.merge(
            v,
            token_expert_values,
            token_expert_indices,
            batch_size,
            seq_len,
            x,
        )

        if self.use_dropped_tokens_head:
            dropped_kv = self.dropped_kv_head(x)
            dropped_k, dropped_v = dropped_kv.split(
                self.dhead,
                dim=-1,
            )
            # print(is_token_dropped.shape, dropped_k.shape, k.shape)
            # is_token_dropped = is_token_dropped.reshape(batch_size, seq_len)
            is_token_dropped = is_token_dropped.unsqueeze(-1)
            k = (
                is_token_dropped * dropped_k.reshape(batch_size, seq_len, self.dhead)
                + ~is_token_dropped * k
            )
            v = (
                is_token_dropped * dropped_v.reshape(batch_size, seq_len, self.dhead)
                + ~is_token_dropped * v
            )
        else:
            k[is_token_dropped] = self.dropped_k
            v[is_token_dropped] = self.dropped_v

        q = q.transpose(1, 2)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        if self.use_qk_norm:
            # q = torch.nn.functional.normalize(q, p=2, dim=-1)
            # k = torch.nn.functional.normalize(k, p=2, dim=-1)
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
            # scale=self.qk_norm_scalar if self.use_qk_norm else None,
            scale=self.attention_scalar_inside if self.scale_attention_inside else None,
        ).transpose(1, 2)
        y = y * self.attention_scalar_outside
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y
