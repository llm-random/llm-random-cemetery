import math
from weakref import ref
import torch
from torch import nn
import torch.nn.functional as F

from lizrd.core.misc import Linear, LoggingLayer

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

# try:
    # from apex.normalization import FusedRMSNorm as RMSNorm
# except ModuleNotFoundError:
#     print("No fused RMSNorm")
from .rms_norm import RMSNorm


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Lowrank(nn.Module):
    def __init__(self, outer_dim, inner_dim, init_type, init_scale, output_dim=None):
        super().__init__()
        self.inner_dim = inner_dim
        self.w1 = Linear(
            outer_dim, inner_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.w2 = Linear(
            inner_dim,
            output_dim or outer_dim,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )

    def forward(self, x):
        return self.w2(self.w1(x))


def manual_attention(q, k, v, causal=True):
    """Preserves flashattention's interface, but also returns attention weights"""
    # ...# manual implementation of attention
    if not causal:
        raise NotImplementedError
    bs, nh, slen, head_dim = q.shape
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    att = att.masked_fill(
        torch.tril(torch.ones(slen, slen).to(att)) == 0, float("-inf")
    )
    att = F.softmax(att.to(torch.float32), dim=-1)
    y = att.to(v) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y, att


class MultiheadFlashDiff1(LoggingLayer):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        # args,
        dmodel,
        # depth,
        n_heads,
        n_positive_heads,
        n_negative_heads,
        use_rope,
        seq_len,
        lowrank_inner_dim,
        flip_negative_heads,
        roll_negative_heads,
        init_type,
        init_scale,
        n_kv_heads=None,
        adapter_type="none",
    ):
        super().__init__()
        # self.args = args
        self.dmodel = dmodel
        self.save_attention_weights = False
        self.attention_weights = None
        # num_heads set to half of Transformer's #heads
        self.n_positive_heads = n_positive_heads  # // args.model_parallel_size
        self.n_negative_heads = n_negative_heads
        assert self.n_positive_heads + self.n_negative_heads == n_heads

        self.n_heads = n_heads
        # self.num_kv_heads = (
        #     args.decoder_kv_attention_heads // args.model_parallel_size
        #     if args.decoder_kv_attention_heads is not None
        #     else num_heads // args.model_parallel_size
        # )

        assert (int(roll_negative_heads) + int(flip_negative_heads)) <= 1

        # TODO QQA + QDA

        self.n_kv_heads = n_kv_heads or n_positive_heads
        self.n_rep = self.n_positive_heads // self.n_kv_heads

        self.dhead = dmodel // (n_positive_heads + n_negative_heads)

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora" and self.lowrank_inner_dim > 0:
            self.lowrank_q = Lowrank(
                self.dhead * self.n_negative_heads, self.lowrank_inner_dim, init_type, init_scale
            )
            self.lowrank_k = Lowrank(
                self.dhead * self.n_negative_heads,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                # output_dim=2 * self.dhead * self.n_kv_heads,
            )
        elif self.adapter_type == "additive":
            self.k_delta = nn.Parameter(
                torch.zeros(
                    # 2 * self.dhead * self.n_kv_heads, dtype=torch.float32
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=0, std=0.1)
            )
            self.q_delta = nn.Parameter(
                torch.zeros(self.dhead * self.n_negative_heads, dtype=torch.float32).normal_(
                    mean=0, std=0.1
                )
            )
        elif self.adapter_type == "multiplicative":
            self.k_delta = nn.Parameter(
                torch.zeros(
                    # 2 * self.dhead * self.n_kv_heads, dtype=torch.float32
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=1, std=0.1)
            )
            self.q_delta = nn.Parameter(
                torch.zeros(self.dhead * self.n_negative_heads, dtype=torch.float32).normal_(
                    mean=1, std=0.1
                )
            )
        elif self.adapter_type == "multiadd":
            self.k_delta_mult = nn.Parameter(
                torch.zeros(
                    # 2 * self.dhead * self.n_kv_heads, dtype=torch.float32
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=1, std=0.1)
            )
            self.q_delta_mult = nn.Parameter(
                torch.zeros(self.dhead * self.n_negative_heads, dtype=torch.float32).normal_(
                    mean=1, std=0.1
                )
            )
            self.k_delta_add = nn.Parameter(
                torch.zeros(
                    # 2 * self.dhead * self.n_kv_heads, dtype=torch.float32
                    self.dhead * self.n_negative_heads, dtype=torch.float32
                ).normal_(mean=0, std=0.1)
            )
            self.q_delta_add = nn.Parameter(
                torch.zeros(self.dhead * self.n_negative_heads, dtype=torch.float32).normal_(
                    mean=0, std=0.1
                )
            )
        elif self.adapter_type == "none" or self.adapter_type == "identity":
            pass
        else:
            raise NotImplementedError

        self.scaling = self.dhead ** -0.5

        self.q_proj = Linear(
            dmodel,
            self.dhead * self.n_positive_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.k_proj = Linear(
            dmodel,
            # self.dhead * 2 * self.n_kv_heads,
            self.dhead * self.n_positive_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            dmodel,
            # self.dhead * 2 * self.n_kv_heads,
            self.dhead * self.n_heads,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len
        self.flip_negative_heads = flip_negative_heads
        self.roll_negative_heads = roll_negative_heads
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead,
                base=10000.0,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.dhead, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.dhead, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        print(f"x.shape: {x.shape}")
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        print(f"q.shape: {q.shape}")

        q_trunc = q[:, :, :self.dhead * self.n_negative_heads]
        k_trunc = k[:, :, :self.dhead * self.n_negative_heads]

        print(f"q_trunc.shape: {q_trunc.shape}")

        if self.adapter_type == "lora":
            # self.lowrank_inner_dim > 0:
            q_negative = (q_trunc + self.lowrank_q(x)).view(
                bsz, tgt_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.lowrank_k(x)).view(
                # bsz, src_len, self.num_kv_heads, self.head_dim
                bsz, src_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)
        elif self.adapter_type == "additive":
            print(q_trunc.shape)
            print(self.q_delta.shape)
            print(self.q_delta.repeat(bsz, tgt_len, 1).shape)
            q_negative = (q_trunc + self.q_delta.repeat(bsz, tgt_len, 1)).view(
                bsz, tgt_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.k_delta.repeat(bsz, src_len, 1)).view(
                # bsz, src_len, self.num_kv_heads, 2 * self.head_dim
                bsz, src_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)
        elif self.adapter_type == "multiplicative":
            q_negative = (q_trunc * self.q_delta.repeat(bsz, tgt_len, 1)).view(
                bsz, tgt_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc * self.k_delta.repeat(bsz, src_len, 1)).view(
                # bsz, src_len, self.num_kv_heads, 2 * self.head_dim
                bsz, src_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)
        elif self.adapter_type == "multiadd":
            q_negative = (q_trunc * self.q_delta_mult.repeat(bsz, tgt_len, 1) + self.q_delta_add.repeat(bsz, tgt_len, 1)).view(
                bsz, tgt_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc * self.k_delta_mult.repeat(bsz, src_len, 1) + self.k_delta_add.repeat(bsz, src_len, 1)).view(
                # bsz, src_len, self.num_kv_heads, self.head_dim
                bsz, src_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)
        elif self.adapter_type == "identity":
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)
            q_negative = q_trunc.clone()
            k_negative = k.trunc.clone()
            # q_negative = q.clone().view(bsz, tgt_len, self.n_negative_heads, 2 * self.dhead)
            # k_negative = k.clone().view(bsz, src_len, self.n_negative_heads, 2 * self.dhead)
        elif self.adapter_type == "none":
            q = q.view(bsz, tgt_len, self.n_positive_heads, self.dhead)
            k = k.view(bsz, src_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, src_len, self.n_heads, self.dhead)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(
                q.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            k = apply_rotary_emb(
                k.to(dtype=torch.float32), *rel_pos, interleaved=True
            ).to(x)
            if self.adapter_type != "none":
                q_negative = apply_rotary_emb(
                    q_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
                ).to(x)
                k_negative = apply_rotary_emb(
                    k_negative.to(dtype=torch.float32), *rel_pos, interleaved=True
                ).to(x)

        if self.adapter_type != "none":
            q1 = q
            q2 = q_negative
            k1 = k
            k2 = k_negative
            v1 = v[:, :, :self.n_positive_heads]
            v2 = v[:, :, self.n_positive_heads:]
            print(f"q1.shape: {q1.shape}")
            print(f"q2.shape: {q2.shape}")
            print(f"k1.shape: {k1.shape}")
            print(f"k2.shape: {k2.shape}")
            print(f"v1.shape: {v1.shape}")
            print(f"v2.shape: {v2.shape}")
            # if self.num_kv_heads != self.num_heads:
            #     k1 = k1.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     k2 = k2.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     assert (
            #         k1.shape == k2.shape == q1.shape == q2.shape
            #     ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"
        else:
            q = q.reshape(bsz, tgt_len, self.n_positive_heads, 2, self.dhead)
            k = k.reshape(bsz, src_len, self.n_kv_heads, 2, self.dhead)
            q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
            k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        # if self.flip_negative_heads:
        #     q2 = torch.flip(q2, dims=(2,))
        #     k2 = torch.flip(k2, dims=(2,))
        # elif self.roll_negative_heads:
        #     q2 = torch.roll(q2, shifts=1, dims=(2,))
        #     k2 = torch.roll(k2, shifts=1, dims=(2,))

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        self.update_cache_for_logging("lambda", lambda_full)

        if self.save_attention_weights:
            attn1, attn1_scores = manual_attention(
                q1.transpose(1, 2),
                k1.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn1 = attn1.transpose(1, 2)
            attn2, attn2_scores = manual_attention(
                q2.transpose(1, 2),
                k2.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn2 = attn2.transpose(1, 2)
            if False and self.block_number == 0:
                reference_attn1 = flash_attn_func(
                    q1,
                    k1,
                    v,
                    causal=True,
                )
                reference_attn2 = flash_attn_func(
                    q2,
                    k2,
                    v,
                    causal=True,
                )
                assert torch.allclose(
                    attn1, reference_attn1, atol=1e-3
                ), f"Manual attn1 does not match reference attn1: {attn1-reference_attn1}"
                assert torch.allclose(
                    attn2, reference_attn2, atol=1e-3
                ), f"Manual attn2 does not match reference attn2"

            differential_scores = attn1_scores - lambda_full * attn2_scores
            self.attention_weights = differential_scores
        else:
            attn1 = flash_attn_func(
                q1,
                k1,
                v,
                causal=True,
            )
            attn2 = flash_attn_func(
                q2,
                k2,
                v,
                causal=True,
            )

        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.n_positive_heads * 2 * self.dhead)

        attn = self.out_proj(attn)
        return attn

    def log_light(self):
        return {
            "lambda": self.logging_cache["lambda"],
        }


class VanillaFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.save_attention_weights = False
        self.attention_weights = None

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.seq_len = seq_len
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, base=10000.0, interleaved=True
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

        self.q_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.k_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.v_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=False, init_type=init_type, init_scale=init_scale
        )

    def forward(
        self,
        x,
        # rel_pos=None,
        # attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(q.to(dtype=torch.float32), *rel_pos, interleaved=True)
            k = apply_rotary_emb(k.to(dtype=torch.float32), *rel_pos, interleaved=True)

        # offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, self.head_dim)
        if self.save_attention_weights:
            attn, attn_scores = manual_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )
            attn = attn.transpose(1, 2)
            if False and self.block_number == 0:
                reference_attn = flash_attn_func(
                    q.to(dtype=torch.bfloat16),
                    k.to(dtype=torch.bfloat16),
                    v.to(dtype=torch.bfloat16),
                    causal=True,
                )
                assert torch.allclose(
                    attn, reference_attn, atol=1e-3
                ), f"Manual attn does not match reference attn: {attn-reference_attn}"

            self.attention_weights = attn_scores
        else:
            attn = flash_attn_func(
                q.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                causal=True,
            )

        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim).to(x)

        attn = self.out_proj(attn)
        return attn
