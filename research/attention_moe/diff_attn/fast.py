import math
import torch
import torch.nn.functional as F
from torch import nn

from lizrd.core.misc import Linear

from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
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


class MultiheadFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized-flash-attention (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """

    def __init__(
        self,
        # args,
        embed_dim,
        # depth,
        num_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads  # // args.model_parallel_size
        # self.num_kv_heads = (
        #     args.decoder_kv_attention_heads // args.model_parallel_size
        #     if args.decoder_kv_attention_heads is not None
        #     else num_heads // args.model_parallel_size
        # )

        self.num_kv_heads = num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5

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

        self.lambda_init = None
        self.use_rope = use_rope
        self.seq_len = seq_len
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                base=10000.0,
                interleaved=True,
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        if self.use_rope:
            rel_pos = (
                self.rotary_emb._cos_cached.as_type(x.dtype).to(x.device),
                self.rotary_emb._sin_cached.as_type(x.dtype).to(x.device),
            )
            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn1 = flash_attn_func(q1, k1, v, causal=True)
        attn2 = flash_attn_func(q2, k2, v, causal=True)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn


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
        init_type,
        init_scale,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # num_heads set to half of Transformer's #heads
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_rope = use_rope

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
        self.lambda_init = None
        self.rotary_emb = None

    def forward(
        self,
        x,
        # rel_pos=None,
        # attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        if self.use_rope and self.rotary_emb is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                base=10000.0,
                interleaved=True,
                device=x.device,
            )
            self.rotary_emb._update_cos_sin_cache(
                tgt_len, device=x.device, dtype=x.dtype
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        if self.use_rope:
            rel_pos = (self.rotary_emb._cos_cached, self.rotary_emb._sin_cached)
            q = apply_rotary_emb(q, *rel_pos, interleaved=True)
            k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        # offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, self.head_dim)
        # q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        # k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn = flash_attn_func(q, k, v, causal=True)
        # attn2 = flash_attn_func(q2, k2, v, causal=True)

        # lambda_1 = torch.exp(
        #     torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        # ).type_as(q)
        # lambda_2 = torch.exp(
        #     torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        # ).type_as(q)
        # lambda_full = lambda_1 - lambda_2 + self.lambda_init
        # attn = attn1 - lambda_full * attn2

        # attn = self.subln(attn)
        # attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim)

        attn = self.out_proj(attn)
        return attn
