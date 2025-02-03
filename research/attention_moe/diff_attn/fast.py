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
        self.n_negative_heads = n_negative_heads or n_heads
        self.seq_len = seq_len

        self.n_heads = n_heads
        # self.num_kv_heads = (
        #     args.decoder_kv_attention_heads // args.model_parallel_size
        #     if args.decoder_kv_attention_heads is not None
        #     else num_heads // args.model_parallel_size
        # )

        assert (int(roll_negative_heads) + int(flip_negative_heads)) <= 1

        # TODO inżynierka configów pairwise
        # TODO QQA + QDA
        # TODO naprawić adaptery z powrotem
        # TODO DoRA jako adapter
        # TODO none razem z resztą
        # TODO sanity check multi/add/multiadd
        # TODO Skąd bierze pomysły?
        # TODO fineweb dataset?
        # TODO run exp --yes
        # TODO folder na expy o tej samej nazwie

        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        # Todo paramater matched GQA bez adapterów
        self.dhead = dmodel // n_heads

        self.adapter_type = adapter_type
        self.lowrank_inner_dim = lowrank_inner_dim
        if self.adapter_type == "lora" and self.lowrank_inner_dim > 0:
            self.lowrank_q = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=self.dhead * self.n_negative_heads,
            )
            self.lowrank_k = Lowrank(
                self.dmodel,
                self.lowrank_inner_dim,
                init_type,
                init_scale,
                output_dim=self.dhead * self.n_negative_heads,
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
        elif self.adapter_type == "none":
            self.q_neg_proj = Linear(
                self.dmodel,
                self.dhead * self.n_negative_heads,
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            )
            self.k_neg_proj = Linear(
                self.dmodel,
                # self.dhead * self.n_negative_heads,
                self.dhead * self.n_kv_heads,
                bias=False,
                init_type=init_type,
                init_scale=init_scale,
            )
        elif self.adapter_type == "identity":
            pass
        else:
            raise NotImplementedError

        self.scaling = self.dhead ** -0.5

        self.q_proj = Linear(
            self.dmodel,
            self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.k_proj = Linear(
            self.dmodel,
            self.dhead * self.n_kv_heads,
            # self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.v_proj = Linear(
            self.dmodel,
            self.dhead * self.n_kv_heads,
            # self.dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.out_proj = Linear(
            self.dmodel, self.dmodel, bias=False, init_type=init_type, init_scale=init_scale
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

        self.subln = RMSNorm(self.dhead, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        rel_pos=None,
        attn_mask=None,
    ):
        global q_negative
        global k_negative
        global q_trunc
        global k_trunc
        bsz, _, _ = x.size()
        # print(f"x.shape: {x.shape}")
       
        if self.lambda_init is None:
            self.lambda_init = lambda_init_fn(self.block_number)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # print(f"q.shape: {q.shape}")

        if self.adapter_type != "none":
            q_trunc = q[:, :, :self.dhead * self.n_negative_heads]
            k_trunc = k[:, :, :self.dhead * self.n_negative_heads]
        # q_trunc = q[:, :, :self.dhead * self.n_positive_heads]
        # k_trunc = k[:, :, :self.dhead * self.n_positive_heads]

        # print(f"q_trunc.shape: {q_trunc.shape}")

        if self.adapter_type == "lora":
            # self.lowrank_inner_dim > 0:
            q_negative = (q_trunc + self.lowrank_q(x)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.lowrank_k(x)).view(
                # bsz, self.seq_len, self.num_kv_heads, self.head_dim
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
        elif self.adapter_type == "additive":
            q_negative = (q_trunc + self.q_delta.repeat(bsz, self.seq_len, 1)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc + self.k_delta.repeat(bsz, self.seq_len, 1)).view(
                # bsz, self.seq_len, self.num_kv_heads, 2 * self.head_dim
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
        elif self.adapter_type == "multiplicative":
            q_negative = (q_trunc * self.q_delta.repeat(bsz, self.seq_len, 1)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc * self.k_delta.repeat(bsz, self.seq_len, 1)).view(
                # bsz, self.seq_len, self.num_kv_heads, 2 * self.head_dim
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
        elif self.adapter_type == "multiadd":
            q_negative = (q_trunc * self.q_delta_mult.repeat(bsz, self.seq_len, 1) + self.q_delta_add.repeat(bsz, self.seq_len, 1)).view(
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            k_negative = (k_trunc * self.k_delta_mult.repeat(bsz, self.seq_len, 1) + self.k_delta_add.repeat(bsz, self.seq_len, 1)).view(
                # bsz, self.seq_len, self.num_kv_heads, self.head_dim
                bsz, self.seq_len, self.n_negative_heads, self.dhead
            )
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
        elif self.adapter_type == "identity":
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)
            q_negative = q_trunc.clone().view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
            k_negative = k_trunc.clone().view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
            # q_negative = q.clone().view(bsz, self.seq_len, self.n_negative_heads, 2 * self.dhead)
            # k_negative = k.clone().view(bsz, self.seq_len, self.n_negative_heads, 2 * self.dhead)
        elif self.adapter_type == "none":
            q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
            k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
            v = v.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

            q_negative = self.q_neg_proj(x).view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
            # k_negative = self.k_neg_proj(x).view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
            k_negative = self.k_neg_proj(x).view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

            # q_negative = q[:, :, self.dhead * self.n_heads:].view(bsz, self.seq_len, self.n_negative_heads, self.dhead)
            # k_negative = k[:, :, self.dhead * self.n__heads:].view(bsz, self.seq_len, self.n_negative_heads, self.dhead)

            # q = q[:, :, :self.dhead * self.n_heads].view(bsz, self.seq_len, self.n_heads, self.dhead)
            # k = k[:, :, :self.dhead * self.n_heads].view(bsz, self.seq_len, self.n_heads, self.dhead)
            # v = v.view(bsz, self.seq_len, self.n_heads, self.dhead)

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
            q2 = q_negative.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            k1 = k
            k2 = k_negative.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            # if self.num_kv_heads != self.num_heads:
            #     k1 = k1.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     k2 = k2.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            #     assert (
            #         k1.shape == k2.shape == q1.shape == q2.shape
            #     ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"
        else:
            # q = q.reshape(bsz, self.seq_len, self.n_heads, self.dhead)
            # k = k.reshape(bsz, self.seq_len, self.n_kv_heads, 2, self.dhead)
            # k = k.reshape(bsz, self.seq_len, self.n_heads, self.dhead)
            # k = k.reshape(bsz, self.seq_len, self.n_kv_heads, 2, self.dhead)
            # q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
            # k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

            q1 = q
            q2 = q_negative #.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)

            k1 = k
            k2 = k_negative

            # # k1 = k
            # # k2 = k_negative.repeat(1, 1, self.n_heads // self.n_negative_heads, 1)
            #
            # k1 = k.repeat(1, 1, self.n_heads // self.n_kv_heads, 1)
            # k2 = k_negative.repeat(1, 1, self.n_heads // self.n_kv_heads, 1)
            # assert (
            #             k1.shape == k2.shape == q1.shape == q2.shape
            #         ), f"Shapes don't match: {k1.shape}, {k2.shape}, {q1.shape}, {q2.shape}"

            # v1 = v[:, :, :self.n_heads // 2]
            # v2 = v[:, :, self.n_heads // 2:]

        # TODO flip/roll
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
            # print(f"attn1_scores.shape: {attn1_scores.shape}")
            # print(f"attn2_scores.shape: {attn2_scores.shape}")

            differential_scores = torch.cat((attn1_scores, -lambda_full * attn2_scores), 1)
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

        # print(f"attn1.shape: {attn1.shape}")
        # print(f"attn2.shape: {attn2.shape}")

        attn2 = -lambda_full * attn2
        attn = attn1 + attn2

        # print(f"attn2lambda.shape: {attn2.shape}")

        # attn = torch.cat((attn1, attn2), 2)

        # print(f"attn.shape: {attn.shape}")

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, self.seq_len, self.n_heads * self.dhead)

        # print(f"attn.shape: {attn.shape}")

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
        dmodel,
        n_heads,
        use_rope,
        seq_len,
        init_type,
        init_scale,
    ):
        super().__init__()
        self.dmodel = dmodel
        # num_heads set to half of Transformer's #heads
        self.n_heads = n_heads
        self.n_kv_heads = self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.save_attention_weights = False
        self.attention_weights = None

        self.dhead = dmodel // n_heads
        self.scaling = self.dhead ** -0.5
        self.seq_len = seq_len
        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.dhead, base=10000.0, interleaved=True
            )
            self.rotary_emb._update_cos_sin_cache(self.seq_len, dtype=torch.float32)

        self.q_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.k_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.v_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )
        self.out_proj = Linear(
            dmodel, dmodel, bias=False, init_type=init_type, init_scale=init_scale
        )

    def forward(
        self,
        x,
        # rel_pos=None,
        # attn_mask=None,
    ):
        bsz, _, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, self.seq_len, self.n_heads, self.dhead)
        k = k.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)
        v = v.view(bsz, self.seq_len, self.n_kv_heads, self.dhead)

        if self.use_rope:
            assert self.rotary_emb._cos_cached.dtype == torch.float32
            rel_pos = (
                self.rotary_emb._cos_cached.to(x.device),
                self.rotary_emb._sin_cached.to(x.device),
            )
            q = apply_rotary_emb(q.to(dtype=torch.float32), *rel_pos, interleaved=True)
            k = apply_rotary_emb(k.to(dtype=torch.float32), *rel_pos, interleaved=True)

        q = q.reshape(bsz, self.seq_len, self.n_heads, self.dhead)
        k = k.reshape(bsz, self.seq_len, self.n_kv_heads, self.dhead)
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

        attn = attn.reshape(bsz, self.seq_len, self.n_heads * self.dhead).to(x)

        attn = self.out_proj(attn)
        return attn
