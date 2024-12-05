import numpy as np

from lizrd.core.misc import Linear, LoggingLayer
from lizrd.core.llm import RoPE

import torch


class MQA(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_heads: int,
        length: int,
        init_type: str,
        init_scale: float,
        use_qk_norm: bool = False,
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
        self.head_dim = self.dmodel // self.n_heads
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        kv_proj = Linear(
            self.dmodel, 2 * self.head_dim, init_type=init_type, init_scale=init_scale
        )
        self.expert_weights = torch.nn.Parameter(kv_proj.weight.T)
        assert self.expert_weights.shape == (
            self.dmodel,
            2 * self.head_dim,
        )
        self.rope = RoPE(self.dhead, length)
        self.use_qk_norm = use_qk_norm
        self.qk_norm_scalar = torch.nn.Parameter(torch.tensor(np.log2(length**2 - length)))

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        kv = torch.einsum("ab,...a->...b", self.expert_weights, x)
        k, v = kv.split(self.head_dim, dim=-1)
        # with sdpa_kernel(backends=[SDPBackend.MATH]):

        q = q.transpose(1, 2)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        q = self.rope(q)
        k = self.rope(k)

        if self.use_qk_norm:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
            scale=self.qk_norm_scalar if self.use_qk_norm else None,
        ).transpose(1, 2)
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y


class VanillaAttention(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_heads: int,
        length: int,
        init_type: str,
        init_scale: float,
        use_qk_norm: bool = False,
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
        self.q_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.o_proj = Linear(
            self.dmodel, self.dmodel, init_type=init_type, init_scale=init_scale
        )
        kv_proj = Linear(
            self.dmodel, 2 * self.dmodel, init_type=init_type, init_scale=init_scale
        )
        self.rope = RoPE(self.dhead, length)
        self.use_qk_norm = use_qk_norm
        self.qk_norm_scalar = torch.nn.Parameter(torch.tensor(np.log2(length**2 - length)))
        self.expert_weights = torch.nn.Parameter(kv_proj.weight.T)
        assert self.expert_weights.shape == (
            self.dmodel,
            2 * self.dmodel,
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.dhead)

        kv = torch.einsum("ab,...a->...b", self.expert_weights, x)
        k, v = kv.view(batch_size, seq_len, self.n_heads, 2 * self.dhead).split(
            self.dhead, dim=-1
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if self.use_qk_norm:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=False,
            scale=self.qk_norm_scalar if self.use_qk_norm else None,
        ).transpose(1, 2)
        y = y.flatten(-2, -1)
        y = self.o_proj(y)
        return y
