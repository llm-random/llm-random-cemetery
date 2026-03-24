import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """Tensorized top-k MoE with SwiGLU experts and expert capacity."""

    def __init__(
        self,
        dmodel: int,
        dff: int,
        linear_fn: Callable[[int, int], nn.Module],
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        router_linear_fn: Optional[Callable[[int, int], nn.Module]] = None,
    ):
        super().__init__()

        if n_experts < 1:
            raise ValueError("n_experts must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > n_experts:
            raise ValueError("top_k must be smaller than or equal to n_experts")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.dmodel = dmodel
        self.dff = dff
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.silu = nn.SiLU()

        self.router_linear_fn = (
            linear_fn if router_linear_fn is None else router_linear_fn
        )
        self.router = self.router_linear_fn(dmodel, n_experts)
        self.ff_pre_act_weight, self.ff_pre_act_bias = self._stack_linear_params(
            linear_fn, dmodel, dff
        )
        self.gate_weight, self.gate_bias = self._stack_linear_params(
            linear_fn, dmodel, dff
        )
        self.ff_post_act_weight, self.ff_post_act_bias = self._stack_linear_params(
            linear_fn, dff, dmodel
        )

    def _stack_linear_params(
        self,
        linear_fn: Callable[[int, int], nn.Module],
        in_features: int,
        out_features: int,
    ) -> tuple[nn.Parameter, Optional[nn.Parameter]]:
        layers = [linear_fn(in_features, out_features) for _ in range(self.n_experts)]
        weight = torch.stack(
            [layer.weight.detach().clone().transpose(0, 1) for layer in layers],
            dim=0,
        )
        bias = None
        if layers[0].bias is not None:
            bias = torch.stack([layer.bias.detach().clone() for layer in layers], dim=0)

        return nn.Parameter(weight), None if bias is None else nn.Parameter(bias)

    def _batched_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.bmm(x, weight)
        if bias is not None:
            x = x + bias.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dmodel:
            raise ValueError(
                f"Expected input hidden size {self.dmodel}, got {x.shape[-1]}"
            )

        original_shape = x.shape
        x_flat = x.reshape(-1, self.dmodel)
        num_tokens = x_flat.shape[0]
        capacity = max(
            1,
            math.ceil(self.capacity_factor * self.top_k * num_tokens / self.n_experts),
        )

        router_logits = self.router(x_flat)
        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        routing_weights = torch.softmax(topk_logits.float(), dim=-1).to(x_flat.dtype)
        flat_expert_indices = topk_indices.reshape(-1)
        flat_token_indices = torch.arange(
            num_tokens, device=x_flat.device
        ).repeat_interleave(self.top_k)

        expert_mask = F.one_hot(flat_expert_indices, num_classes=self.n_experts).to(
            torch.int64
        )
        flat_position_in_expert = torch.cumsum(expert_mask, dim=0) - 1
        flat_position_in_expert = flat_position_in_expert.gather(
            1, flat_expert_indices.unsqueeze(-1)
        ).squeeze(-1)

        valid_assignments = flat_position_in_expert < capacity
        routing_weights = routing_weights.reshape(-1) * valid_assignments.to(
            x_flat.dtype
        )
        routing_weights = routing_weights.view(num_tokens, self.top_k)
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(routing_weights.dtype).eps)
        flat_routing_weights = routing_weights.reshape(-1)
        valid_assignments = valid_assignments & (flat_routing_weights > 0)

        expert_slots = flat_expert_indices * capacity + flat_position_in_expert
        expert_inputs = x_flat.new_zeros(self.n_experts * capacity, self.dmodel)
        expert_inputs.index_add_(
            0,
            expert_slots[valid_assignments],
            x_flat.index_select(0, flat_token_indices[valid_assignments]),
        )
        expert_inputs = expert_inputs.view(self.n_experts, capacity, self.dmodel)

        ff_pre_act = self._batched_linear(
            expert_inputs, self.ff_pre_act_weight, self.ff_pre_act_bias
        )
        gate = self._batched_linear(expert_inputs, self.gate_weight, self.gate_bias)
        expert_hidden = ff_pre_act * self.silu(gate)
        expert_outputs = self._batched_linear(
            expert_hidden, self.ff_post_act_weight, self.ff_post_act_bias
        )

        flat_expert_outputs = expert_outputs.reshape(self.n_experts * capacity, -1)
        token_outputs = flat_expert_outputs.index_select(
            0, expert_slots[valid_assignments]
        )
        token_outputs = token_outputs * flat_routing_weights[valid_assignments].unsqueeze(
            -1
        )

        combined_output = torch.zeros_like(x_flat)
        combined_output.index_add_(
            0, flat_token_indices[valid_assignments], token_outputs
        )

        return combined_output.view(*original_shape)
