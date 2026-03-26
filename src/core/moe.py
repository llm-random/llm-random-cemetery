import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import logging


logger = logging.getLogger(__name__)


@torch.no_grad()
def _truncated_normal_(weight: torch.Tensor, fan_in: int, scale: float) -> None:
    std = scale * (1 / fan_in) ** 0.5
    trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


class SonicMoEFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_load_balancing_loss_factor: float = 0.0,
        kernel_backend: str = "auto",
        activation_function: str = "swiglu",
        add_bias: bool = False,
        init_scale: float = 1.0,
        **_ignored_kwargs,
    ):
        super().__init__()

        from sonicmoe import MoE
        from sonicmoe.enums import ActivationType, KernelBackendMoE
        import sonicmoe.moe as sonicmoe_moe

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_load_balancing_loss_factor = moe_load_balancing_loss_factor
        self.requested_kernel_backend = kernel_backend
        self.backend_enum = KernelBackendMoE
        self.kernel_backend = self._resolve_kernel_backend(
            kernel_backend,
            KernelBackendMoE,
        )
        self.layer = MoE(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=ActivationType(activation_function),
            add_bias=add_bias,
            std=1.0,
        )
        self._activation_fn = {
            ActivationType.SWIGLU: sonicmoe_moe._swiglu,
            ActivationType.GEGLU: sonicmoe_moe._geglu,
            ActivationType.REGLU: sonicmoe_moe._reglu,
            ActivationType.GELU: sonicmoe_moe._gelu,
            ActivationType.RELU: sonicmoe_moe._relu,
            ActivationType.SILU: sonicmoe_moe._silu,
            ActivationType.RELU_SQ: sonicmoe_moe._relu_sq,
        }[self.layer.activation_function]
        self.aux_loss = None

        _truncated_normal_(self.layer.router.weight, hidden_size, init_scale)
        _truncated_normal_(self.layer.c_fc.weight, hidden_size, init_scale)
        _truncated_normal_(self.layer.c_proj.weight, intermediate_size, init_scale)

        with torch.no_grad():
            if self.layer.c_fc.bias is not None:
                self.layer.c_fc.bias.zero_()
            if self.layer.c_proj.bias is not None:
                self.layer.c_proj.bias.zero_()

    def _resolve_kernel_backend(self, kernel_backend: str, backend_enum):
        supports_sonicmoe = (
            self.hidden_size >= 512
            and self.hidden_size % 64 == 0
            and self.intermediate_size % 64 == 0
        )
        if kernel_backend == "auto":
            return (
                backend_enum.sonicmoe if supports_sonicmoe else backend_enum.torch
            )
        if kernel_backend == "sonicmoe" and not supports_sonicmoe:
            logger.warning(
                "Falling back to sonicmoe torch backend because hidden_size=%s and intermediate_size=%s do not satisfy the sonicmoe kernel constraints.",
                self.hidden_size,
                self.intermediate_size,
            )
            return backend_enum.torch
        return backend_enum(kernel_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x, self.aux_loss = self._forward_with_backend(
                x,
                self.kernel_backend,
            )
        except Exception as exc:
            if (
                self.requested_kernel_backend == "auto"
                and self.kernel_backend == self.backend_enum.sonicmoe
            ):
                logger.warning(
                    "Falling back to sonicmoe torch backend after sonicmoe kernel initialization failed: %s",
                    exc,
                )
                self.kernel_backend = self.backend_enum.torch
                x, self.aux_loss = self._forward_with_backend(
                    x,
                    self.kernel_backend,
                )
            else:
                raise
        return x

    def _forward_with_backend(self, x: torch.Tensor, kernel_backend):
        if kernel_backend == self.backend_enum.torch:
            return self._forward_torch_backend(x)
        return self.layer(
            x,
            kernel_backend_moe=kernel_backend,
        )

    def _forward_torch_backend(self, x: torch.Tensor):
        original_shape = x.shape
        hidden_states = x.view(-1, self.hidden_size)
        router_logits, router_weights, selected_experts = (
            self.layer._compute_routing_weights(hidden_states)
        )
        hidden_states, expert_frequency = self._compute_experts_torch(
            hidden_states=hidden_states,
            router_weights=router_weights,
            selected_experts=selected_experts,
        )
        hidden_states = hidden_states.view(original_shape)
        if self.training:
            aux_loss = self.layer._compute_switch_loss(
                logits=router_logits,
                probs=F.softmax(router_logits, dim=-1, dtype=torch.float32),
                expert_frequency=expert_frequency,
            )
        else:
            aux_loss = None
        return hidden_states, aux_loss

    def _compute_experts_torch(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ):
        total_tokens = hidden_states.size(0)
        selected_experts = selected_experts.flatten()

        with torch.no_grad():
            _sorted_expert_idxs, sorted_scattered_idxs = selected_experts.sort()

        expert_frequency = selected_experts.bincount(
            minlength=self.layer.num_experts
        ).to(torch.int32)
        fan_in_index = sorted_scattered_idxs // self.layer.top_k
        batch_gates = router_weights.flatten()[sorted_scattered_idxs]

        hidden_states = hidden_states[fan_in_index]
        hidden_states = self.layer.c_fc.torch_forward(
            input=hidden_states,
            expert_frequency=expert_frequency,
            return_list=True,
        )
        hidden_states = [self._activation_fn(tensor) for tensor in hidden_states]
        hidden_states = self.layer.c_proj.torch_forward(
            input=hidden_states,
            expert_frequency=None,
            return_list=False,
        )
        hidden_states = hidden_states * batch_gates.unsqueeze(-1)

        zeros = torch.zeros(
            (total_tokens, self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = zeros.index_add(0, fan_in_index, hidden_states)
        return hidden_states, expert_frequency


class TorchMoEFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_load_balancing_loss_factor: float = 0.0,
        activation_function: str = "swiglu",
        add_bias: bool = False,
        init_scale: float = 1.0,
        **_ignored_kwargs,
    ):
        super().__init__()

        if activation_function != "swiglu":
            raise ValueError(
                f"TorchMoEFeedForward supports only swiglu, got {activation_function}."
            )
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok={num_experts_per_tok} must be <= num_experts={num_experts}."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_load_balancing_loss_factor = moe_load_balancing_loss_factor
        self.aux_loss = None

        self.router = nn.Module()
        self.router.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.ff_pre_act = nn.Module()
        self.ff_pre_act.weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.ff_post_act = nn.Module()
        self.ff_post_act.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )

        if add_bias:
            self.router.bias = nn.Parameter(torch.empty(num_experts))
            self.ff_pre_act.bias = nn.Parameter(
                torch.empty(num_experts, intermediate_size)
            )
            self.gate.bias = nn.Parameter(torch.empty(num_experts, intermediate_size))
            self.ff_post_act.bias = nn.Parameter(
                torch.empty(num_experts, hidden_size)
            )
        else:
            self.router.register_parameter("bias", None)
            self.ff_pre_act.register_parameter("bias", None)
            self.gate.register_parameter("bias", None)
            self.ff_post_act.register_parameter("bias", None)

        _truncated_normal_(self.router.weight, hidden_size, init_scale)
        _truncated_normal_(self.ff_pre_act.weight, hidden_size, init_scale)
        _truncated_normal_(self.gate.weight, hidden_size, init_scale)
        _truncated_normal_(self.ff_post_act.weight, intermediate_size, init_scale)

        with torch.no_grad():
            if self.router.bias is not None:
                self.router.bias.zero_()
                self.ff_pre_act.bias.zero_()
                self.gate.bias.zero_()
                self.ff_post_act.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        hidden_states = x.reshape(-1, self.hidden_size)

        # Route each token to its top-k experts in fp32 for stable probabilities.
        router_logits = torch.einsum(
            "th,eh->te",
            hidden_states,
            self.router.weight,
        )
        if self.router.bias is not None:
            router_logits = router_logits + self.router.bias
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_weights, selected_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        # Gather the chosen expert weights so the expert MLP stays fully batched.
        expert_inputs = hidden_states.unsqueeze(1).expand(
            -1, self.num_experts_per_tok, -1
        )
        ff_pre_act = torch.einsum(
            "tkh,tkdh->tkd",
            expert_inputs,
            self.ff_pre_act.weight[selected_experts],
        )
        gate = torch.einsum(
            "tkh,tkdh->tkd",
            expert_inputs,
            self.gate.weight[selected_experts],
        )
        if self.ff_pre_act.bias is not None:
            ff_pre_act = ff_pre_act + self.ff_pre_act.bias[selected_experts]
            gate = gate + self.gate.bias[selected_experts]
        expert_outputs = torch.einsum(
            "tkd,tkhd->tkh",
            ff_pre_act * F.silu(gate),
            self.ff_post_act.weight[selected_experts],
        )
        if self.ff_post_act.bias is not None:
            expert_outputs = expert_outputs + self.ff_post_act.bias[selected_experts]

        output = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=1)
        output = output.reshape(original_shape)

        # Match the switch-style load-balancing term expected by the trainer.
        if self.training:
            expert_frequency = selected_experts.reshape(-1).bincount(
                minlength=self.num_experts
            )
            expert_frequency = expert_frequency.to(router_probs.dtype)
            expert_frequency = expert_frequency / expert_frequency.sum()
            self.aux_loss = self.num_experts * (
                router_probs.mean(dim=0) * expert_frequency
            ).sum()
        else:
            self.aux_loss = None

        return output
