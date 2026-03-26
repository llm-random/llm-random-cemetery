import torch
import torch.nn as nn
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
        kernel_backend: str = "auto",
        activation_function: str = "swiglu",
        add_bias: bool = False,
        init_scale: float = 1.0,
        **_ignored_kwargs,
    ):
        super().__init__()

        from sonicmoe import MoE
        from sonicmoe.enums import ActivationType, KernelBackendMoE

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        if kernel_backend == self.backend_enum.torch and x.dtype != torch.float32:
            if self.layer.router.weight.dtype != torch.float32:
                self.layer = self.layer.float()
            output, aux_loss = self.layer(
                x.float(),
                kernel_backend_moe=kernel_backend,
            )
            return output.to(dtype=x.dtype), aux_loss
        return self.layer(
            x,
            kernel_backend_moe=kernel_backend,
        )
