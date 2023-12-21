import torch
from einops import rearrange

from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.utils.layer_manager import LoggingLayer


class MambaInProj(LoggingLayer):
    def __init__(self, batch_size, conv_proj, gate_proj):
        super().__init__()
        self.conv_proj = conv_proj
        self.gate_proj = gate_proj
        self.bias = None
        self.batch_size = batch_size

    @property
    def weight(self):
        return self

    def forward(self, x):
        x = rearrange(x, "d (b l) -> b l d", b=self.batch_size)
        return torch.cat(
            (
                rearrange(self.conv_proj(x), "b l d -> d (b l)"),
                rearrange(self.gate_proj(x), "b l d -> d (b l)"),
            )
        ).type(torch.float32)

    def __matmul__(self, other):
        return self.forward(other)


class MambaConvMoE(ExpertChoiceFF):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def forward(self, x):
        x = rearrange(x, "d (b l) -> b l d", b=self.batch_size)
        return rearrange(super().forward(x), "b l d -> d (b l)")
