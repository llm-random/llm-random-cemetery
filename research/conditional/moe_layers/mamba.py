import torch
from einops import rearrange

from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.utils.layer_manager import LoggingLayer


class MambaInProj(LoggingLayer):
    def __init__(self, conv_proj, gate_proj):
        super().__init__()
        self.conv_proj = conv_proj
        self.gate_proj = gate_proj
        self.bias = None

    @property
    def weight(self):
        return self

    def forward(self, x):
        raise NotImplementedError

    def __matmul__(self, other):
        return torch.cat((self.conv_proj(other), self.gate_proj(other)))


class MambaConvMoE(ExpertChoiceFF):
    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def forward(self, x):
        x = rearrange(x, "d (b l) -> b l d", b=self.batch_size)
        return rearrange(super().forward(x), "b l d -> d (b l)")
