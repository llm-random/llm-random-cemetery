import torch

from research.conditional.utils.layer_manager import LoggingLayer


class MambaInProj(LoggingLayer):
    def __init__(self, conv_proj, gate_proj):
        super().__init__()
        self.conv_proj = conv_proj
        self.gate_proj = gate_proj
        self.bias = None

    @property
    def weights(self):
        return self

    def forward(self, x):
        return torch.cat((self.conv_proj(x), self.gate_proj(x)))

    def __matmul__(self, other):
        return self.forward(other)
