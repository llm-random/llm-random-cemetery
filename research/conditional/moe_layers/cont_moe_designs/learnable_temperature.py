import dataclasses

import torch

from lizrd.core import nn
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTemp(ContinuousMoeBaseClass):
    """
    learnable temperature,
    either shared by experts or not,
    either shared for merge and emit or not
    """

    share_by_experts: bool = True
    share_by_emit_merge: bool = True
    is_temperature_learning = False


    def get_temperature(self):
        if self.is_temperature_learning:
            return self.temperature_merge.detach(), self.temperature_emit.detach()
        else:
            return self.temperature_merge, self.temperature_emit

    def init_additional_parameters(self):
        if self.share_by_experts:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(1))
                self.temperature_merge = nn.Parameter(torch.ones(1))
        else:
            if self.share_by_emit_merge:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = self.temperature_emit
            else:
                self.temperature_emit = nn.Parameter(torch.ones(self.n_experts, 1))
                self.temperature_merge = nn.Parameter(torch.ones(self.n_experts, 1))

