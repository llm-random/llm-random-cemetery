import dataclasses

import torch

from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoEAdaTempPositive(ContinuousMoEAdaTemp):
    """
    learnable temperature,
    just like ContinuousMoEAdaTemp, but with temperature > 0
    inherit from ContinuousMoEAdaTemp
    """

    def get_temperature(self):
        if self.logging_switch:
            self.update_cache_for_logging("temperature_merge", self.temperature_merge.data.clone().detach())
            self.update_cache_for_logging("temperature_emit", self.temperature_emit.data.clone().detach())
        return torch.exp(self.temperature_merge - 1.0), torch.exp(
            self.temperature_emit - 1.0
        )

    def log_heavy(self):
        log = super().log_heavy()
        log[
            "log_merge_temperature"
        ] = self.logging_cache["temperature_merge"].flatten().tolist()
        log[
            "log_emit_temperature"
        ] = self.logging_cache["temperature_emit"].flatten().tolist()
        return log
