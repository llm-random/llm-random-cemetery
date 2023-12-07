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
            self.update_cache_for_logging("temperature_merge", (self.temperature_merge -1.0).data.clone().detach())
            self.update_cache_for_logging("temperature_emit", (self.temperature_emit -1.0).data.clone().detach())

        exp_temp_merge = torch.exp(self.temperature_merge - 1.0)
        exp_temp_emit = torch.exp(
                self.temperature_emit - 1.0
            )
        if self.is_temperature_learning:
            return exp_temp_merge, exp_temp_emit
        else:
            return exp_temp_merge.detach(), exp_temp_emit.detach()

    def log_heavy(self):
        log = super().log_heavy()
        log[
            "log_merge_temperature"
        ] = self.logging_cache["temperature_merge"].flatten().tolist()
        log[
            "log_emit_temperature"
        ] = self.logging_cache["temperature_emit"].flatten().tolist()
        return log
