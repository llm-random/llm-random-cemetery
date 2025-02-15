from lizrd.core import nn, misc
from research.conditional.moe_layers.continuous_moe import ContinuousMoeBaseClass
from research.conditional.utils.misc_tools import stable_softmax_temperature


class ContinuousMoEMergeDifferentlySimple(ContinuousMoeBaseClass):
    """
    Emits tokens with separate weights, instead of using the weights from the merging step.
    """

    def init_parameters(self):
        self.lin1 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.dm
            )
        )
        self.lin2 = nn.Parameter(
            misc.get_init_weight(
                (self.dm, self.n_experts, self.expert_size), fan_in=self.expert_size
            )
        )
        self.controller_merge = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )
        self.controller_emit = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_merge)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        emit_logits = misc.einsum("B S c d, d e -> B S e c", x, self.controller_emit)
        self.update_cache_for_logging("emit_logits", emit_logits)
        emit_weights = stable_softmax_temperature(emit_logits, self.temperature)
        self.update_cache_for_logging("emit_weights", emit_weights)
        return merge_weights, emit_weights
