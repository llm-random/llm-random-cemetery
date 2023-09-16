import dataclasses
from typing import Union
import itertools

import einops
import numpy as np
import torch
from plotly import express as px

from lizrd.core import misc, nn
from lizrd.support.logging import make_histogram
from research.conditional.utils.misc_tools import stable_softmax_temperature, entropy
from research.conditional.utils.layer_manager import LoggingLayer, measure_time


@dataclasses.dataclass(eq=False, repr=False)
class ContinuousMoeBaseClass(LoggingLayer):
    """
    1. Groups tokens into groups of fixed size,
    2. Each expert independently aggregates tokens within a group (aggregate means take a weighted combination, weights sum to 1) into a single token,
    3. Each expert processes the token constructed above to output a token of size dmodel
    4. The mapped token is then redistributed to the original tokens, with weights determined by the expert's weighting from step 2.
    """

    dm: int
    dff: int
    n_experts: int
    group_size: int
    sparsity_dim: int
    temperature: float
    expert_size: Union[int, None]
    use_opt_einsum: bool = False
    flop_matched: bool = False

    def __post_init__(self):
        super().__init__()
        if self.flop_matched:
            assert (
                self.dff == 4 * self.dm
            ), f"dff = {self.dff} is not equal to 4*dm = {4*self.dm} as in vanilla transformer"
            self.dff *= self.group_size
        if self.expert_size is None:
            assert (
                self.dff % self.n_experts == 0
            ), f"dff = {self.dff} is not divisible by n_experts = {self.n_experts}"
            print(
                f"expert_size is None, setting it to dff // n_experts = {self.dff // self.n_experts}"
            )
            self.expert_size = self.dff // self.n_experts
        self.init_parameters()

    def forward(self, x):
        with measure_time(self, "reshape_into_token_groups"):
            x = self.reshape_into_token_groups(x)
        with measure_time(self, "get_merge_and_emit_weights"):
            merge_weights, emit_weights = self.get_merge_and_emit_weights(x)
        with measure_time(self, "merge_map_emit"):
            x = self.merge_map_emit(x, merge_weights, emit_weights)
        with measure_time(self, "reshape_into_original"):
            x = self.reshape_into_original(x)
        return x

    def reshape_into_token_groups(self, x):
        """
        :param x: normal input tensor of shape (B, S, dmodel)
        :return: x reshaped so that one of dimensions is split into groups of size self.group_size, (the dimension is determined by self.sparsity_dim)
        """
        # we want to split the input into groups of size self.group_size according to sparsity_dim
        if self.sparsity_dim == 0:
            # gather tokens from the same position in each sequence (mixes data from different examples within a batch)
            x = einops.rearrange(x, "(B g) S d -> B S g d", g=self.group_size)
        elif self.sparsity_dim == 1:
            # gather tokens from the same sequence (does not mix data from different examples within a batch)
            x = einops.rearrange(x, "B (S g) d -> B S g d", g=self.group_size)
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return x

    def get_merge_and_emit_weights(self, x):
        merge_logits = misc.einsum("B S g d, d e -> B S e g", x, self.controller)
        self.update_cache_for_logging("merge_logits", merge_logits)
        merge_weights = stable_softmax_temperature(merge_logits, self.temperature)
        self.update_cache_for_logging("merge_weights", merge_weights)
        return merge_weights, merge_weights

    def merge_map_emit(self, x, merge_weights, emit_weights):
        # Reorder the dimensions so that the calculations are faster
        input_order = "B S g d"
        original_x = x.clone()
        for _ in range(20):
            for output_order in itertools.permutations(["B", "S", "g", "d"]):
                x = einops.rearrange(
                    original_x, f"{input_order} -> {' '.join(output_order)}"
                ).clone()
                with measure_time(self, f"merge_and_process_{'_'.join(output_order)}"):
                    x = misc.einsum(
                        f"{' '.join(output_order)}, B S e c, d e f -> B S e f",
                        x,
                        merge_weights,
                        self.lin1,
                        use_opt_einsum=self.use_opt_einsum,
                    )
                with measure_time(self, f"relu_{'_'.join(output_order)}"):
                    x = torch.relu_(x)
                with measure_time(self, f"process_and_emit_{'_'.join(output_order)}"):
                    x = misc.einsum(
                        "B S e f, d e f, B S e c -> B S c d",
                        x,
                        self.lin2,
                        emit_weights,
                        use_opt_einsum=self.use_opt_einsum,
                    )
        return x

    def reshape_into_original(self, x):
        if self.sparsity_dim == 0:
            out = einops.rearrange(x, "B S g d -> (B g) S d")
        elif self.sparsity_dim == 1:
            out = einops.rearrange(x, "B S g d -> B (S g) d")
        else:
            raise NotImplementedError("sparsity_dim must be 0 or 1")
        return out

    def init_parameters(self):
        # lin1 is parameter, one dimension for experts of size dmodel to dff/n_experts
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
        # controller: send a token of size dmodel to n_experts scalars
        self.controller = nn.Parameter(
            misc.get_init_weight((self.dm, self.n_experts), fan_in=self.dm)
        )

    def log_light(self):
        return {}

    def log_heavy(self):
        log = {}
        if self.group_size == 1:
            return log


        instr_names = list(self.logging_cache["time"].keys())
        instr_times = list(self.logging_cache["time"].values())
        merge_maps = []
        merge_map_names = []
        for instr_name, instr_time in zip(instr_names, instr_times):
            if "merge_and_process" in instr_name:
                assert isinstance(instr_time, list)
                assert isinstance(instr_time[0], float)
                mean = np.mean(instr_time)
                assert isinstance(mean, float)
                merge_maps.append(mean)
                merge_map_names.append(instr_name)

        merge_best_id = np.argmin(merge_maps)
        merge_best = merge_maps[merge_best_id]
        merge_best_signature = merge_map_names[merge_best_id]

        merge_worst_id = np.argmax(merge_maps)
        merge_worst = merge_maps[merge_worst_id]
        merge_worst_signature = merge_map_names[merge_worst_id]

        log[f"merge_and_process/best_{merge_best_signature}"] = merge_best
        log[f"merge_and_process/worst_{merge_worst_signature}"] = merge_worst
        default_time = np.mean(self.logging_cache["time"]["merge_and_process_B_S_g_d"])
        log[f"merge_and_process/default_time"] = default_time
        log[f"merge_and_process/best_to_default_ratio"] = merge_best / default_time
        log[f"merge_and_process/best_to_worst_ratio"] = merge_best / merge_worst
        # log process_and_emit time
        log["process_and_emit_time"] = self.logging_cache["time"][
            "process_and_emit_B_S_g_d"
        ]
        log["best_merge_to_emit_ratio"] = (
            merge_best / self.logging_cache["time"]["process_and_emit_B_S_g_d"]
        )
        log["worst_merge_to_emit_ratio"] = (
            merge_worst / self.logging_cache["time"]["process_and_emit_B_S_g_d"]
        )

        times_fig = px.bar(x=instr_names, y=instr_times)
        log["forward_pass_times"] = times_fig

        return log


class ContinuousMoE(ContinuousMoeBaseClass):
    pass
