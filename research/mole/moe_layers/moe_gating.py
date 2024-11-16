from typing import Any, Union, Literal
import torch
from fancy_einsum import einsum
from plotly import express as px

from lizrd.support.logging import make_histogram
from lizrd.train import checkpointing
import torch.nn.functional as F

from research.mole.moe_layers.load_balancing_loss import calculate_load_balancing_loss, calculate_biased_balancing_loss
from lizrd.core.misc import LoggingLayer, measure_time
from lizrd.core.initialization import get_init_fun
from research.conditional.moe_layers.moe_gating import MoeGating


class InputWiseRouterBias:
    def __init__(self) -> None:
        self.router_target_bias: torch.Tensor = None

    def set_router_target_bias(self, router_target_bias: torch.Tensor):
        self.router_target_bias = router_target_bias

    def remove_router_target_bias(self):
        self.router_target_bias = None


class ManagerMaskSetter:
    def __init__(self, model: torch.nn.Module, router_target_bias: torch.Tensor):
        self.router_target_bias = router_target_bias
        self._layers: list[InputWiseRouterBias] = []
        for _, layer in model.named_modules():
            if isinstance(layer, InputWiseRouterBias):
                self._layers.append(layer)
        if len(self._layers) == 0:
            raise Exception("No InputWiseMask modules in provided model")

    def __enter__(self):
        for layer in self._layers:
            layer.set_router_target_bias(self.router_target_bias)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for layer in self._layers:
            layer.remove_router_target_bias()

class TokenGatingBiased(MoeGating, InputWiseRouterBias):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        biased_balancing_loss_weight: float,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        **kwargs,
    ):
        MoeGating().__init__(
            dmodel=dmodel,
            n_experts=n_experts,
            group_by_batch=False,
            softmax_ungrouped=False,
            softmax_over="experts",
            use_torch_bmm=not use_einsum,
            **kwargs,
        )
        InputWiseRouterBias.__init__(self)

        self.capacity_factor = capacity_factor
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.biased_balancing_loss_weight = biased_balancing_loss_weight
        self.use_einsum = use_einsum
        self.routing_top_k = routing_top_k
        self.router_target_bias: torch.Tensor = None
        self.biased_balancing_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        # x is (batch, seq_len, dmodel)
        batch_size, seq_len, _ = x.shape
        n_tokens = batch_size * seq_len
        capacity = min(
            int(self.capacity_factor * n_tokens * self.routing_top_k / self.n_experts),
            n_tokens,
        )
        self.update_cache_for_logging("n_tokens", torch.Tensor([n_tokens]))

        gate_out = self.calculate_gate(x, batch_size, seq_len).T
        assert gate_out.shape == (n_tokens, self.n_experts)

        with measure_time(self, "choose_expert"):
            expert_index, expert_gate = self.calculate_topk(
                gate_out, self.routing_top_k
            )

        self.update_cache_for_logging("gate_softmax_values", expert_gate)
        self.update_cache_for_logging("max_indices", expert_index)

        return self.apply_capacity(capacity, expert_index, gate_out, n_tokens)

    def calculate_balancing_loss(self, gate_out, expert_mask):
        with measure_time(self, "calculate aux loss"):
            tokens_per_expert = expert_mask.sum(dim=0, dtype=gate_out.dtype)
            load_balancing_loss = calculate_load_balancing_loss(
                self.load_balancing_loss_weight,
                gate_out,
                tokens_per_expert,
                use_einsum=self.use_einsum,
                router_target_bias=self.router_target_bias
            )
            biased_balancing_loss = calculate_biased_balancing_loss(
                gate_out,
                router_target_bias=self.router_target_bias,
                loss_fn=self.biased_balancing_loss_fn
                alpha=self.biased_balancing_loss_weight,
            )
        if "load_balancing_losses" not in self.forward_pass_cache:
            self.forward_pass_cache["load_balancing_losses"] = [load_balancing_loss]
        else:
            self.forward_pass_cache["load_balancing_losses"].append(load_balancing_loss)
        if "biased_balancing_loss" not in self.forward_pass_cache:
            self.forward_pass_cache["biased_balancing_loss"] = [biased_balancing_loss]
        else:
            self.forward_pass_cache["biased_balancing_loss"].append(biased_balancing_loss)
        self.update_cache_for_logging("tokens_per_expert", tokens_per_expert)
        self.update_cache_for_logging("load_balancing_loss", load_balancing_loss)

    def apply_capacity(self, capacity, expert_index, gate_out, n_tokens):
        # create a mask telling if a token is assigned to an expert
        with measure_time(self, "create_expert_mask"):
            expanded_expert_mask = F.one_hot(expert_index, num_classes=self.n_experts)
            assert expanded_expert_mask.shape == (
                n_tokens,
                self.routing_top_k,
                self.n_experts,
            )
            expert_mask = expanded_expert_mask.sum(dim=1)
            assert expert_mask.shape == (n_tokens, self.n_experts)

        # now apply fixed capacity: for a given expert we can have only capacity tokens
        with measure_time(self, "experts_lists"):
            (
                top_tokens_per_expert_values,
                top_tokens_per_expert_indices,
            ) = expert_mask.topk(k=capacity, dim=0)

        self.log_dropped_tokens(
            top_tokens_per_expert_values,
            top_tokens_per_expert_indices,
            expert_mask,
            n_tokens,
        )
        # from a list of finally chosen tokens, create a mask with their respective values
        expert_values = (
            torch.gather(gate_out, 0, top_tokens_per_expert_indices)
            * top_tokens_per_expert_values
        )
        self.calculate_balancing_loss(gate_out, expert_mask)
        return top_tokens_per_expert_indices, expert_values

    def log_dropped_tokens(
        self,
        top_tokens_per_expert_values,
        top_tokens_per_expert_indices,
        expert_mask,
        n_tokens,
    ):
        # TODO this below is just for logging, we maybe should remove it
        with measure_time(self, "create_truncated_mask"):
            truncated_expert_mask = torch.zeros_like(expert_mask)
            truncated_expert_mask.scatter_(
                dim=0,
                index=top_tokens_per_expert_indices,
                src=top_tokens_per_expert_values,
            )
        n_selected_tokens = truncated_expert_mask.sum().item()
        self.update_cache_for_logging(
            "dropped_tokens_ratio",
            ((n_tokens * self.routing_top_k) - n_selected_tokens)
            / (n_tokens * self.routing_top_k),
        )

    def log_light(self):
        return {
            "dropped_tokens_ratio": self.logging_cache["dropped_tokens_ratio"],
            "load_balancing_loss": self.logging_cache["load_balancing_loss"],
            "z_loss": self.logging_cache["z_loss"],
        }

    def log_heavy(self):
        return {
            "gate_softmax_all_values": make_histogram(
                self.logging_cache["gate_softmax_all_values"].flatten()  # move
            ),
            "tokens_per_expert_counts": make_histogram(
                self.logging_cache["tokens_per_expert"]
            ),
        }


def make_heatmap(tensor, expert_num, **kwargs):
    logits_for_expert = tensor[expert_num]
    batch_size, seq_len = logits_for_expert.shape
    flatten_dist = logits_for_expert.flatten()
    dist_for_expert = torch.softmax(flatten_dist.float(), dim=-1)
    dist_for_expert = dist_for_expert.reshape(batch_size, seq_len)
    return px.imshow(dist_for_expert.detach().cpu().numpy(), **kwargs)
