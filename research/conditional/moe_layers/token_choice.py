from typing import Optional

import torch

from research.conditional.utils.layer_manager import (
    LoggingLayer,
    time_measured,
)
from research.conditional.moe_layers.moe_gating import TokenGating, init_gate


class TokenChoiceFF(LoggingLayer):
    def __init__(
        self,
        dmodel: int,
        n_experts: int,
        capacity_factor: float,
        load_balancing_loss_weight: float,
        init_type: str,
        init_scale: float,
        expert_inner_function: LoggingLayer,
        doutput: Optional[int] = None,
        routing_top_k: int = 1,
        use_einsum: bool = False,
        get_router_values_from: str = "weights",
        moe_values_exp: Optional[int] = 1,
    ):
        """
        Args:
            dmodel: dimension of the input
            doutput: dimension of the output (default: dmodel)
            n_experts: number of experts
            expert_size: size of each expert
            capacity_factor: scalar that determines how many tokens can be assigned to each expert
            load_balancing_loss_weight: weight of the auxillary loss
            expert_logic: expert logic layer, takes input of shape (n_experts, capacity, dmodel) and returns output of shape (n_experts, capacity, dmodel)
        """
        super().__init__()

        self.dmodel = dmodel
        self.doutput = self.dmodel if doutput is None else doutput
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.expert_inner_function = expert_inner_function
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.moe_values_exp = (
            moe_values_exp
            if moe_values_exp != -1
            else torch.nn.Parameter(torch.tensor(1.0))
        )
        self.gate = None

        get_gate = init_gate(
            self, dmodel, get_router_values_from, init_scale, init_type, n_experts
        )
        self.router = TokenGating(
            dmodel=dmodel,
            n_experts=n_experts,
            capacity_factor=capacity_factor,
            load_balancing_loss_weight=load_balancing_loss_weight,
            init_type=init_type,
            init_scale=init_scale,
            routing_top_k=routing_top_k,
            use_einsum=use_einsum,
            get_gate=get_gate,
        )

    @time_measured("assign_tokens_to_input")
    def extract(self, x, tokens_per_expert_indices, tokens_per_expert_values):
        capacity = tokens_per_expert_indices.shape[0]
        indicies_reshaped = tokens_per_expert_indices.T.reshape(
            self.n_experts * capacity
        )
        values_reshaped = tokens_per_expert_values.T.reshape(
            self.n_experts * capacity, 1
        )
        experts_input = x[indicies_reshaped, :] * values_reshaped
        experts_input = experts_input.reshape(self.n_experts, capacity, self.dmodel)
        return experts_input

    @time_measured("assign_tokens_to_output")
    def merge(
        self,
        experts_output,
        masked_expert_gate,
        tokens_per_expert_indices,
        batch_size,
        seq_len,
        x,
    ):
        output = torch.zeros(
            batch_size * seq_len,
            self.doutput,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
        )
        output.index_add_(
            dim=0,
            index=tokens_per_expert_indices.T.flatten(),
            source=experts_output.reshape(
                self.n_experts * experts_output.shape[1], self.doutput
            ),
        )
        if self.moe_values_exp != 1.0 or not isinstance(self.moe_values_exp, float):
            masked_expert_gate **= self.moe_values_exp
        output *= masked_expert_gate.sum(dim=1, keepdim=True)
        output = output.reshape(batch_size, seq_len, self.doutput)
        return output

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        (
            tokens_per_expert_indices,
            tokens_per_expert_values,
            masked_expert_gate,
        ) = self.router(x)

        x = x.flatten(start_dim=0, end_dim=1)
        experts_input = self.extract(
            x, tokens_per_expert_indices, tokens_per_expert_values
        )
        experts_output = self.expert_inner_function(experts_input).to(x.dtype)
        output = self.merge(
            experts_output,
            masked_expert_gate,
            tokens_per_expert_indices,
            batch_size,
            seq_len,
            x,
        )
        return output
