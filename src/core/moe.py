"""Dropless token-choice top-k mixture-of-experts feed-forward, on grouped matmuls.

Each of the T*k assignments gets its own row in a buffer grouped by expert, and each
SwiGLU projection is one grouped matmul over all experts (F.grouped_mm). Nothing is
dropped, so there is no capacity factor, train and eval route the same way, and right
padding in the downstream eval can't touch real tokens. The buffer has a fixed number
of rows (T*k plus per-expert alignment padding), so compile_blocks_full still traces a
block whole; only the group boundaries depend on the routing.

Expert weights are 3D parameters (num_experts, out, in), one per projection, stacked
from the dense ff_layer_fn partials so their init matches the dense layer's. Muon has to
treat them as a batch of matrices. FSDP2 shards them along the expert dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.model import GatedFeedForward, llm_random_weight_init

# rows per expert group are padded to a multiple of this, for the grouped GEMM kernels
ALIGN = 16


def load_balancing_loss(router_probs, expert_ids, num_experts):
    """Switch-style balance term on the routing; 1 under uniform routing."""
    one_hot = expert_ids.reshape(-1, 1) == torch.arange(
        num_experts, device=expert_ids.device
    )
    assignment_share = one_hot.float().mean(0)
    return num_experts * (router_probs.mean(0) * assignment_share).sum()


def grouped_linear(x, weight, ends):
    """x @ weight[e].T for each expert's row group; rows past ends[-1] are ignored."""
    if x.is_cuda:  # the CUDA kernels take bf16 only
        return F.grouped_mm(x.bfloat16(), weight.bfloat16().mT, offs=ends).type_as(x)
    return F.grouped_mm(x, weight.mT, offs=ends)


class MoE(nn.Module):
    # what forward() stores in self.stats, logged by the trainer as train/moe_<name>
    STAT_NAMES = ("balance_loss", "z_loss", "max_load")

    def __init__(
        self,
        ff_pre_act_fn,
        ff_post_act_fn,
        gate_fn,
        num_experts: int,
        topk: int,
        load_balancing_loss_factor: float,
        router_z_loss_factor: float,
    ):
        super().__init__()
        if not 0 < topk <= num_experts:
            raise ValueError(f"topk={topk} must be in [1, num_experts={num_experts}].")
        self.num_experts = num_experts
        self.topk = topk
        self.load_balancing_loss_factor = load_balancing_loss_factor
        self.router_z_loss_factor = router_z_loss_factor

        # built in the order the per-expert ModuleList used, so the init is unchanged
        experts = [
            GatedFeedForward(ff_pre_act_fn, ff_post_act_fn, gate_fn)
            for _ in range(num_experts)
        ]
        self.pre_act_weight = nn.Parameter(
            torch.stack([e.ff_pre_act.weight.detach() for e in experts])
        )
        self.gate_weight = nn.Parameter(
            torch.stack([e.gate.weight.detach() for e in experts])
        )
        self.post_act_weight = nn.Parameter(
            torch.stack([e.ff_post_act.weight.detach() for e in experts])
        )
        dmodel = self.pre_act_weight.shape[-1]
        self.router_weight = nn.Parameter(torch.empty(num_experts, dmodel))
        llm_random_weight_init(dmodel, scale=1)(self.router_weight)

    def expert_weights(self):
        return (self.pre_act_weight, self.gate_weight, self.post_act_weight)

    def group_by_expert(self, expert_ids):
        """Buffer row of each assignment, with experts' groups contiguous and padded to ALIGN.

        Returns (rows, ends, dest, load): a static row count, each group's end (int32),
        each assignment's unique row, and the number of assignments per expert.
        """
        one_hot = expert_ids.unsqueeze(1) == torch.arange(
            self.num_experts, device=expert_ids.device
        )
        load = one_hot.sum(0)
        place = (one_hot.long().cumsum(0) * one_hot).sum(1) - 1
        padded = (load + ALIGN - 1) // ALIGN * ALIGN
        ends = padded.cumsum(0)
        dest = (ends - padded)[expert_ids] + place
        # one ALIGN block per expert more than any ends[-1], so ends[-1] < rows always
        rows = expert_ids.numel() + self.num_experts * ALIGN
        return rows, ends.int(), dest, load

    def forward(self, x):
        tokens = x.reshape(-1, x.shape[-1])
        n_tokens, dmodel = tokens.shape
        logits = F.linear(tokens, self.router_weight).float()
        probs = logits.softmax(dim=-1)
        weights, expert_ids = probs.topk(self.topk, dim=-1)
        # token-major: token t owns assignments t*k..t*k+k-1
        expert_ids = expert_ids.reshape(-1)

        rows, ends, dest, load = self.group_by_expert(expert_ids)
        grouped = tokens.new_zeros(rows, dmodel).index_copy(
            0, dest, tokens.repeat_interleave(self.topk, dim=0)
        )
        hidden = F.silu(grouped_linear(grouped, self.gate_weight, ends))
        hidden = hidden * grouped_linear(grouped, self.pre_act_weight, ends)
        expert_outputs = grouped_linear(hidden, self.post_act_weight, ends)
        # index_select, not [dest]: the bf16 backward of advanced indexing is serial
        per_assignment = expert_outputs.index_select(0, dest)
        per_assignment = per_assignment.view(n_tokens, self.topk, dmodel)
        output = (per_assignment * weights.unsqueeze(-1).to(tokens.dtype)).sum(1)

        if self.training:
            balance = load_balancing_loss(probs, expert_ids, self.num_experts)
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            self.aux_loss = (
                self.load_balancing_loss_factor * balance
                + self.router_z_loss_factor * z_loss
            )
            # busiest expert's assignments over the mean (1 = balanced)
            max_load = load.max() * self.num_experts / expert_ids.numel()
            self.stats = torch.stack([balance, z_loss, max_load.float()]).detach()
        return output.reshape(x.shape)
