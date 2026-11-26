"""The dropless MoE layer (src/core/moe.py): grouping, outputs and grads, padding, balance loss."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F

from src.core.model import Linear, llm_random_weight_init
from src.core.moe import ALIGN, MoE, load_balancing_loss

DMODEL, DFF = 8, 16


def _moe(num_experts, topk):
    def linear(in_features, out_features):
        init = partial(llm_random_weight_init, scale=1)
        return partial(Linear, in_features, out_features, partial_init_fn=init)

    torch.manual_seed(0)
    return MoE(
        ff_pre_act_fn=linear(DMODEL, DFF),
        ff_post_act_fn=linear(DFF, DMODEL),
        gate_fn=linear(DMODEL, DFF),
        num_experts=num_experts,
        topk=topk,
        load_balancing_loss_factor=0.01,
        router_z_loss_factor=0.001,
    )


def _reference(moe, x):
    """Every token through its top-k experts, one at a time: what the layer must equal."""
    tokens = x.reshape(-1, DMODEL)
    probs = F.linear(tokens, moe.router_weight).float().softmax(-1)
    weights, ids = probs.topk(moe.topk, dim=-1)
    gate = torch.einsum("td,tkfd->tkf", tokens, moe.gate_weight[ids])
    pre = torch.einsum("td,tkfd->tkf", tokens, moe.pre_act_weight[ids])
    out = torch.einsum("tkf,tkdf->tkd", F.silu(gate) * pre, moe.post_act_weight[ids])
    return (out * weights.unsqueeze(-1)).sum(1).reshape(x.shape)


def test_groups_are_contiguous_padded_and_unique():
    moe = _moe(num_experts=4, topk=2)
    expert_ids = torch.randint(0, 4, (64,))
    rows, ends, dest, load = moe.group_by_expert(expert_ids)
    assert rows == 64 + 4 * ALIGN and ends[-1] < rows
    assert dest.unique().numel() == 64
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    assert ((ends - starts) % ALIGN == 0).all()
    for expert in range(4):
        mine = dest[expert_ids == expert]
        assert mine.numel() == load[expert]
        # contiguous from the group's start, in token order
        assert mine.tolist() == list(
            range(starts[expert], starts[expert] + mine.numel())
        )


@pytest.mark.parametrize("training", [True, False])
def test_outputs_and_grads_match_the_per_token_reference(training):
    moe = _moe(num_experts=4, topk=2)
    moe.train(training)
    x = torch.randn(3, 7, DMODEL)
    results = []
    for fn in (moe, partial(_reference, moe)):
        moe.zero_grad()
        xi = x.clone().requires_grad_()
        out = fn(xi)
        out.square().sum().backward()
        results.append((out, xi.grad, [p.grad.clone() for p in moe.parameters()]))
    (out, dx, grads), (ref, ref_dx, ref_grads) = results
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(dx, ref_dx)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad)


def test_nothing_is_dropped_when_every_token_wants_one_expert():
    moe = _moe(num_experts=4, topk=1)
    with torch.no_grad():
        moe.router_weight.zero_()
        moe.router_weight[0, 0] = 10.0
    x = torch.randn(2, 8, DMODEL)
    x[..., 0] = x[..., 0].abs() + 1
    torch.testing.assert_close(moe(x), _reference(moe, x))
    assert moe.stats[2].item() == pytest.approx(4.0)  # all 16 on one of 4 experts


@pytest.mark.parametrize("training", [True, False])
def test_right_padding_cannot_change_real_tokens(training):
    moe = _moe(num_experts=4, topk=2)
    moe.train(training)
    x = torch.randn(2, 8, DMODEL)
    alone = moe(x[:1, :5])
    padded = torch.cat([x[:1, :5], torch.full((1, 3, DMODEL), 5.0)], dim=1)
    batched = moe(torch.cat([padded, x[1:]]))
    torch.testing.assert_close(batched[:1, :5], alone)


def test_the_balance_loss_is_at_its_minimum_for_uniform_routing():
    experts, tokens = 4, 32
    logits = torch.zeros(tokens, experts, requires_grad=True)
    round_robin = torch.arange(tokens) % experts
    uniform = load_balancing_loss(logits.softmax(-1), round_robin, experts)
    assert uniform.item() == pytest.approx(1.0)
    uniform.backward()
    assert logits.grad.abs().max() < 1e-7  # a stationary point
    # a skew costs more; full collapse onto one expert costs num_experts
    all_zero = torch.zeros(tokens, dtype=torch.long)
    skewed = torch.zeros(tokens, experts)
    skewed[:, 0] = 0.5
    assert load_balancing_loss(skewed.softmax(-1), all_zero, experts) > 1.0
    collapsed = F.one_hot(all_zero, experts).float()
    assert load_balancing_loss(collapsed, all_zero, experts).item() == pytest.approx(
        experts
    )
