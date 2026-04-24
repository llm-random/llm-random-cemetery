"""Check that MoE.forward_2 matches MoE.forward on outputs, losses, and gradients.

Run: pixi run python scripts/test_moe_forward_2.py
"""

import torch

from src.core.moe import MoE


def run_once(moe: MoE, x: torch.Tensor, use_v2: bool):
    # Fresh leaf tensor so gradients don't accumulate across runs
    x_leaf = x.detach().clone().requires_grad_(True)
    fn = moe.forward_2 if use_v2 else moe.forward
    out = fn(x_leaf)
    lb = moe.moe_load_balancing_loss
    rz = moe.router_z_loss
    # Scalar loss mixing everything so grads probe all code paths
    loss = out.square().sum() + lb + rz
    params = [
        moe.router_weight,
        moe.ff_pre_act_weight,
        moe.gate_weight,
        moe.ff_post_act_weight,
    ]
    grads = torch.autograd.grad(loss, [x_leaf, *params], retain_graph=False)
    return out.detach(), lb.detach(), rz.detach(), [g.detach() for g in grads]


def compare(name, a, b, atol=1e-6, rtol=1e-5):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    diff = (a - b).abs().max().item()
    print(f"  {name:30s} max|Δ|={diff:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def run_case(
    dmodel, dff, num_experts, topk, capacity_factor, normalize, batch, seq, seed
):
    print(
        f"\n[case] dmodel={dmodel} dff={dff} E={num_experts} topk={topk} "
        f"cap={capacity_factor} norm={normalize} batch={batch} seq={seq} seed={seed}"
    )
    torch.manual_seed(seed)
    moe = MoE(
        dmodel=dmodel,
        dff=dff,
        num_experts=num_experts,
        topk=topk,
        capacity_factor=capacity_factor,
        moe_load_balancing_loss_factor=0.01,
        moe_router_z_loss_factor=0.001,
        normalize_router_logits=normalize,
    ).double()
    moe.train()
    x = torch.randn(batch, seq, dmodel, dtype=torch.float64)

    out1, lb1, rz1, g1 = run_once(moe, x, use_v2=False)
    out2, lb2, rz2, g2 = run_once(moe, x, use_v2=True)

    ok = True
    ok &= compare("output", out1, out2)
    ok &= compare("lb_loss", lb1, lb2)
    ok &= compare("z_loss", rz1, rz2)
    labels = ["grad_x", "grad_router", "grad_ff_pre", "grad_gate", "grad_ff_post"]
    for label, ga, gb in zip(labels, g1, g2):
        ok &= compare(label, ga, gb)
    return ok


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    cases = [
        dict(dmodel=32, dff=64, num_experts=4, topk=1, capacity_factor=1.25,
             normalize=False, batch=2, seq=16, seed=0),
        dict(dmodel=32, dff=64, num_experts=4, topk=2, capacity_factor=1.25,
             normalize=True, batch=2, seq=16, seed=1),
        # Tight capacity so many tokens get dropped -> exercises dump-slot path
        dict(dmodel=24, dff=48, num_experts=8, topk=2, capacity_factor=0.3,
             normalize=True, batch=3, seq=20, seed=2),
        # Loose capacity so nothing gets dropped
        dict(dmodel=16, dff=32, num_experts=4, topk=2, capacity_factor=4.0,
             normalize=False, batch=2, seq=12, seed=3),
        # topk == num_experts
        dict(dmodel=16, dff=32, num_experts=3, topk=3, capacity_factor=1.0,
             normalize=True, batch=2, seq=10, seed=4),
    ]
    all_ok = True
    for c in cases:
        all_ok &= run_case(**c)
    print("\nALL OK" if all_ok else "\nFAILURES")
    raise SystemExit(0 if all_ok else 1)
