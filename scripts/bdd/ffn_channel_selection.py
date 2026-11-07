"""Choose which FFN channels of Llama-3.2-1B to keep, so the kept weights look like init.

The selection algorithm is the one in ~/bdd/notebooks/ffn_weight_distribution.ipynb
(section 4), carried over unchanged: for every layer, pick K of the 8192 FFN channels
(the same index set for gate_proj rows, up_proj rows and down_proj columns, i.e. whole
FFN neurons) minimising

    J(S) = 1/3 * sum_m W1(F_{m,S}, N(0, sigma_ref_{m,l})) / sigma_ref_{m,l}

against one of two init references:
    hf  -- HF LlamaConfig: N(0, 0.02) for all three matrices
    tt  -- torchtitan depth_init: gate N(0, 0.02), up/down N(0, 0.02 / sqrt(2(l+1)))

Search: best of N_RANDOM uniform subsets, then local search by channel swaps, then a
Frank-Wolfe lower bound on any K-subset.

Differences from the notebook, all deliberate:
  * every layer (the notebook only ran LAYERS = [0, 3, 6, 9, 12, 15]), because pruning
    the model needs an index set for each of the 16;
  * several keep fractions in one run (the notebook fixed KEEP = 0.7);
  * output is what src/projected_compression/pruning.py:prune() reads:
        {"dmodel_importances": [dmodel], "dff_importances": [n_layers x [dff]]}
    prune() keeps topk(dff_importances[l], target_dff), so a kept channel scores >= 1
    and a dropped one < 0.5. Within each group scores fall with the index, so topk
    returns the kept channels in their original order and dmodel in identity order --
    no permutation of the residual stream or of the FFN.

Usage (writes keep{50,60,70,80,90}_{hf,tt}.pt and a summary.json into --out-dir):
    python scripts/bdd/ffn_channel_selection.py --out-dir <dir>
    python scripts/bdd/ffn_channel_selection.py --out-dir <dir> --keep 0.7 --layers 0 15
"""

import argparse
import glob
import json
import math
import os
import time

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

MATS = ["gate_proj", "up_proj", "down_proj"]
TARGETS = ("hf", "tt")


# ---------------------------------------------------------------------------------------
# Algorithm: copied from the notebook (cell "class Problem ... def fw_gap").
# ---------------------------------------------------------------------------------------


def ref_std(target, m, l):
    if target == "hf" or m == "gate_proj":
        return 0.02
    return 0.02 / math.sqrt(2 * (l + 1))


class Problem:
    # Channel CDFs of matrix w on a grid + the reference N(0, sigma).
    def __init__(self, w, sigma, n_q=1024):
        self.sigma = sigma
        rows = w.sort(dim=1).values
        flat = rows.flatten().sort().values
        data_q = flat[torch.linspace(0, flat.numel() - 1, n_q).round().long()]
        ref_q = sigma * torch.linspace(-6, 6, n_q)
        e = torch.cat([data_q, ref_q]).unique()  # sorted, contains the data min and max
        self.edges = e
        self.C = (
            torch.searchsorted(
                rows, e.expand(rows.shape[0], -1).contiguous(), right=True
            ).float()
            / rows.shape[1]
        )
        self.F = torch.special.ndtr(e.double() / sigma).float()
        self.D = torch.cat([e.diff(), torch.zeros(1)])
        za, zb = e[0].item() / sigma, e[-1].item() / sigma
        phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
        # int_{-inf}^{e0} Phi(x/s) dx  +  int_{e_last}^{inf} (1 - Phi(x/s)) dx
        self.tail = sigma * (phi(za) + za * Phi(za)) + sigma * (
            phi(zb) - zb * (1 - Phi(zb))
        )

    def w1(self, cdf):  # cdf: [..., E], the subset's CDF on the grid
        return (cdf - self.F).abs() @ self.D + self.tail


def objective(probs, sums, k):
    return sum(p.w1(s / k) / p.sigma for p, s in zip(probs, sums)) / len(probs)


def random_subsets(n, k, r, gen):
    idx = torch.rand(r, n, generator=gen).argsort(dim=1)[:, :k]
    return torch.zeros(r, n).scatter_(1, idx, 1.0)


def local_search(probs, sel, t=256, t_max=1024, m_exact=64, max_steps=3000):
    k = int(sel.sum())
    sums = [sel.float() @ p.C for p in probs]
    cost = objective(probs, sums, k).item()
    history = [cost]
    while len(history) < max_steps:
        g = sum(
            p.C @ (torch.sign(s / k - p.F) * p.D) / p.sigma
            for p, s in zip(probs, sums)
        )
        g_in = torch.where(sel, torch.inf, g)  # candidates to add: smallest gradient
        g_out = torch.where(sel, g, -torch.inf)  # candidates to drop: largest gradient
        ins, outs = g_in.argsort()[:t], g_out.argsort(descending=True)[:t]
        good = g_in[ins] < g_out[outs]
        ins, outs = ins[good], outs[good]
        new_cost = math.inf
        if t >= 1 and len(ins):
            new_sums = [
                s + p.C[ins].sum(0) - p.C[outs].sum(0) for p, s in zip(probs, sums)
            ]
            new_cost = objective(probs, new_sums, k).item()
        if new_cost >= cost - 1e-9:
            # the linear step does not help (|.| in W1 is not differentiable): check all
            # swap pairs among the m_exact best candidates exactly and take the best
            ins = g_in.argsort()[:m_exact]
            outs = g_out.argsort(descending=True)[:m_exact]
            pair = [
                (s[None, None] + p.C[ins][:, None] - p.C[outs][None, :])
                for p, s in zip(probs, sums)
            ]
            costs = objective(probs, pair, k)
            best = costs.argmin().item()
            if costs.flatten()[best].item() >= cost - 1e-9:
                break
            ins = ins[best // m_exact : best // m_exact + 1]
            outs = outs[best % m_exact : best % m_exact + 1]
            new_sums = [
                s + p.C[ins].sum(0) - p.C[outs].sum(0) for p, s in zip(probs, sums)
            ]
            new_cost = objective(probs, new_sums, k).item()
            t = max(t // 2, 1)
        else:
            t = min(2 * t, t_max)
        sel = sel.clone()
        sel[ins], sel[outs] = True, False
        sums, cost = new_sums, new_cost
        history.append(cost)
    return sel, history


def fw_gap(probs, sel):
    # J is convex over the relaxation x in [0,1]^n, sum x = K, so J* >= J(S) - gap.
    k = int(sel.sum())
    sums = [sel.float() @ p.C for p in probs]
    g = sum(
        p.C @ (torch.sign(s / k - p.F) * p.D) / p.sigma for p, s in zip(probs, sums)
    ) / (len(probs) * k)
    return (g[sel].sum() - g.topk(k, largest=False).values.sum()).item()


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------


def keep_count(keep, ffn):
    return round(keep * ffn)


def to_importances(keep_masks, dmodel):
    """prune()'s input: kept channels score in [1, 2), dropped ones in [0, 0.5)."""
    dff = []
    for sel in keep_masks:
        n = sel.numel()
        order = 1.0 - torch.arange(n, dtype=torch.float32) / n  # (0, 1], falls with index
        dff.append(torch.where(sel, 1.0 + order, 0.5 * order))
    return {
        "dmodel_importances": 1.0 - torch.arange(dmodel, dtype=torch.float32) / dmodel,
        "dff_importances": dff,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--keep", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    path = snapshot_download(args.model, allow_patterns=["*.safetensors", "config.json"])
    st = safe_open(glob.glob(f"{path}/*.safetensors")[0], "pt")
    cfg = json.load(open(f"{path}/config.json"))
    n_layers, dmodel, ffn = (
        cfg["num_hidden_layers"],
        cfg["hidden_size"],
        cfg["intermediate_size"],
    )
    layers = args.layers if args.layers else list(range(n_layers))

    def channels(l, m):
        w = st.get_tensor(f"model.layers.{l}.mlp.{m}.weight").float()
        return w.T.contiguous() if m == "down_proj" else w

    # One generator for the whole run, consumed in a fixed order, so a rerun with the
    # same arguments gives the same subsets.
    gen = torch.Generator().manual_seed(args.seed)
    masks = {(keep, t): [None] * n_layers for keep in args.keep for t in args.targets}
    summary = []
    for l in layers:
        W = {m: channels(l, m) for m in MATS}
        for t in args.targets:
            probs = [Problem(W[m], ref_std(t, m, l)) for m in MATS]
            full = objective(probs, [p.C.sum(0) for p in probs], ffn).item()
            for keep in args.keep:
                k = keep_count(keep, ffn)
                t0 = time.time()
                cand = random_subsets(ffn, k, args.n_random, gen)
                rand_cost = objective(probs, [cand @ p.C for p in probs], k)
                best = rand_cost.argmin()
                opt, history = local_search(probs, cand[best].bool())
                masks[keep, t][l] = opt
                row = dict(
                    layer=l,
                    target=t,
                    keep=keep,
                    k=k,
                    J_full=full,
                    J_rand_median=rand_cost.median().item(),
                    J_best_rand=rand_cost.min().item(),
                    J_opt=history[-1],
                    J_lower_bound=history[-1] - fw_gap(probs, opt),
                    steps=len(history) - 1,
                    seconds=round(time.time() - t0, 1),
                )
                summary.append(row)
                print(
                    f"layer {l:2d} {t} keep={keep:.0%}: J full={full:.4f} "
                    f"best rand={row['J_best_rand']:.4f} opt={row['J_opt']:.4f} "
                    f"LB={row['J_lower_bound']:.4f} ({row['steps']} steps, {row['seconds']}s)",
                    flush=True,
                )

    for (keep, t), per_layer in masks.items():
        name = f"keep{round(keep * 100)}_{t}"
        if any(m is None for m in per_layer):
            # A partial run (--layers) cannot drive prune(), which needs every layer.
            torch.save(
                {l: m.nonzero().squeeze(1) for l, m in enumerate(per_layer) if m is not None},
                f"{args.out_dir}/{name}_indices_partial.pt",
            )
            continue
        assert all(int(m.sum()) == keep_count(keep, ffn) for m in per_layer)
        torch.save(to_importances(per_layer, dmodel), f"{args.out_dir}/{name}.pt")
        torch.save(
            {l: m.nonzero().squeeze(1) for l, m in enumerate(per_layer)},
            f"{args.out_dir}/{name}_indices.pt",
        )
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump(dict(args=vars(args), rows=summary), f, indent=1)
    print("written:", sorted(os.listdir(args.out_dir)))


if __name__ == "__main__":
    main()
