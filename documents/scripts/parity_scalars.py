#!/usr/bin/env python
"""ame_bf16_scalars_plan.md, Step 2, Stages A and B: the per-head scalars'
gradients over the 16 held-out groups, from `parity_numerics.py scalars`.

    python documents/scripts/parity_scalars.py scalars_0_..._7.pt scalars_8_..._15.pt [--json OUT]

Stage A, per element of tau, alpha and beta (720 numbers: 12 layers x 12 heads,
alpha and beta with 2 coefficients each), for production and AME-F against the
fp32 oracle, over the 16 groups of 512 sequences:

- bias: the mean error over the RMS of the true gradient;
- t: the mean error over its standard error (16 groups) -- |t| >> 2 on many
  elements means a systematic error, not noise;
- the summed-error projection <sum e, sum g> / |sum g|^2 per parameter kind;
- the error of one group (512 sequences) against the error of the sum over all
  16 (8,192): a bias grows linearly with tokens while a stationary true gradient
  grows like sqrt(N), so a biased path's relative error *rises* with N.

Stage B: 16 AdamW updates from the checkpoint's moments, one per group, with
each path's gradients; the update's relative error and sign agreement against
the oracle's.
"""

import argparse
import json
import math

import torch

KINDS = ("tau", "alpha", "beta")


def load(paths):
    merged = None
    for path in paths:
        part = torch.load(path, map_location="cpu", weights_only=False)
        if merged is None:
            merged = part
            continue
        merged["groups"] = merged["groups"] + part["groups"]
        for variant, grads in part["grads"].items():
            for k in KINDS:
                merged["grads"][variant][k] = torch.cat([merged["grads"][variant][k], grads[k]])
        for key in ("params", "adam", "lr", "betas", "eps"):
            if not merged.get(key) and part.get(key):
                merged[key] = part[key]
    order = sorted(range(len(merged["groups"])), key=lambda i: merged["groups"][i])
    for variant in merged["grads"]:
        for k in KINDS:
            merged["grads"][variant][k] = merged["grads"][variant][k][order]
    merged["groups"] = [merged["groups"][i] for i in order]
    return merged


def stage_a(data, variant, oracle="ame_ref32"):
    out = {}
    G = len(data["groups"])
    for k in KINDS:
        g_o = data["grads"][oracle][k].reshape(G, -1)  # [groups, elements]
        g_v = data["grads"][variant][k].reshape(G, -1)
        e = g_v - g_o
        rms_true = g_o.pow(2).mean(0).sqrt().clamp_min(1e-30)
        mean_e = e.mean(0)
        sd_e = e.std(0, unbiased=True).clamp_min(1e-30)
        bias = mean_e / rms_true
        t = mean_e / (sd_e / math.sqrt(G))
        sum_e, sum_g = e.sum(0), g_o.sum(0)
        per_group_rel = (e.norm(dim=1) / g_o.norm(dim=1)).tolist()
        out[k] = {
            "elements": g_o.shape[1],
            "bias_over_rms_true": {"median_abs": float(bias.abs().median()), "max_abs": float(bias.abs().max())},
            "t": {
                "median_abs": float(t.abs().median()),
                "frac_abs_gt_2": float((t.abs() > 2).float().mean()),
                "frac_abs_gt_4": float((t.abs() > 4).float().mean()),
            },
            "summed_error_projection": float((sum_e * sum_g).sum() / (sum_g * sum_g).sum()),
            "rel_err_one_group_mean": sum(per_group_rel) / G,
            "rel_err_sum_of_16": float(sum_e.norm() / sum_g.norm()),
        }
    return out


def adamw_replay(theta, m, v, step, grads, lr, betas, eps):
    """Plain AdamW (wd 0, as these parameters have), one update per gradient."""
    b1, b2 = betas
    theta, m, v = theta.clone(), m.clone(), v.clone()
    for i, g in enumerate(grads):
        t = step + i + 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        theta = theta - lr * mhat / (vhat.sqrt() + eps)
    return theta


def stage_b(data, variant, oracle="ame_ref32"):
    out = {}
    for k in KINDS:
        theta0 = torch.stack(data["params"][k])  # [layers, ...]
        m0 = torch.stack([s["exp_avg"] for s in data["adam"][k]])
        v0 = torch.stack([s["exp_avg_sq"] for s in data["adam"][k]])
        step0 = data["adam"][k][0]["step"]
        upd = {}
        for name in (oracle, variant):
            grads = [g for g in data["grads"][name][k]]
            upd[name] = adamw_replay(theta0, m0, v0, step0, grads, data["lr"], data["betas"], data["eps"]) - theta0
        du_o, du_v = upd[oracle].flatten(), upd[variant].flatten()
        out[k] = {
            "update_rel_err": float((du_v - du_o).norm() / du_o.norm()),
            "sign_agreement": float((torch.sign(du_v) == torch.sign(du_o)).float().mean()),
            "update_rms_oracle": float(du_o.pow(2).mean().sqrt()),
            "param_rms": float(theta0.pow(2).mean().sqrt()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    data = load(args.paths)
    report = {"groups": data["groups"], "lr": data.get("lr")}
    for variant in [v for v in data["grads"] if v != "ame_ref32"]:
        report[variant] = {
            "stage_a": stage_a(data, variant),
            "stage_b": stage_b(data, variant) if data.get("params") else {k: {"update_rel_err": float("nan"), "sign_agreement": float("nan")} for k in KINDS},
        }
        print(f"=== {variant} against the fp32 oracle, {len(data['groups'])} groups of 512")
        for k in KINDS:
            a, b = report[variant]["stage_a"][k], report[variant]["stage_b"][k]
            print(
                f"  {k:<5} bias/RMS median {a['bias_over_rms_true']['median_abs']:.3f} max {a['bias_over_rms_true']['max_abs']:.3f}; "
                f"|t| median {a['t']['median_abs']:.2f}, >2: {a['t']['frac_abs_gt_2']:.2f}, >4: {a['t']['frac_abs_gt_4']:.2f}; "
                f"proj {a['summed_error_projection']:+.4f}; rel err one group {a['rel_err_one_group_mean']:.3f} -> sum of 16 {a['rel_err_sum_of_16']:.3f}; "
                f"Adam x16: update err {b['update_rel_err']:.3f}, sign agreement {b['sign_agreement']:.3f}"
            )
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
