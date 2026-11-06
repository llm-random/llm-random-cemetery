#!/usr/bin/env python
"""Parity audit, Phase 2: read the training gates and apply the plan's rule.

    python documents/scripts/parity_gates.py [--set 300|220] [--json OUT]

Fetches every gate of W&B project ideas_cv/nano-ame-parity by run name (cached
under $PARITY_GATES_CACHE, default ~/.cache/nano/parity_gates), then:

- the paired per-step train-loss gap of each gate against G0a (same batches,
  same step);
- the floor F: the larger of |G0b - G0a| and |G0c - G0a|, as the mean over the
  last 50 steps, and the same for final_ce;
- the plan's "harmful at N steps": the mean gap over the last 50 steps is at
  least max(3F, 0.005), at least 45 of those 50 steps share its sign, and the
  gate's final_ce agrees in sign;
- grad-norm traces: the step-10 spike and the post-warmup level (comparison
  doc's F3 and F4).

`train/loss` is the Trainer's bf16 `reported_loss` (quantized to 2^-6 near 3 and
2^-5 above 4), and `final_ce` is the mean of 16 bf16 batch losses; the floors
include that quantization, which is why they are measured, not assumed.
"""

import argparse
import json
import os

import numpy as np

PROJECT = "ideas_cv/nano-ame-parity"
CACHE = os.environ.get(
    "PARITY_GATES_CACHE", os.path.expanduser("~/.cache/nano/parity_gates")
)

SETS = {
    "300": {
        "base": "parity_g0a_s2ps",
        "floor": ["parity_g0b_s2ps", "parity_g0c_s2ps_mb16"],
        "gates": ["parity_g1_s2ps_mpnull"],
        "dense_base": "parity_gd0a_s1d",
        "dense_floor": ["parity_gd0b_s1d"],
    },
    "220": {
        "base": "parity_g0a_s2ps_220",
        "floor": ["parity_g0b_s2ps_220", "parity_g0c_s2ps_mb16_220"],
        "gates": ["parity_gref_s2ps_220"],
        "dense_base": None,
        "dense_floor": [],
    },
}


def load(name, refresh=False):
    import pandas as pd

    path = os.path.join(CACHE, f"{name}.parquet")
    meta_path = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path), json.load(open(meta_path))
    import wandb

    api = wandb.Api(timeout=180)
    runs = [r for r in api.runs(PROJECT, filters={"display_name": name})]
    finished = [r for r in runs if r.state == "finished"]
    if not finished:
        raise RuntimeError(f"{name}: no finished run ({[r.state for r in runs]})")
    run = sorted(finished, key=lambda r: r.created_at)[-1]
    # WandbLogger files every key twice, under steps/ and tokens/.
    keys = ["train/loss", "train/grad_norm", "train/lr", "eval/loss", "train/step_time"]
    rows = list(run.scan_history(keys=None, page_size=1000))
    df = pd.DataFrame(rows)
    df = df.rename(columns={f"steps/{k}": k for k in keys})
    keep = ["_step"] + [k for k in keys if k in df.columns]
    df = df[keep].sort_values("_step")
    summary = {k: v for k, v in run.summary.items() if "final_ce" in k}
    final_ce = next(iter(summary.values())) if summary else None
    meta = {
        "id": run.id,
        "final_ce": final_ce,
        "final_ce_keys": sorted(summary),
        "slurm_job": run.config.get("run_env", {}).get("slurm_job_id"),
        "n_steps": run.config.get("trainer", {}).get("n_steps"),
        "ga": run.config.get("trainer", {}).get("gradient_accumulation_steps"),
    }
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(path)
    json.dump(meta, open(meta_path, "w"))
    return df, meta


def series(df, key):
    s = df[["_step", key]].dropna()
    return dict(zip(s["_step"].astype(int), s[key].astype(float)))


def paired(a, b):
    """Per-step a - b over the steps both have, as (steps, gaps)."""
    steps = sorted(set(a) & set(b))
    return np.array(steps), np.array([a[s] - b[s] for s in steps])


def last50(steps, gaps):
    order = np.argsort(steps)
    return gaps[order][-50:]


def judge(gate, base, floor_names, runs):
    loss_base = series(runs[base][0], "train/loss")
    loss_gate = series(runs[gate][0], "train/loss")
    steps, gaps = paired(loss_gate, loss_base)
    tail = last50(steps, gaps)
    floors = []
    floors_ce = []
    for f in floor_names:
        fs, fg = paired(series(runs[f][0], "train/loss"), loss_base)
        floors.append(abs(last50(fs, fg).mean()))
        floors_ce.append(abs(runs[f][1]["final_ce"] - runs[base][1]["final_ce"]))
    F = max(floors) if floors else float("nan")
    F_ce = max(floors_ce) if floors_ce else float("nan")
    mean = float(tail.mean())
    sign = np.sign(mean)
    same_sign = int((np.sign(tail) == sign).sum())
    ce_gap = runs[gate][1]["final_ce"] - runs[base][1]["final_ce"]
    threshold = max(3 * F, 0.005)
    harmful = (
        abs(mean) >= threshold and same_sign >= 45 and np.sign(ce_gap) == sign
    )
    return {
        "gate": gate,
        "base": base,
        "last50_mean_gap": mean,
        "last50_same_sign": same_sign,
        "last50_nonzero": int((tail != 0).sum()),
        "all_steps_mean_gap": float(gaps.mean()),
        "floor_F": F,
        "floor_terms": dict(zip(floor_names, map(float, floors))),
        "floor_F_final_ce": F_ce,
        "threshold": threshold,
        "final_ce_gap": ce_gap,
        "harmful": bool(harmful),
    }


def grad_norm_view(df):
    g = series(df, "train/grad_norm")
    steps = sorted(g)
    arr = np.array([g[s] for s in steps])
    post = np.array([g[s] for s in steps if s >= 50])
    return {
        "step10": g.get(10),
        "max_first_20": float(arr[:20].max()),
        "argmax_first_20": int(steps[int(arr[:20].argmax())]),
        "median_after_50": float(np.median(post)) if len(post) else None,
        "clipped_after_50": int((post > 1.0).sum()) if len(post) else None,
    }


GROUPS_300 = {
    "production (Tier 2, as run)": [
        "parity_g0a_s2ps", "parity_g0b_s2ps", "parity_g0d_s2ps", "parity_g0e_s2ps",
    ],
    "production, micro-batch 16": ["parity_g0c_s2ps_mb16"],
    "ordinary path (mp null: A1-A3 off)": [
        "parity_g1_s2ps_mpnull", "parity_g1r_s2ps_mpnull", "parity_g1r2_s2ps_mpnull",
    ],
    "Tier 2, fp32 compute copy (A2 off)": ["parity_g1a_s2ps_fp32copy"],
    "Tier 2, eager gather (A3 off)": ["parity_g1b_s2ps_eagergather"],
    "Tier 2, fp32 prep on bf16 rows": ["parity_g1c_s2ps_fp32prep"],
    "dense": ["parity_gd0a_s1d", "parity_gd0b_s1d"],
}


def distribution(refresh=False):
    """Every 300-step gate as one sample: the last-50-step mean train loss and
    final_ce, grouped by configuration, with the spread inside each group and a
    rank test between the two groups with more than one sample each."""
    rows = {}
    for group, names in GROUPS_300.items():
        for name in names:
            try:
                df, meta = load(name, refresh)
            except RuntimeError as exc:
                print(f"skip {exc}")
                continue
            loss = series(df, "train/loss")
            tail = [loss[s] for s in sorted(loss)[-50:]]
            bumps = early_bumps(loss)
            rows[name] = (group, float(np.mean(tail)), meta["final_ce"], bumps)
    print(f"{'gate':<30} {'group':<36} {'last-50':>8} {'final_ce':>9}  early bumps (step: rise)")
    for name, (group, tail, ce, bumps) in rows.items():
        print(f"{name:<30} {group:<36} {tail:8.4f} {ce:9.5f}  {bumps}")
    by = {}
    for name, (group, tail, ce, _) in rows.items():
        by.setdefault(group, []).append((tail, ce))
    print()
    for group, vals in by.items():
        t = np.array([v[0] for v in vals])
        c = np.array([v[1] for v in vals])
        sd = f"{t.std(ddof=1):.4f}" if len(t) > 1 else "-"
        print(f"{group:<36} n={len(t)}  last-50 mean {t.mean():.4f} (sd {sd})  final_ce mean {c.mean():.5f}")
    a = [v[0] for v in by.get("production (Tier 2, as run)", [])]
    b = [v[0] for v in by.get("ordinary path (mp null: A1-A3 off)", [])]
    if len(a) > 1 and len(b) > 1:
        from scipy.stats import mannwhitneyu, ttest_ind

        print(
            f"ordinary - production: {np.mean(b) - np.mean(a):+.4f}; Mann-Whitney p "
            f"{mannwhitneyu(b, a, alternative='two-sided').pvalue:.4f}; Welch t p "
            f"{ttest_ind(b, a, equal_var=False).pvalue:.4f}"
        )
    return rows


def early_bumps(loss, start=15, end=60, rise=0.15):
    """Steps in [start, end) where the loss rises by more than `rise` above the
    best loss of the previous 5 steps: the early instabilities that decide a
    300-step gate."""
    out = []
    steps = sorted(s for s in loss if start <= s < end)
    for s in steps:
        prev = [loss[p] for p in range(s - 5, s) if p in loss]
        if prev and loss[s] - min(prev) > rise:
            out.append(f"{s}: +{loss[s] - min(prev):.2f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="300", choices=list(SETS) + ["dist"])
    parser.add_argument("--json", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.set == "dist":
        rows = distribution(args.refresh)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1, default=float)
        return
    spec = SETS[args.set]
    names = [spec["base"], *spec["floor"], *spec["gates"]]
    if spec["dense_base"]:
        names += [spec["dense_base"], *spec["dense_floor"]]
    runs = {}
    for name in names:
        try:
            runs[name] = load(name, args.refresh)
        except RuntimeError as exc:
            print(f"skip {exc}")
    out = {"runs": {n: runs[n][1] for n in runs}, "verdicts": [], "floors": {}}
    base = spec["base"]
    if base in runs:
        for f in spec["floor"]:
            if f in runs:
                out["verdicts"].append(judge(f, base, [x for x in spec["floor"] if x in runs and x != f], runs))
        for gate in spec["gates"]:
            if gate in runs:
                out["verdicts"].append(judge(gate, base, [x for x in spec["floor"] if x in runs], runs))
    if spec["dense_base"] in runs:
        for f in spec["dense_floor"]:
            if f in runs:
                out["verdicts"].append(judge(f, spec["dense_base"], [], runs))
    out["grad_norm"] = {n: grad_norm_view(runs[n][0]) for n in runs}
    for v in out["verdicts"]:
        print(
            f"{v['gate']:<28} vs {v['base']:<22} last-50 gap {v['last50_mean_gap']:+.5f} "
            f"({v['last50_same_sign']}/50 same sign, {v['last50_nonzero']} nonzero), all-steps "
            f"{v['all_steps_mean_gap']:+.5f}, F {v['floor_F']:.5f} -> threshold "
            f"{v['threshold']:.5f}; final_ce gap {v['final_ce_gap']:+.5f}; harmful {v['harmful']}"
        )
    for n, m in out["runs"].items():
        print(f"{n:<28} {m['id']}  job {m['slurm_job']}  final_ce {m['final_ce']}")
    for n, g in out["grad_norm"].items():
        print(f"{n:<28} grad-norm {g}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
