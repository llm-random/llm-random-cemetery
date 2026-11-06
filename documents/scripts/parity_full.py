#!/usr/bin/env python
"""ame_bf16_scalars_plan.md, Step 4: the verdicts on the full-length entropy runs.

    python documents/scripts/parity_full.py --scores DIR [--json OUT]

Reads each run's W&B history (project ideas_cv/nano-ame-parity, names
parity_full_*; cached under $PARITY_FULL_CACHE, default ~/.cache/nano/parity_full)
and the `parity_numerics.py score` files in DIR (score_<run>.json: precise CE on
the full held-out set under the common evaluator, <arm>_ref32, and under the
run's own production path). Then:

- Q-P1 (P1 against AME-B) and Q-REF (P-REF against AME-B): harmful if production
  is worse than the item-off runs by more than max(3 x the AME-B spread, 0.005)
  under the common evaluator, *and* the 50-step paired train-loss gap keeps that
  sign over >= 90% of the last 1,000 steps;
- Q-X2: I = [CE(P1-16) - CE(AME-F)] - [CE(D-B) - CE(D-F)] with a 95% interval
  from the groups' spreads (Welch-Satterthwaite); harmful if the interval lies
  above 0 and I >= 0.005, negligible if the whole interval lies inside +-0.005,
  otherwise inconclusive with the bound.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

PROJECT = "ideas_cv/nano-ame-parity"
CACHE = os.environ.get("PARITY_FULL_CACHE", os.path.expanduser("~/.cache/nano/parity_full"))

GROUPS = {
    "AME-B": ["ame_b1", "ame_b2", "ame_b3"],
    "P1": ["p1_1", "p1_2", "p1_3"],
    "P1-16": ["p1_16_1", "p1_16_2", "p1_16_3"],
    "AME-F": ["ame_f1", "ame_f2", "ame_f3"],
    "D-B": ["db1", "db2", "db3"],
    "D-F": ["df1", "df2", "df3"],
    "P-REF": ["pref"],
    "P0b": ["p0b"],
}
COMMON = {"AME-B": "ame_ref32", "P1": "ame_ref32", "P1-16": "ame_ref32", "AME-F": "ame_ref32",
          "P-REF": "ame_ref32", "P0b": "ame_ref32", "D-B": "dense_ref32", "D-F": "dense_ref32"}
OWN = {"AME-B": "ame_prod", "P1": "ame_std", "P1-16": "ame_std", "AME-F": "ame_f",
       "P-REF": "ame_pref", "P0b": "ame_prod", "D-B": "dense_prod", "D-F": "dense_f"}


def history(short, refresh=False):
    import pandas as pd

    name = f"parity_full_{short}"
    path = os.path.join(CACHE, f"{name}.parquet")
    meta_path = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path), json.load(open(meta_path))
    import wandb

    api = wandb.Api(timeout=180)
    runs = [r for r in api.runs(PROJECT, filters={"display_name": name}) if r.state == "finished"]
    if not runs:
        return None, None
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    keys = ["train/loss", "train/grad_norm", "eval/loss", "train/step_time"]
    rows = list(run.scan_history(page_size=2000))
    df = pd.DataFrame(rows).rename(columns={f"steps/{k}": k for k in keys})
    readouts = [c for c in df.columns if "attention_layer.layer/" in c and c.startswith("steps/")]
    df = df[["_step"] + [k for k in keys if k in df.columns] + readouts].sort_values("_step")
    summary = {k: v for k, v in run.summary.items() if "final_ce" in k}
    meta = {"id": run.id, "final_ce": next(iter(summary.values())) if summary else None,
            "slurm_job": run.config.get("run_env", {}).get("slurm_job_id")}
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(path)
    json.dump(meta, open(meta_path, "w"))
    return df, meta


def welch(groups_terms):
    """sum of signed group means and its 95% interval; groups_terms is a list of
    (sign, values)."""
    from scipy.stats import t

    est = sum(sign * np.mean(v) for sign, v in groups_terms)
    var_terms = [np.var(v, ddof=1) / len(v) if len(v) > 1 else float("nan") for _, v in groups_terms]
    se = math.sqrt(sum(var_terms))
    dof = sum(var_terms) ** 2 / sum(vt**2 / (len(v) - 1) for vt, (_, v) in zip(var_terms, groups_terms))
    half = t.ppf(0.975, dof) * se
    return est, est - half, est + half, se, dof


def paired_gap_sign(prod_runs, off_runs, last=1000, window=50):
    """The 50-step paired train-loss gap, mean(production) - mean(item off),
    over the last `last` steps: the share of steps carrying the gap's sign."""
    import pandas as pd

    def mean_curve(runs):
        cols = [r.set_index("_step")["train/loss"] for r in runs]
        return pd.concat(cols, axis=1).mean(axis=1)

    gap = (mean_curve(prod_runs) - mean_curve(off_runs)).rolling(window).mean().dropna()
    tail = gap.iloc[-last:]
    sign = np.sign(tail.mean())
    return float(tail.mean()), float((np.sign(tail) == sign).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    scores = {}
    for path in glob.glob(os.path.join(args.scores, "**", "score_*.json"), recursive=True):
        d = json.load(open(path))
        scores[d["tag"]] = d
    table, hist = {}, {}
    for group, shorts in GROUPS.items():
        for short in shorts:
            df, meta = history(short, args.refresh)
            sc = scores.get(short, {})
            table[short] = {
                "group": group,
                "trainer_final_ce": meta["final_ce"] if meta else None,
                "common": sc.get(COMMON[group]),
                "own": sc.get(OWN[group]),
                "wandb": meta["id"] if meta else None,
                "s_per_step": float(df.loc[df["_step"] >= 15, "train/step_time"].median()) if df is not None and "train/step_time" in df else None,
            }
            hist[short] = df
            print(f"{group:<6} {short:<9} final_ce {table[short]['trainer_final_ce']}  common {table[short]['common']}  own {table[short]['own']}  s/step {table[short]['s_per_step']}")
    out = {"runs": table}

    def vals(group, key="common"):
        return [table[s][key] for s in GROUPS[group] if table[s][key] is not None]

    b = vals("AME-B")
    spread = float(np.std(b, ddof=1)) if len(b) > 1 else float("nan")
    out["ame_b_spread_sd"] = spread
    for q, off in (("Q-P1", "P1"), ("Q-REF", "P-REF")):
        o = vals(off)
        if not b or not o:
            continue
        diff = float(np.mean(b) - np.mean(o))  # > 0: production worse
        threshold = max(3 * spread, 0.005)
        runs_b = [hist[s] for s in GROUPS["AME-B"] if hist.get(s) is not None]
        runs_o = [hist[s] for s in GROUPS[off] if hist.get(s) is not None]
        gap, share = paired_gap_sign(runs_b, runs_o) if runs_b and runs_o else (float("nan"), float("nan"))
        harmful = diff >= threshold and gap > 0 and share >= 0.9
        interval = welch([(1, b), (-1, o)]) if len(o) > 1 else None
        out[q] = {"production_minus_off": diff, "threshold": threshold, "paired_gap_last1000": gap,
                  "share_with_sign": share, "harmful": bool(harmful), "welch": interval}
        print(f"{q}: AME-B - {off} = {diff:+.5f} (threshold {threshold:.5f}); paired gap {gap:+.5f}, sign share {share:.2f}; harmful {harmful}; interval {interval}")
    if all(vals(g) for g in ("P1-16", "AME-F", "D-B", "D-F")):
        est, lo, hi, se, dof = welch([(1, vals("P1-16")), (-1, vals("AME-F")), (-1, vals("D-B")), (1, vals("D-F"))])
        verdict = "harmful" if (lo > 0 and est >= 0.005) else ("negligible" if (lo > -0.005 and hi < 0.005) else "inconclusive")
        out["Q-X2"] = {"I": est, "lo": lo, "hi": hi, "se": se, "dof": dof, "verdict": verdict,
                       "ame_gain": float(np.mean(vals("P1-16")) - np.mean(vals("AME-F"))),
                       "dense_gain": float(np.mean(vals("D-B")) - np.mean(vals("D-F")))}
        print(f"Q-X2: I = {est:+.5f}, 95% [{lo:+.5f}, {hi:+.5f}] (se {se:.5f}, dof {dof:.1f}); verdict {verdict}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
