#!/usr/bin/env python
"""Per-run MFU averaged over training, for W&B runs selected by tags.

The W&B summary keeps only the last step's MFU. This reads the full history:
  mfu_mean    total work / total time over training; eval and checkpoint pauses
              between steps count against it
  mfu_median  the typical training step; for formula runs the typical 100-step
              window, which still holds eval pauses, so it sits closer to the mean
source says where the numbers come from:
  logged      runs since 2026-10-30 (commit 997a16f7) log train/mfu every step,
              with FLOPs measured on the live model; averaged as is
  formula     older runs: FLOPs from the run's config via Setup
              (jupyter_notebooks/mfu_calc.py), time from steps/100/time
n/a: neither is available (run died before step 100, unknown GPU or FF type).

Usage:
  pixi run python wandb_mfu.py --project llm-random-test --tags 2026-04-30 k24 [--csv out.csv]
"""

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent / "jupyter_notebooks"))
from mfu_calc import PEAK_BF16, Setup  # noqa: E402

MFU_KEY = "steps/train/mfu"
STEP_TIME_KEY = "steps/train/step_time"
WINDOW_TIME_KEY = "steps/100/time"  # mean step time over each 100-step window
FF_MATRICES = {"SwiGLU": 3, "GatedFeedForward": 3, "GeluFeedForward": 2}


def logged_mfu(run):
    if MFU_KEY not in run.summary:
        return None, None  # never logged, skip the history download
    rows = run.scan_history(keys=[MFU_KEY, STEP_TIME_KEY], page_size=10_000)
    mfus, times = [], []
    for row in rows:
        mfus.append(row[MFU_KEY])
        times.append(row[STEP_TIME_KEY])
    if not mfus:
        return None, None
    # mfu * step_time is the step's work in peak-seconds
    mean = sum(m * t for m, t in zip(mfus, times)) / sum(times)
    return mean, statistics.median(mfus)


def setup_from_config(run, gpu, n_gpus):
    common = run.config["common"]
    if common.get("datt", common["dmodel"]) != common["dmodel"]:
        return None  # Setup assumes the attention width is dmodel
    ff = run.config["model"]["encoder"]["block_fn"]["ff_layer_fn"]
    kind = ff["_target_"].rsplit(".", 1)[-1]
    if kind == "MoE":
        n_experts, topk, dff = ff["num_experts"], ff["topk"], ff["dff"]
        ff_matrices = 3 if ff["activation_function"] == "swiglu" else 2
    elif kind in FF_MATRICES:
        n_experts, topk, dff, ff_matrices = 1, 1, common["dff"], FF_MATRICES[kind]
    else:
        return None
    return Setup(
        n_layers=common["n_blocks"],
        dmodel=common["dmodel"],
        dff=dff,
        q_heads=common["q_heads"],
        kv_heads=common["kv_heads"],
        ff_matrices=ff_matrices,
        n_experts=n_experts,
        topk=topk,
        vocab=common["vocab_size"],
        sl=common["sequence_length"],
        bs=common["batch_size"],
        gpu=gpu,
        n_gpus=n_gpus,
        include_attention_scores=True,
    )


def recalculated_mfu(run, setup):
    rows = run.scan_history(keys=[WINDOW_TIME_KEY], page_size=10_000)
    times = [row[WINDOW_TIME_KEY] for row in rows]
    if not times:
        return None, None
    # equal-length windows, so their plain mean is the time-weighted mean step time
    ideal_step_time = setup.flops_per_step / setup.peak
    return ideal_step_time / statistics.mean(times), ideal_step_time / statistics.median(times)


def describe(run):
    metadata = run.metadata or {}
    n_gpus = run.config.get("world_size") or metadata.get("gpu_count")
    gpu_name = metadata.get("gpu", "?")
    source = "logged"
    mfu_mean, mfu_median = logged_mfu(run)
    if mfu_mean is None:
        source = "formula"
        gpu = next((key for key in PEAK_BF16 if key in gpu_name.lower()), None)
        setup = gpu and n_gpus and setup_from_config(run, gpu, n_gpus)
        if setup:
            mfu_mean, mfu_median = recalculated_mfu(run, setup)
    last_step = run.summary.get("step")
    return {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "gpus": f"{n_gpus}x {gpu_name.removeprefix('NVIDIA ')}",
        "steps": None if last_step is None else last_step + 1,
        "hours": run.summary.get("_runtime", 0) / 3600,
        "source": source if mfu_mean is not None else None,
        "mfu_mean": mfu_mean,
        "mfu_median": mfu_median,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entity", default="ideas_cv")
    parser.add_argument("--project", required=True)
    parser.add_argument("--tags", nargs="+", required=True, help="a run must carry all of them")
    parser.add_argument("--csv", help="also write the table to this file")
    args = parser.parse_args()

    filters = {"$and": [{"tags": {"$in": [tag]}} for tag in args.tags]}
    runs = wandb.Api(timeout=60).runs(
        f"{args.entity}/{args.project}", filters=filters, order="-created_at"
    )
    rows = []
    for run in runs:
        print(f"reading {run.name} ({run.id})", file=sys.stderr)
        rows.append(describe(run))
    if not rows:
        sys.exit(f"no runs in {args.entity}/{args.project} carry all of {args.tags}")

    table = pd.DataFrame(rows).astype({"steps": "Int64"})
    shown = table.astype(object)
    for column, fmt in {"hours": "{:.2f}", "mfu_mean": "{:.2%}", "mfu_median": "{:.2%}"}.items():
        shown[column] = [None if pd.isna(x) else fmt.format(x) for x in table[column]]
    print(shown.fillna("n/a").to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
