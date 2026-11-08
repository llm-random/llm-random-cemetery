"""Paired train-loss gap between same-seed runs, plus health readouts.

Built for documents/notes/ame_init_exp.md. Two runs with the same seed see the
same batches at the same step, so `new - reference` at each step cancels the
batch-to-batch noise, and a window mean of that difference is a finer measure
than final_ce, which is bf16-quantized (+-0.008 per run).

usage:
  python scripts/paired_gap.py NEW_ID REF_ID [NEW_ID REF_ID ...]

Per pair it prints final_ce and its difference; the mean gap over each quarter
of training, the first 100 steps and the last 500 steps; and for each run the
grad-norm max after step 100, loss spikes after step 100 (a step more than 0.1
above the median of the 50 before it), the median s/step and the peak memory.
Runs are read from W&B project ideas_cv/nano-ame.
"""

import sys

import numpy as np
import wandb

PROJECT = "ideas_cv/nano-ame"
SPIKE = 0.1


def series(run, key):
    """{step: value} for one logged key."""
    out = {}
    for row in run.scan_history(keys=["step", key], page_size=10000):
        value = row.get(key)
        if value is not None:
            out[int(row["step"])] = float(value)
    return out


def spikes(loss):
    steps = sorted(loss)
    values = np.array([loss[s] for s in steps])
    found = []
    for i in range(50, len(steps)):
        if steps[i] <= 100:
            continue
        if values[i] - np.median(values[i - 50 : i]) > SPIKE:
            found.append((steps[i], values[i]))
    return found


def health(run, loss):
    grad = series(run, "steps/train/grad_norm")
    late = {s: g for s, g in grad.items() if s > 100}
    worst = max(late, key=late.get) if late else None
    step_time = list(series(run, "steps/train/step_time").values())
    memory = list(series(run, "steps/train/max_memory_gib").values())
    found = spikes(loss)
    return {
        "grad_max_after_100": (late[worst], worst) if worst is not None else None,
        "spikes": len(found),
        "spike_steps": [s for s, _ in found[:10]],
        "s_per_step_median": float(np.median(step_time)) if step_time else None,
        "max_memory_gib": max(memory) if memory else None,
        "last_step": max(loss) if loss else None,
    }


def window_mean(gap, lo, hi):
    values = [g for s, g in gap.items() if lo <= s < hi]
    return (float(np.mean(values)), len(values)) if values else (None, 0)


def compare(api, new_id, ref_id):
    new = api.run(f"{PROJECT}/{new_id}")
    ref = api.run(f"{PROJECT}/{ref_id}")
    n_steps = int(ref.config["trainer"]["n_steps"])
    new_loss = series(new, "steps/train/loss")
    ref_loss = series(ref, "steps/train/loss")
    common = sorted(set(new_loss) & set(ref_loss))
    gap = {s: new_loss[s] - ref_loss[s] for s in common}
    last = max(common) if common else 0

    print(f"== {new.name} ({new_id}) against {ref.name} ({ref_id})")
    new_ce = new.summary.get("steps/final_ce")
    ref_ce = ref.summary.get("steps/final_ce")
    delta = (
        f"{new_ce - ref_ce:+.4f}" if new_ce is not None and ref_ce is not None else "-"
    )
    print(f"  final_ce  new {new_ce}  ref {ref_ce}  delta {delta}")
    print(f"  paired steps: {len(common)}, last {last} of {n_steps}")
    windows = [("steps 0-99", 0, 100)]
    quarter = n_steps // 4
    for q in range(4):
        lo, hi = q * quarter, (q + 1) * quarter if q < 3 else n_steps
        windows.append((f"quarter {q + 1} ({lo}-{hi - 1})", lo, hi))
    windows.append((f"last 500 ({n_steps - 500}-{n_steps - 1})", n_steps - 500, n_steps))
    windows.append((f"last 500 logged ({last - 499}-{last})", last - 499, last + 1))
    for name, lo, hi in windows:
        mean, count = window_mean(gap, lo, hi)
        shown = f"{mean:+.4f}" if mean is not None else "-"
        print(f"  gap {name:32s} {shown}  (n={count})")
    for label, run, loss in (("new", new, new_loss), ("ref", ref, ref_loss)):
        h = health(run, loss)
        print(f"  {label}: {h}")


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        sys.exit(__doc__)
    api = wandb.Api(timeout=120)
    for i in range(0, len(argv), 2):
        compare(api, argv[i], argv[i + 1])


if __name__ == "__main__":
    main(sys.argv[1:])
