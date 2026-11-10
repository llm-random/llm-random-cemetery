"""Document-mask runs against their baselines: the numbers in doc_mask_report.md.

Read-only on W&B (ideas_cv/nano-ame). Each pair shares seed, data order, steps
and schedule, so per-step train losses pair, and their difference is free of
batch-to-batch noise.

    python documents/scripts/doc_mask_results.py <new run name or id> <baseline id> [--last N]

Prints, for the steps both runs have logged so far:
- s/step: median `train/step_time` over steps >= 100 (stage_results.py's convention);
- the paired train-loss gap (new - baseline) per part of training and over the
  last N steps (default 500), the finer measure (reference: final_ce precision);
- NaN and spikes: a train loss more than 0.5 above the median of the previous 50;
- eval/loss gaps at the shared eval steps, and final_ce once both have it.

Run it with the project env; WANDB_API_KEY must be set.
"""

import argparse
import math
import statistics

import wandb

PROJECT = "ideas_cv/nano-ame"
N_STEPS = 3815
PARTS = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]


def find_run(api, key):
    """A run by W&B id, or by display name (the most recent of that name)."""
    try:
        return api.run(f"{PROJECT}/{key}")
    except Exception:
        runs = api.runs(PROJECT, filters={"display_name": key}, order="-created_at")
        runs = list(runs)
        if not runs:
            raise SystemExit(f"no run {key!r} in {PROJECT}")
        return runs[0]


def history(run, keys):
    """step -> {key: value}. The trainer logs under `steps/<key>` against the
    custom x-axis `step`. One scan per key: scan_history(keys=...) returns only
    the rows that hold every key it is given, and these are logged at
    different intervals."""
    rows = {}
    for key in keys:
        for row in run.scan_history(keys=["step", f"steps/{key}"], page_size=10000):
            if row.get("step") is not None and row.get(f"steps/{key}") is not None:
                rows.setdefault(int(row["step"]), {})[key] = row[f"steps/{key}"]
    return rows


def series(rows, key):
    return {s: r[key] for s, r in rows.items() if key in r}


def spikes(loss, window=50, threshold=0.5):
    steps = sorted(loss)
    found = []
    for i, step in enumerate(steps):
        previous = [loss[s] for s in steps[max(0, i - window) : i]]
        if len(previous) >= 10 and loss[step] > statistics.median(previous) + threshold:
            found.append((step, loss[step]))
    return found


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new")
    parser.add_argument("baseline")
    parser.add_argument("--last", type=int, default=500)
    args = parser.parse_args()

    api = wandb.Api()
    new, base = find_run(api, args.new), find_run(api, args.baseline)
    keys = ["train/loss", "train/step_time", "eval/loss", "train/grad_norm"]
    hn, hb = history(new, keys), history(base, keys)

    print(f"new      {new.name} ({new.id}), state {new.state}")
    print(f"baseline {base.name} ({base.id}), state {base.state}")
    print(
        f"document_mask in new config: "
        f"{new.config.get('common', {}).get('attention', {}).get('document_mask')}"
    )

    for label, rows in (("new", hn), ("baseline", hb)):
        times = [t for s, t in series(rows, "train/step_time").items() if s >= 100]
        loss = series(rows, "train/loss")
        nan = [s for s, v in loss.items() if v != v or math.isinf(v)]
        print(
            f"{label:8s} steps logged {max(loss) + 1 if loss else 0}, "
            f"s/step {statistics.median(times) if times else float('nan'):.3f}, "
            f"NaN/inf {len(nan)}, spikes {spikes(loss)[:5]}"
        )
        grads = series(rows, "train/grad_norm")
        late = [v for s, v in grads.items() if s >= 100]
        if late:
            print(
                f"{'':8s} grad norm max after step 100 {max(late):.3f}, last {grads[max(grads)]:.3f}"
            )

    ln, lb = series(hn, "train/loss"), series(hb, "train/loss")
    shared = sorted(set(ln) & set(lb))
    if not shared:
        print("no shared train steps yet")
        return
    gap = {s: ln[s] - lb[s] for s in shared}
    last = shared[-1]
    print(f"\npaired train-loss gap (new - baseline), steps 0..{last}:")
    for lo, hi in PARTS:
        steps = [s for s in shared if lo * N_STEPS <= s < hi * N_STEPS]
        if steps:
            print(
                f"  {int(lo*100):3d}-{int(hi*100):3d}%  steps {steps[0]}-{steps[-1]}: "
                f"{mean(gap[s] for s in steps):+.4f} over {len(steps)}"
            )
    tail = shared[-args.last :]
    print(
        f"  last {len(tail)} steps ({tail[0]}-{tail[-1]}): {mean(gap[s] for s in tail):+.4f}"
    )
    for width in (250,):
        print(
            f"  per {width}-step window:",
            ", ".join(
                f"{start}: {mean(gap[s] for s in shared if start <= s < start + width):+.3f}"
                for start in range(0, last + 1, width)
                if any(start <= s < start + width for s in shared)
            ),
        )

    en, eb = series(hn, "eval/loss"), series(hb, "eval/loss")
    evals = sorted(set(en) & set(eb))
    if evals:
        print(
            "\neval/loss gap at shared eval steps (last 8):",
            ", ".join(f"{s}: {en[s] - eb[s]:+.4f}" for s in evals[-8:]),
        )

    fn = new.summary.get("steps/final_ce", new.summary.get("final_ce"))
    fb = base.summary.get("steps/final_ce", base.summary.get("final_ce"))
    if fn is not None and fb is not None:
        print(f"\nfinal_ce new {fn:.4f}, baseline {fb:.4f}, difference {fn - fb:+.4f}")
    else:
        print(f"\nfinal_ce new {fn}, baseline {fb}")


if __name__ == "__main__":
    main()
