#!/usr/bin/env python
"""Per-kernel ms/step of two kprof dumps side by side, sorted by what B adds over A.

Each side is a log file holding `kprof_us:` lines (trainer.kernel_profile=true), or a
W&B run id whose output.log is fetched. Inductor's trailing `_<n>` kernel index is
dropped so the same fused kernel matches across two compiles.

Usage:
    pixi run python kprof_diff.py <A: log or run id> <B: log or run id> [--top 40]
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from kprof_table import ROW, short  # noqa: E402

RUN_PATH = "ideascv_llms/llm-random-test"
CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "kprof_diff"


def log_lines(source):
    if os.path.isfile(source):
        return open(source, errors="replace")
    import wandb

    root = CACHE / source
    wandb.Api(timeout=120).run(f"{RUN_PATH}/{source}").file("output.log").download(
        root=str(root), replace=True
    )
    return open(root / "output.log", errors="replace")


def kernels(source, steps):
    """{kernel: [ms/step, launches/step]}"""
    out = defaultdict(lambda: [0.0, 0])
    for line in log_lines(source):
        m = ROW.search(line)
        if m:
            # Same fused kernel, different index between two compiles.
            name = re.sub(r"^(triton_\w+?)_\d+$", r"\1", short(m[3].strip(), 200))
            out[name][0] += int(m[1]) / (steps * 1000)
            out[name][1] += int(m[2]) // steps
    if not out:
        raise SystemExit(f"no kprof_us lines in {source}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2, help="steps the dump covers")
    args = parser.parse_args()

    a, b = kernels(args.a, args.steps), kernels(args.b, args.steps)
    rows = []
    for name in a.keys() | b.keys():
        (ms_a, n_a), (ms_b, n_b) = a.get(name, (0.0, 0)), b.get(name, (0.0, 0))
        rows.append((ms_b - ms_a, ms_a, ms_b, n_a, n_b, name))
    rows.sort(key=lambda r: -abs(r[0]))

    print(f"A = {args.a}   B = {args.b}   (ms/step, launches/step)")
    print(f"{'B-A':>8} {'A':>8} {'B':>8} {'nA':>5} {'nB':>5}  kernel")
    for delta, ms_a, ms_b, n_a, n_b, name in rows[: args.top]:
        print(f"{delta:+8.2f} {ms_a:8.2f} {ms_b:8.2f} {n_a:5d} {n_b:5d}  {name[:110]}")
    total_a, total_b = sum(v[0] for v in a.values()), sum(v[0] for v in b.values())
    print(
        f"\nTOTAL  A {total_a:,.1f}  B {total_b:,.1f}  B-A {total_b - total_a:+,.1f} ms/step"
        f" ({(total_b - total_a) / total_a:+.1%})"
    )


if __name__ == "__main__":
    main()
