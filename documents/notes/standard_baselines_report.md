# Standard baselines: AME v4 and dense v3 at h12, h16 and h20

**Status (2026-11-27 02:10 CET):**
- The six 1-GPU runs are queued on LEM, PENDING (Priority).
- The four 4-GPU twins are not queued yet. Both of LEM's Slurm controllers went down at
  about 02:00, while the twins were being submitted. A retry loop submits them once the
  controller answers, for up to 6 h.

Maciej asked for this on 2026-11-27: run AME and dense at h12, h16 and h20 on LEM, one GPU
each, with the tuned learning rates, the established token budgets and everything else as
in [`configs/standard/`](../../configs/standard/). These are the first runs of recipe v3
(dense) and v4 (AME). They pair with no earlier run, so there is no target value: a run passes
on health checks alone ([comparing-runs.md](../concepts/comparing-runs.md#which-runs-pair)).
The AME and dense runs at one H pair with each other in data order, steps and schedule, so
the gap at each H is a paired gap.

Maciej then asked for a 4-GPU twin of each h16 and h20 run as well: those should finish
sooner, and comparing each pair shows what the GPU count changes
([below](#1-gpu-against-4-gpus)).

## Runs

| Config | Recipe | LR | Tokens (steps) | Micro-batch × steps | Limit | Expected | Job | W&B | State | `final_ce` |
|---|---|---|---|---|---|---|---|---|---|---:|
| `ame_h12` | v4 | 2⁻⁸, banks 4× | 4B (3815) | 32 × 16 | 7 h | ~4.5 h, 3.9 s/step | 6449445 | | PENDING | |
| `dense_h12` | v3 | 2⁻⁸ | 4B (3815) | 32 × 16 | 7 h | ~4 h, 3.45 s/step | 6449446 | | PENDING | |
| `ame_h16` | v4 | 2⁻⁸, banks 4× | 8B (7630) | 16 × 32 | 24 h | ~15.5 h, 7.1 s/step | 6449447 | | PENDING | |
| `dense_h16` | v3 | 2⁻⁹ | 8B (7630) | 16 × 32 | 24 h | ~14 h, 6.27 s/step | 6449448 | | PENDING | |
| `ame_h20` | v4 | 2⁻⁸, banks 4× | 16B (15259) | 4 × 128 | 72 h | ~50 h, ~12 s/step | 6449449 | | PENDING | |
| `dense_h20` | v3 | 2⁻⁹ | 16B (15259) | 4 × 128 | 72 h | ~47 h, ~11 s/step | 6449450 | | PENDING | |
| `ame_h16_4gpu` | v4 | 2⁻⁸, banks 4× | 8B (7630) | 4 GPUs × 16 × 8 | 8 h | ~4 h, ~1.85 s/step | — | | not queued | |
| `dense_h16_4gpu` | v3 | 2⁻⁹ | 8B (7630) | 4 GPUs × 16 × 8 | 8 h | ~3.3 h, ~1.5 s/step | — | | not queued | |
| `ame_h20_4gpu` | v4 | 2⁻⁸, banks 4× | 16B (15259) | 4 GPUs × 16 × 8 | 24 h | ~14 h, 3.1 s/step | — | | not queued | |
| `dense_h20_4gpu` | v3 | 2⁻⁹ | 16B (15259) | 4 GPUs × 16 × 8 | 24 h | ~11 h, 2.52 s/step | — | | not queued | |

The first six use one H100 each. The `_4gpu` twins use one whole node each (4 H100), with
FSDP2 and bf16 from its mixed precision; AME's banks are column-sharded over the four GPUs.
The twins come from `66faba7c`; each inherits its 1-GPU config and changes only the parallel
settings. The h20 twins are exactly the 4-GPU recipe h20 had until `ed4dbaef` (smokes: 2.96
and 2.52 s/step, v1). The h16 twins take the settings of the best 4-GPU h16 cell measured
(`configs/ame/h16_4gpu.yaml`).

**Common to the first six:**
- one H100 on LEM, r16-11 excluded;
- seed 123, train `num_workers` 2, batch 512 × 2049 tokens;
- autocast bf16 without FSDP2, `compile_blocks_full`;
- the final checkpoint only.

**Where the code is:** `ed4dbaef` on `ame_main`, which moved h20 onto one GPU. Each cemetery
branch is that commit plus its `generated_configs/` and `exp.job`. W&B project: `nano-ame`.

**Where the expected times come from:**
- h12 and h16: the 60-step smokes of 2026-11-21
  ([standard_models_report.md](archive/standard_models_report.md)). The AME times include v4's
  +4.0% and +4.9% FLOPs.
- h20: estimated from the 1-GPU measurements in
  [ame_optimization_report.md](ame_optimization_report.md), §1.16 and §1.22. The h20 config
  headers show the arithmetic.

Cemetery clones, under `/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`:

| Config | Branch and clone |
|---|---|
| `ame_h12` | `std_ame_h12_lr8_v4_2026-11-27_01-40-24` |
| `dense_h12` | `std_dense_h12_lr8_v3_2026-11-27_01-40-43` |
| `ame_h16` | `std_ame_h16_lr8_v4_2026-11-27_01-41-00` |
| `dense_h16` | `std_dense_h16_lr9_v3_2026-11-27_01-41-18` |
| `ame_h20` | `std_ame_h20_lr8_v4_2026-11-27_01-41-36` |
| `dense_h20` | `std_dense_h20_lr9_v3_2026-11-27_01-41-52` |
| `ame_h16_4gpu` | `std_ame_h16_lr8_v4_4gpu_2026-11-27_01-57-38` |
| `dense_h16_4gpu` | `std_dense_h16_lr9_v3_4gpu_2026-11-27_01-58-00` |
| `ame_h20_4gpu` | `std_ame_h20_lr8_v4_4gpu_2026-11-27_01-58-20` |
| `dense_h20_4gpu` | `std_dense_h20_lr9_v3_4gpu_2026-11-27_01-58-37` |

## 1 GPU against 4 GPUs

A twin and its 1-GPU run start from the same weights: every rank builds the whole model from
the seed before FSDP2 shards it (`main.py`), and the banks' column shards slice the full draw.
They read the same batches in the same order (`world_size_independent`). What differs is
numerics:

| | 1 GPU | 4 GPUs |
|---|---|---|
| bf16 from | autocast (fp32 weights, bf16 casts per op) | FSDP2's MixedPrecisionPolicy |
| AME banks | whole, hand-managed path | column-sharded over the 4 GPUs |
| micro-batch | 16 at h16, 4 at h20 | 16 |
| gradient sum | over micro-batches | over micro-batches, then over 4 ranks |

- **The measure:** the paired gap, the mean of (4 GPU − 1 GPU) `steps/train/loss` over the
  last 500 steps (`scripts/paired_gap.py`, on branch `bank_sigma_grid`), with the `final_ce`
  difference (±0.016) beside it.
- **The step-0 loss** may differ in its last digits, because the precision path differs.
- **The band:** a difference under 0.022 is a tie for AME (the h12 band; none is measured at
  h16 or h20). Dense has no band; a dense gap is read with its loss curves
  ([comparing-runs.md](../concepts/comparing-runs.md#when-a-difference-is-a-tie)).

## How each run is checked

- **Start, about 20 minutes after the job starts** (one check):
  - the step-0 loss is about 11.2 (smokes: AME 11.25, dense 11.19–11.25);
  - s/step is within 10% of the expected value;
  - no NaN;
  - AME logs its readouts;
  - peak memory: AME h16 is the tightest. Its v1 smoke peaked at 72.2 GiB, and v4's wider
    feed-forward adds about 1.4 GiB, an estimate, of 79.2.
- **End:** `final_ce` is logged. A job that fails at the end-of-run checkpoint save after
  `final_ce` is "done, then quota": its result stands
  ([settle_runs_report.md](archive/settle_runs_report.md#runs)).
- **If AME h16 runs out of memory:** resubmit at micro-batch 8 (`gradient_accumulation_steps:
  64`). The data order does not change with the micro-batch.

## Risks

- **LEM file quota.** On 2026-11-27 the project held 950,076 files, over the 900,096 soft
  limit, with 5d19h of grace left, so the grace ends about 2026-12-02 20:40.
  - After that, new files fail: a run that ends later loses its final checkpoint, but not
    its `final_ce`.
  - The h20 runs take about 2 days, so they must start by about 2026-11-30 to finish before
    then.
  - The admins have announced a 200,000-file quota
    ([lem_file_quota_request.md](lem_file_quota_request.md)). Once it applies, a job dies as
    soon as it creates a file.
- **GitHub from LEM.** `github.com:22` refused mid-submission, so `ame_h16` and `dense_h20` were
  cloned over `ssh.github.com:443`. LEM's own `github.com` host keys were written under that
  name to `…/nano/plgmstefaniak/known_hosts_github443`; `$HOME` is full.

## Log

- **2026-11-27, 01:15–01:46.**
  - h20 moved to one GPU (`ed4dbaef`): micro-batch 4 for both models, no FSDP2, a 3-day
    limit.
  - The six configs composed and passed run_exp's checks;
    [tests/test_data_order.py](../../tests/test_data_order.py) passed, 21 tests.
  - The local halves ran from a clean worktree with the secrets unset and pushed 01:40–01:42.
    Each branch is `ed4dbaef` plus its generated files.
  - The six clones were checked: the head, a sparse checkout, no Slurm log, no live secret
    line, the `#SBATCH` lines and the config values.
  - All six went to `sbatch` from one shell that read the W&B key from stdin.
  - 101 jobs were pending in the partition.
- **2026-11-27, 01:50–02:10.**
  - The 4-GPU twins were written (`66faba7c`). Resolved, the h20 twins equal the 4-GPU files
    at `19d4d83e` except in name, tags and job name. Each twin differs from its 1-GPU config
    only in the parallel settings. [tests/test_data_order.py](../../tests/test_data_order.py):
    25 passed.
  - The local halves pushed 01:57–01:58. The four clones were made over `ssh.github.com:443`
    and checked.
  - At about 02:00 all four `sbatch` calls failed with "Unable to contact slurm controller". At
    02:03 `scontrol ping` reported both controllers DOWN (`slurm-1`, `slurm-2`), and
    `slurm-db` timed out.
  - A retry loop now submits the four when `squeue` answers again. It skips any job name
    already in the queue, so a lost reply cannot make a duplicate.
