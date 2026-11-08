# Stage 1c — dense h20 at 16B tokens, on 4 GPUs: report

**Status: in progress (started 2026-11-08 00:40). The Helios compile fix works (job
25860605). The S1c-D grid, n ∈ {8, 9, 10}, is queued on Helios as job 25862221 (submitted
01:15), behind its 4-GPU smoke runs (25862155). At 09:30 every job was still pending, behind
higher-priority jobs, after 8 h 15 min in the queue (decision 5). At 11:30 Slurm first
forecast a start for the smoke runs, Tuesday 2026-11-10 at 05:05, and none for the real
runs.**

The plan is [ame_experiments.md, "Stage 1c — dense h20 at 16B tokens, on 4 GPUs"](ame_experiments.md#stage-1c--dense-h20-at-16b-tokens-on-4-gpus),
with its section "Helios — what Stages 1c and 2c need first". This file is its record:
every job, result and decision, each with its reason. Times are CET. Stage 2c runs
alongside it on the same commit, worktree, cluster and recipe; its record is
[stage_2c_report.md](stage_2c_report.md). The Helios fix and the submission method below
serve both stages, and are recorded here only.

## Result

Pending.

## Setup

| | |
|---|---|
| Question | What learning rate does a dense h20 baseline need at 16B tokens, and what loss does it reach? It is Stage 2c's reference and the third point of the dense series, after Stage 1 (h12, 4B) and Stage 1b (h16, 8B) |
| Arm | S1c-D: S1b-D on h_balanced h20, `gqa` attention with RoPE (base 10000), 20 query heads and 10 KV heads |
| Code and configs | Commit **`683c316b`** on `ame_main`, for every Stage 1c and 2c run. Jobs are submitted from the clean worktree `~/code/nano-stage1c2c` (branch `stage1c2c_helios`), held at that commit while other work lands on `ame_main`. Since Stage 2b's `89975bbf`, nothing in `src/core/ame.py`, `src/core/memory_parallel.py`, `src/core/trainer.py` or `main.py`'s model build has changed that the plan does not name; `683c316b` adds the two Stage 1c–2c configs, the compile-proof config and the Helios torch fix |
| Config | [configs/stage_1c/dense_h20_lr_grid.yaml](../../configs/stage_1c/dense_h20_lr_grid.yaml): one sbatch array of 3 runs, n ∈ {8, 9, 10} |
| Cluster | Helios: partition `plgrid-gpu-gh200`, account `plgllmefficont3-gpu-gh200`, one whole node of 4 GH200 per run (`gres: gpu:4`), 32 CPUs and 109G per GPU from [configs/_cluster/helios.yaml](../../configs/_cluster/helios.yaml). The partition allows 2 days, so the 24 h limit is accepted. FineWeb is read from the `$SCRATCH` copy while its stamp is fresh; each job log prints `FineWeb: <dir>` |
| W&B | `ideas_cv/nano-ame`, runs `s1c_d_h20_lr<n>`, tag `stage_1c` |
| Recipe | Stage 1b's, with the plan's changes: h_balanced h20 (20 blocks, 20 heads, 10 KV, d_model 1,280, d_head 64, SwiGLU d_ff 5,120); 15,259 steps × 1,048,576 tokens = 16.00B tokens; warmup 152 steps; FSDP2 over 4 GPUs with `reshard_after_forward: false` and bf16 from its mixed precision (`autocast_bf16: false`); micro-batch 16 × 8 per GPU; seed 123; eval every 100 steps on 10 batches; `final_ce` on the fixed held-out set of 8192 × 2048 tokens; the final checkpoint only; 24 h limit |
| Data order | `world_size_independent: false`, so the order depends on the GPU count. Every Stage 1c–2c run uses 4 GPUs and shares one order, which is not the order of Stages 1–2b |

**Pre-flight, 2026-11-08 00:55, on `683c316b`'s files.** All three configs compose, every
grid point passes `run_exp.py`'s schema check, and S2c-PS without a learning rate is
refused, as intended. The repo's hygiene, typed-config and pre-commit tests pass. Every
model builds at the plan's sizes, counted on the meta device:

| | S1c-D, dense | S2c-PS, AME | Plan |
|---|---:|---:|---|
| Total parameters | 620,350,720 | 3,163,200,720 | 620.4M, 3.16B |
| Backbone, non-embedding, non-bank | 491,572,480 | 458,857,680 | 491.6M, 458.9M |
| Q/K banks | — | 2,575,564,800 | 2,575.6M |
| Embedding + LM head, untied | 128,778,240 | 128,778,240 | 128.8M |
| Training FLOPs per token, seq 2048 | 3.9646 G | 3.7683 G | 3.965 G, 3.768 G |
| `bank_gate` | — | `none` | `none` |

16B tokens is 25.8 tokens per dense parameter (21.8 in Stage 1, 22.6 in Stage 1b).

## The Helios compile fix

**What was broken.** Helios's aarch64 env had torch 2.6.0+cu126, whose wheel declares no
Triton. So every config with `compile_blocks_full`, and AME's flex attention, died at the
first forward (job 25828493, 2026-11-07).

**The fix, as the plan names it** (`683c316b`, [pixi.toml](../../pixi.toml)):
- `linux-aarch64` torch → `==2.7.1` from `https://download.pytorch.org/whl/cu128`, the
  version lem and entropy run. Its aarch64 wheel declares `triton==3.3.1` and bundles its
  CUDA libraries (the `nvidia-*` dependencies are x86-only).
- **triton `==3.3.1` is pinned from the same index for aarch64.** PyPI has no aarch64
  wheel of triton 3.3.1; the PyTorch index hosts one. Without the pin, a later re-solve
  would look for triton on PyPI first and fail on aarch64.
- `linux-64` and `osx-arm64` are unchanged.

**How the lock was made, at v6.** Helios's pixi cannot be trusted to read v7, so the lock
stays at v6 (`.githooks/pre-commit`, `tests/test_repo_hygiene.py`). The local pixi
(0.81.0) writes v7, and fetching an older pixi binary was refused by this session's
permission rules. So the lock was edited by hand, with exactly the changes pixi's own
re-solve reports:
1. `pixi lock --dry-run --json` on the new manifest and the old lock, in a scratch copy:
   three `linux-aarch64` changes and nothing else — sympy 1.13.1 → 1.14.0 (torch 2.7.1
   needs ≥ 1.13.3), torch 2.6.0+cu126 → 2.7.1+cu128, triton 3.3.1 added.
2. Those three edits, with the hashes and the dependency lists pixi reported. The aarch64
   wheels' metadata is identical to the x86 wheels' already in the lock.
3. `pixi lock --dry-run --json` on the new manifest and the edited lock: no changes.

The aarch64 set is now linux-64's resolution without the x86-only `nvidia-*` packages.

**Environment rebuild, 00:56–00:59.** `update_pixi.py` from the worktree, with the
compile-proof config (so `--qos=now`, which started it at once). It copies `pixi.toml` and
`pixi.lock` to `$PLG_GROUPS_STORAGE/plggllmeffi3/pc/plgmstefaniak/pixi` (the old pair is
kept in `old_pixi_files/obsolete_since_2026_11_08_00_56_51/`) and runs `pixi install` on a
compute node, as srun job 25860316. Helios's pixi read the v6 lock as up to date, installed
it and left it at v6. The env now holds torch 2.7.1+cu128, triton 3.3.1 and sympy 1.14.0.

**The proof.** The plan's 1-GPU `--qos=now` smoke run:
[configs/stage_1c/helios_compile_smoke_h16_ps.yaml](../../configs/stage_1c/helios_compile_smoke_h16_ps.yaml),
Stage 2b's S2b-PS file moved to Helios and cut to 60 steps (`compile_blocks_full` and flex
attention, both Triton). Job 25860605, submitted at 01:00:06 and started within a minute on
x1002c6s0b0n0; it reads FineWeb from the `$SCRATCH` copy.

**Compile works on Helios.** The job log reads `torch.compile applied to 16 whole blocks`,
and the run (W&B `7f1uzqtu`) trained through its first forward and on: step 10 at 01:13,
loss 11.25 → 8.06, 6.35 s/step on one GH200 (S2b-PS's LEM smoke: 6.765 s/step on an H100).
The W&B key reached it through the sbatch environment, as described below. So both stages
stay on Helios. The run completed at 01:21:33 (exit 0:0, 21 min 24 s in all):

| Check | Helios, 1 GH200, n = 9 | S2b-PS's LEM smoke, 1 H100, n = 10 ([stage_1b_report.md](stage_1b_report.md)) |
|---|---|---|
| Start to step 0 | 12 min 11 s (01:00:09 → 01:12:20): env, FineWeb from `$SCRATCH`, model build, the first step's compile | — |
| s/step, median over steps 15–59 | **6.398** | 6.765 |
| MFU, as the trainer logs it | 35.05% | 33.2% (offline, 989 TFLOP/s) |
| Peak memory | 71.92 GiB of the GH200's 96 GB | 71.92 GiB of 93.6 |
| AME readouts in W&B | yes, 240 keys (1 GPU, no FSDP2) | yes |
| `final_ce` after 60 steps | 7.1602 | 6.9980 |
| Checkpoint | written at 01:20:29–01:20:48: a 4,055,929,647-byte model file and 19,780,378,103 bytes of bank state, 23.84 GB; deleted at 01:22 | 23.84 GB |

- **One GH200 runs this config 5.4% faster than one LEM H100.** The plan's h20 estimates
  scaled entropy's speed up by LEM's ×1.10–1.14, so the 4-GPU smokes will likely come in
  under them.
- The 60-step `final_ce` only shows that the end-of-run path works; the two runs used
  different n.

## Submission: how the W&B key reaches the jobs

`run_exp.py`'s remote half writes the W&B key into the clone's `exp.job` with a remote
`sed`, which puts the key in a process's arguments on the login node and in a file. On
Helios the cemetery is group-readable (`2755`, group `plggllmeffi3`), so the key would be
readable by the whole group. So every Stage 1c–2c job is submitted in two halves:
- **Local half: `run_exp.py`'s own functions**, called in its order from the worktree: compose,
  `create_grid_config`, `check_point` on every point, `dump_grid_configs`,
  `generate_sbatch_script`, `version_code` (which pushes the cemetery branch). The secrets
  are unset locally, so `exp.job`'s two export lines are written commented out.
- **Remote half, by hand:** clone the cemetery branch into the cemetery dir, check that
  `exp.job` exports no placeholder, then `sbatch exp.job` from a shell that read the key
  from stdin. The key is in no argument list and no file. sbatch passes it to the job in
  its environment (`--export=ALL` is the default, and Helios sets no `SBATCH_EXPORT`), and
  `srun` passes it on to the ranks.

The clone, the configs and `exp.job` are otherwise what `run_exp.py` would produce. No tmux
session is made, so the run-name dot problem of Stages 2 and 2b
([stage_2_report.md](stage_2_report.md), decision 2) does not arise.

## Smoke runs

**Submitted at 01:14, together with the real runs (decision 3)**, with
`trainer.n_steps=60 trainer.eval_interval=20`, named `smoke_*`, tagged `smoke`, with a
1.5 h limit, as in Stages 1 and 1b ([stage_1_report.md](stage_1_report.md), decision 2):
- **dense:** the S1c-D grid file, 3 points, job 25862155 (`smoke_s1c_d_h20_lr<n>`);
- **AME:** the S2c-PS file with `trainer.learning_rate=9`, job 25862201
  (`smoke_s2c_ps_h20_lr9`), recorded in [stage_2c_report.md](stage_2c_report.md).

The checks, as the plan lists them: micro-batch 16 fits under FSDP2 on Helios; the s/step
against the plan's estimate (2.60–2.70 dense, 3.14–3.25 AME); `final_ce` is logged and
the checkpoint written; AME's readouts fail with a warning per layer and the run goes on.
Results: pending.

## Jobs

| Job | What | n | W&B run | Node | Submitted | Started | Ended | State | `final_ce` |
|---|---|---:|---|---|---|---|---|---|---:|
| 25860316 | `update_pixi.py`: rebuild the Helios env (srun, `--qos=now`, 4 CPUs, no GPU) | — | — | — | 00:56 | 00:57 | 00:59 | completed | — |
| 25860605 | Compile proof: S2b-PS on 1 GH200, 60 steps, `--qos=now` | 9 | `7f1uzqtu` | x1002c6s0b0n0 | 01:00 | 01:00 | 01:21 | completed | 7.1602 (60 steps) |
| 25862155_0–2 | 4-GPU smoke: the S1c-D grid file, 60 steps | 8, 9, 10 | | | 01:14 | | | pending | |
| 25862221_0–2 | **S1c-D grid** | 8, 9, 10 | | | 01:15 | | | pending | |

## Decisions

1. **Stage 0's faster code is not used.** The plan says Stage 0's later code (about 12%
   faster at h20 on 4 GPUs, on `ame_optimize`) shortens Stage 2c only if it is merged
   before Stage 2c's first run. On 2026-11-08 at 00:50 `ame_optimize` (`9623ada2`) is not
   merged into `ame_main`, so both stages run `ame_main`'s code.
2. **AME's readouts under FSDP2 are still not logged on `ame_main`**
   ([src/core/trainer.py](../../src/core/trainer.py), `_log_layer_instrumentation`: one
   warning per layer, then dropped). So Stage 2c reads α₂, β₂ and τ from its final
   checkpoints, as the plan says.
3. **The 4-GPU smoke runs and the real runs were submitted together, at 01:14–01:16,
   once the compile proof had trained 10 steps.** The plan's Helios section: prove the fix
   with the 1-GPU run, "then submit the 4-GPU jobs without delay". 4-GPU jobs queue by age
   under normal QoS (a grant job waited 25 h on 2026-11-07), so holding the grid back until
   the smokes finished could cost a day of queue. The smokes have a 1.5 h limit, which the
   backfill scheduler can fit into gaps, so they are likely to start, and finish, before the
   24 h runs. If a smoke fails, the queued runs are mine to cancel, fix and resubmit, at no
   GPU cost. The compile proof had by then shown what the 4-GPU runs share with it: the
   Helios env, compile and flex attention, FineWeb from `$SCRATCH`, and the W&B key.
4. **Stage 2c's n = 9 run was submitted with the Stage 1c grid**, as the plan allows: n = 9
   is in Stage 2c's grid whichever of 8, 9 or 10 Stage 1c picks
   ([stage_2c_report.md](stage_2c_report.md)).
5. **The jobs stay queued on Helios after 8 h 15 min without a start (09:30).** Every job
   is pending with reason `Priority`. The partition is full:
   - of its 110 nodes, every usable one has all 4 GPUs allocated, and most are planned for
     higher-priority jobs;
   - Helios hides other users' jobs (`PrivateData=jobs`), so the queue ahead of ours
     cannot be seen.

   Our priority is 11,612:
   - fair-share 9,986 of 10,000 and partition 1,000;
   - age 563, rising to 5,000 at 3 days;
   - job size 63.

   QoS `normal` adds nothing. The account's only other QoS, `now`, allows 1 GPU. The plan
   expects this wait: a grant job waited 25 h on 2026-11-07. It names LEM only for the
   case where compile still fails, and keeps both stages on one cluster. So nothing is
   moved or resubmitted; the jobs keep aging in the queue.

   At 11:30 the backfill scheduler gave its first forecasts:
   - the dense smoke runs (25862155) at 2026-11-10 05:05;
   - the AME smoke run (25862201) at 05:17;
   - none for the 24 h runs.

   These forecasts assume every running job uses its full time limit and no
   higher-priority job arrives, so they are rough in both directions.

## For Maciej

- **The Helios queue may cost about two days before the first run starts.** At 11:30 on
  2026-11-08, Slurm forecast the smoke runs for Tuesday 05:05 and gave no forecast for the
  24 h runs (decision 5). The plan allows LEM only if compile fails, so the runs stay on
  Helios. Moving them to LEM (`gres: gpu:hopper:4`, r16-11 excluded) is your call.
