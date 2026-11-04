# AME training-efficiency plan — three tiers

**Status:** plan only, nothing implemented. Written 2026-11-04 against branch
`ame_optimize` (HEAD `f302ae12`).
**Companion:** [ame_optimization.md](ame_optimization.md) — the full 31-item survey of
pkmf's static-memory stack. This document is the subset we committed to, in order, with
the experiments that decide each step.

**Target:** AME **h20** on the **entropy** cluster, **4×H100**. Metric is **MFU per GPU**
(GPU-count invariant); tokens/s is reported alongside it.

---

## Goal

Raise AME **h20** training MFU on **4×H100** by executing Tiers 1→3 in order, measuring
every step on the entropy `h100` partition, and keeping the mathematics unchanged
throughout.

- **Tier 1** establishes a trustworthy 4-GPU baseline (B1/B2), lands the gradient-sync
  fix (O9), and produces the per-submodule attribution table. That table is what
  *authorises* Tiers 2 and 3 — if the bank gradient path is not a top-two term after O9,
  re-rank before building anything.
- **Tier 2** removes the 2.576B bank parameters from FSDP2's dense gradient and
  optimizer path (O1–O5).
- **Tier 3** shards those banks vertically across the four GPUs (O6–O8).

**Done** = each tier's exit criteria met, every run recorded in the §3 ledger, and a
measured MFU/GPU delta against B1 reported per tier. A tier that measures *worse* is a
completed tier with a negative result, not a failure — record it and stop, do not tune
past the gate.

**Not in scope:** anything listed in §0 as cut (O10, O11, O16, O17) or deferred
(O14/O15, O19/O20, O21–O27). Do not reintroduce them to hit a number.

---

## 0. Scope: what is in, and what we deliberately left out

Everything here raises **MFU** — compute efficiency. Items whose only effect is peak
memory were cut, by decision on 2026-11-04, even where pkmf ships them:

| Cut | pkmf item | Reason |
|---|---|---|
| Activation checkpointing | O10 | Trades ~30% recompute for memory. Lowers MFU by construction. |
| Liger fused-linear-CE | O11 | Removes the logit tensor; a memory win, not a FLOP win. |
| `ChunkedAdamW` | O16 | Bounds optimizer temporaries. Memory only. |
| CPU-offloaded optimizer moments | O17 | Host↔device transfer is strictly slower than staying resident. Ruled out outright — **nothing gets offloaded to CPU in this plan.** |
| Fused/deduplicated AME gather | O14/O15 | Deferred (former tier 4). Highest ceiling but must be written from scratch — `pkmf.memory.*` is not in the drop. |
| Per-group LR, 4× bank multiplier | O19/O20 | Deferred. Quality, not throughput. |
| Batch ramp, token-space WSD, branch-decay | O21–O27 | Deferred. Experiment throughput, not MFU. |

**Consequence to accept up front:** with activation checkpointing off, the micro-batch is
bounded by what fits. At 1 GPU that was 4 against the dense baseline's 16. At 4 GPUs
FSDP2 shards parameters, gradients and optimizer state four ways, so the bound is much
looser — how much looser is the first thing Tier 1 measures (§2.3).

What remains, in dependency order:

| Tier | Items | Mechanism |
|---|---|---|
| **1** | baseline + **O9** + **O31** | Measure honestly; delete the redundant gradient traffic |
| **2** | **O1–O5** | Take the banks out of FSDP2's gradient/optimizer path |
| **3** | **O6–O8** | Shard the banks vertically across the 4 GPUs |

Tier 2 is a hard prerequisite for Tier 3: vertical sharding needs a hand-managed
`main_grad` to reduce in a subgroup.

---

## 1. Test bed

### 1.1 Cluster

**entropy**, partition `h100` — a single node `h100a`, 8×H100. It is the only node with
`/storage_nvme_4`, which holds FineWeb, the pixi env and `HF_HOME`. Nothing in this plan
runs on `a100` or `common`.

- ssh alias is **`entropy_j`** (the config's `server: entropy` fails auth — override it).
- Only `~/.ssh/id_rsa` authenticates as `j321m`. Rapid retries trip fail2ban (~1 min);
  retry at most once a minute, never in parallel.
- `gres` syntax is `gpu:h100:N`; `configs/_cluster/entropy.yaml` leaves it `???`, so the
  experiment config **must** set it.
- Defaults from `_cluster/entropy.yaml`: `cpus_per_gpu: 14`, `mem_per_gpu: 90G`,
  `partition: h100`, `nodes: 1`.
- The GPU cap is **shared across the whole account** — these jobs compete with other
  people's sessions. Check with
  `sacctmgr -n -P show qos <qos> format=Name,MaxTRESPerUser`.
- GitHub:22 worked from the login node as of 2026-11-03 (verified by an unattended
  submission, job 32306). Re-check before assuming otherwise.

### 1.2 Submission

```bash
./.pixi/envs/default/bin/python run_exp.py \
    --config-path configs/ame --config-name <bare-name>
```

Per-directory `--config-path`, bare `--config-name`, and **never via `pixi run`** — it
rewrites `pixi.lock` v6→v7.

Rules that have cost hours before and apply to every submission in this plan:

1. **`version_code` `git add`s everything, including untracked files**, and pushes it to
   the cemetery branch before restoring your tree. `pkmf/` is already gitignored
   (`.gitignore:203`); check for new scratch before each submit.
2. **Submit from a named branch** — `version_code()` reads `repo.active_branch.name`, so
   a detached worktree dies instantly. Submitting from a throwaway branch in a clean
   worktree also stops other sessions' untracked files being swept into the push.
3. **Never SIGTERM `run_exp.py` mid-flight** — the restore is in a `finally` that a
   `timeout` wrapper bypasses, leaving a stray commit and HEAD on the experiment branch.
   Give it a generous timeout and background it.
4. `WANDB_API_KEY` is forwarded from the local shell. If it is missing locally the job
   runs with no logging and you find out hours later.
5. **Submitting is outward-facing — confirm with Maciej before each launch.**
6. **Verify ~2 minutes after submit:** job running, first steps in the log, first metric
   in wandb. Bad paths and init errors all fail inside the first couple of minutes.
7. Use `sbatch`, never an interactive `srun` over SSH — the job dies with the connection.
   A CPU-only `srun` (no `--gres`) still lands on the GPU node, sees `/storage_nvme_4`
   and the env, and costs no GPU quota: use it for path and version checks.
8. If a job sits `Pending`, triage before resubmitting:
   `sbatch --test-only --wrap="true"` predicts the start time and submits nothing.
   `QOSMaxGRESPerUser` is the account cap; `Resources` while GPUs look free usually means
   **RAM** (4 GPUs × 90G = 360G, which one memory-holding session can block).

Results land in wandb `ideas_cv/llm-random-test`, project `nano-ame`.

### 1.3 Workload, held constant across every run in this plan

Base config is `configs/ame/h20_smoke.yaml`. Every measurement keeps:

| | Value |
|---|---|
| Model | AME h20 — `d_model` 1280, 20 blocks, 20 heads, `d_head` 64, vocab 50304 |
| Parameters | **3.163B** (2.576B banks + 587M backbone) |
| FLOPs/token | **3.77 G** (`src/core/mfu.py`; the banks correctly contribute zero) |
| Sequence length | 2048 |
| **Global** batch | **512 sequences** → 1,048,576 tokens/optimizer step |
| Steps | 30, warmup 10; **eval off, checkpointing off** |
| LR | `2^-12` = 0.000244140625 |
| Distributed | FSDP2 from `_trainer/llama.yaml`, `MixedPrecisionPolicy(bf16, fp32)` |
| Compile | `compile_blocks_full: true`; no `autocast_bf16`, no `compile_loss`, no `compile_ce` |

`common.batch_size` is the **global** batch; `get_dataloader` hands each rank
`512 / world_size`. So the micro-batch is set by gradient accumulation:

```
gradient_accumulation_steps = 512 / (ngpu × microbatch_per_gpu)
```

| ngpu | micro-batch 4 | 8 | 16 |
|---|---|---|---|
| 1 | ga 128 | 64 | 32 |
| 4 | ga 32 | 16 | 8 |

**This is the pkmf invariant (ablation §5 of the survey): hold the global token schedule
fixed and vary only the physical split.** Every cell trains the identical 512×2048-token
step, so tokens/s and MFU are directly comparable across GPU counts and micro-batches.

MFU denominator is H100 bf16 dense peak, **989 TFLOP/s**. At 3.77 GF/token:

```
tokens/s per GPU = MFU × 989e12 / 3.77e9 = MFU × 262,335
```

so 18.1% = 47,474 tok/s/GPU and 44% = 115,400 tok/s/GPU. Report both.

### 1.4 Correctness gate — applies to every tier

Every change in Tiers 2 and 3 alters where gradients are reduced and who steps the
parameters. None of them may alter the mathematics. Before any throughput number is
quoted:

1. **CPU unit test on a toy model** (`d_model` 64, 2 blocks, vocab 256, 3 steps): the new
   path and the baseline path must produce **identical parameter tensors** after N steps
   under a fixed seed. A frozen-parameter or clipping bug reproduces here in seconds and
   costs a multi-hour run otherwise.
2. **Loss-trajectory equivalence on the cluster**: same seed, same config, 30 steps,
   baseline vs changed path. Compare per-step loss. Tier 1 (O9) must be **bit-identical**
   — suppressing sync during accumulation changes nothing mathematically. Tiers 2–3 are
   not bit-identical (different reduction order, separate clip) and need a stated
   tolerance plus a matching trajectory shape.
3. **Grad-norm trace** logged every step. A silent divisor error (§Tier 3, O8) shows up
   here as a constant factor and nowhere else.

---

## 2. Baselines

### 2.1 What exists

| Run | GPUs | ga | micro-batch | tokens/s | MFU | step | source |
|---|---:|---:|---:|---:|---:|---:|---|
| **AME h20**, fused kernel | 1 | 128 | 4 | 47,391 | **18.1%** | 22.1 s | entropy job 32360 |
| dense GPT-2-arch h20, FSDP2+compile | 1 | 32 | 16 | 109,361 | 45.57% | 9.58 s | `mfu_audit/h20_mgpu.yaml` |
| dense GPT-2-arch h20 | 2 | 16 | 16 | 214,499 | 44.69% | 4.89 s | ″ |
| dense GPT-2-arch h20 | 4 | 8 | 16 | **425,054** | **44.28%** | 2.47 s | ″ |

### 2.2 The baseline caveat, stated precisely

The multi-GPU dense table above is the **GPT-2-arch** h20 (4.122 GF/token), produced by
adding `GeluFeedForward`, `n_blocks=25` and `kv_heads=q_heads` overrides. The
**H-balanced** h20 that `configs/ame/h20_smoke.yaml` is built on is 3.97 GF/token and
**has never been measured multi-GPU**. AME itself is MHA-shaped at 3.77 GF/token.

So there are three different denominators in play. MFU percentages are comparable across
them (each uses its own FLOPs/token); raw tokens/s are **not**.

### 2.3 Baselines this plan must produce first

**B1 — AME h20 at 4 GPUs, micro-batch sweep.** The number every later result is read
against, and the test of whether the micro-batch-4 bound was a 1-GPU artefact.
Config: copy `configs/ame/h20_smoke.yaml` → `configs/ame/h20_4gpu.yaml`, set
`gres: gpu:h100:4`, and submit three variants:

| variant | `gradient_accumulation_steps` | micro-batch/GPU |
|---|---:|---:|
| `h20_4gpu_mb4` | 32 | 4 |
| `h20_4gpu_mb8` | 16 | 8 |
| `h20_4gpu_mb16` | 8 | 16 |

Record tokens/s, MFU/GPU, step time, **peak GiB/GPU**. Expect `mb16` to fit: at 4 GPUs
FSDP2 shards the 3.163B parameters, fp32 masters, fp32 grads and both Adam moments four
ways, so the ~57 GiB that forced micro-batch 4 at 1 GPU becomes ~14 GiB/GPU of resident
state. If `mb16` OOMs, record where and fall back; that is a finding, not a failure.

**B2 — dense H-balanced h20 at 4 GPUs**, micro-batch 16, ga 8, as the honest
same-architecture comparator. One run, `configs/mfu_audit/h20_mgpu.yaml` with
`gres: gpu:h100:4` and `gradient_accumulation_steps: 8`, **no** GPT-2-arch overrides.

Everything downstream is reported as a delta against B1, with B2 as the ceiling.

---

## Tier 1 — Measure, and delete the redundant gradient traffic

**Goal:** a trustworthy 4-GPU AME baseline, the per-submodule attribution that tells us
where the remaining time goes, and the one free throughput fix.

### T1.1 — Baselines B1 and B2

As §2.3. No code change. Gate: B1 recorded at all three micro-batches with peak memory,
B2 recorded.

**Configs written and pre-flighted 2026-11-04** (compose + `check_point` + `instantiate`
on the `meta` device, all four clean):

| run | config | ga | mb/GPU | params | wandb name |
|---|---|---:|---:|---:|---|
| B1/mb4 | `configs/ame/h20_4gpu_mb4.yaml` | 32 | 4 | 3.163B | `ame_h20_4gpu_mb4` |
| B1/mb8 | `configs/ame/h20_4gpu_mb8.yaml` | 16 | 8 | 3.163B | `ame_h20_4gpu_mb8` |
| B1/mb16 | `configs/ame/h20_4gpu_mb16.yaml` | 8 | 16 | 3.163B | `ame_h20_4gpu_mb16` |
| B2 | `configs/mfu_audit/h20_hbal_4gpu.yaml` | 8 | 16 | 0.620B | `mfu_h20_hbal_4gpu_mb16` |

All four are `gres: gpu:h100:4`, 30 steps, global batch 512 × 2048, eval and
checkpointing off, `compile_blocks_full: true`, project `nano-ame`.

The submission command needs one CLI override that is deliberately **not** in the
committed config — `infrastructure.server=entropy_j`. `_cluster/entropy.yaml` ships
`server: entropy`, which is the `mstefaniak` alias and fails auth; `entropy_j` is
`j321m`. Personal account aliases stay out of shared config.

```bash
./.pixi/envs/default/bin/python run_exp.py \
    --config-path configs/ame --config-name h20_4gpu_mb4 \
    infrastructure.server=entropy_j
```

**Metrics come from the trainer, not from arithmetic.** `_log_throughput`
(`src/core/trainer.py:716`) already logs `train/step_time`, `train/tokens_per_s` and
`train/mfu`, and the MFU denominator is correctly the **global** peak (per-GPU peak ×
world size), so `train/mfu` is already MFU *per GPU*. `train/tokens_per_s` is global.
`train/max_memory_gib` is logged once at the end of training, from rank 0. The job log
also prints one `MFU accounting:` line naming the measured GFLOPs/token, which is how
§2.2's three denominators get confirmed rather than assumed.

### T1.2 — O9: suppress gradient sync during accumulation

**What.** `src/core/trainer.py:549` `calculate_loss` loops `loss.backward()` over
`gradient_accumulation_steps` chunks with no sync control. Under FSDP2 every one of those
backwards triggers a full reduce-scatter of all 3.163B gradients, in **fp32**
(`reduce_dtype=torch.float32`, `src/core/distributed_training.py:66`). Only the last one
is needed.

**Change.** Disable gradient sync for micro-steps `0 .. GA-2`, enable it for the last —
FSDP2's `set_requires_gradient_sync(bool)` on the root module, guarded so non-FSDP2 and
CPU paths are unaffected. Roughly 10 lines in `calculate_loss`.

**Why it is first.** Traffic estimate at B1's `mb4` shape (ga 32), fp32 reduce-scatter of
3.163B parameters ≈ 12.65 GB moved per micro-step:

| | reduce-scatters/step | fp32 bytes moved/step |
|---|---:|---:|
| today, 4 GPU, ga 32 | 32 | ~405 GB |
| today, 1 GPU, ga 128 | 128 | ~1.6 TB (local copy + cast, no NCCL) |
| after O9 | **1** | ~12.65 GB |

These are estimates from parameter count and dtype, not measurements — T1.3 is what
converts them into attributed time. But the ratio is 32:1 and the fix is ten lines.

**Gate.** Bit-identical loss trajectory vs B1 over 30 steps (it must be — this changes
only *when* gradients are reduced). Then re-run the B1 micro-batch sweep and report the
delta.

**Scope note.** This is a framework fix, not an AME fix. It benefits every large model in
nano and should be reviewed as such.

### T1.3 — O31: per-submodule forward/backward attribution

**What.** CUDA-event pairs around each AME sub-block — bank gather, QK-norm, the KV-shift
mix, `flex_attention`, `v_proj`/`o_proj`, and the CE — reporting forward and backward
milliseconds over a fixed step window (pkmf profiles steps 5–55; 30-step runs give us
5–25). Model after `pkmf/src/poirot/callbacks_profiling.py`.

nano already has `LoopTimers` (forward/backward/clip/optim/log) and a kernel profiler
(`kprof_us` / `kprof_op`). Neither attributes time to a *submodule*, which is exactly what
is missing: 2.5× of AME's gap is currently unattributed.

**Gate.** An attribution table for AME h20 at 4 GPUs that accounts for ≥90% of step time.
**This table decides whether Tiers 2 and 3 are worth their cost** — if the banks'
gradient and optimizer path is not a top-two term after O9, we re-rank before building
anything.

### Tier 1 exit criteria

- B1 (3 micro-batches) and B2 recorded in this document.
- O9 landed, bit-identical, with its measured delta.
- Attribution table covering ≥90% of step time, with the bank grad/optimizer path
  quantified.

---

## Tier 2 — Hand-managed bank gradient and optimizer path (O1–O5)

**Goal:** stop FSDP2 and AdamW touching 2.576B of bank parameters densely on every
micro-step. pkmf measured **1.43×** end-to-end for this mechanism alone on 1 GPU.

### What gets built

A controller modelled on `pkmf/src/poirot/memory_parallel.py:696`
`MemoryParallelController`, owning every `q_bank.weight` / `k_bank.weight`:

| | Mechanism | pkmf reference |
|---|---|---|
| **O1** | For each bank: set `requires_grad=False` so FSDP2 creates no master, no reduce and no step; attach a **persistent fp32 `main_grad`**; keep a **private fp32 master** that a dedicated AdamW steps. The backward scatters into `main_grad` and never zeros it, so it accumulates across micro-steps for free. | `memory_parallel.py:696` |
| **O2** | One `all_reduce(SUM)` of `main_grad` per optimizer step, then `div_(world_size)` to preserve mean semantics. | `memory_parallel.py:1024` |
| **O3** | fp32 master → bf16 model param copy after each step. The forward reads the bf16 param. | ″ |
| **O4** | Separate `clip_grad_norm_` over the masters. | ″ |
| **O5** | Three hook points in the step loop: *step begin* → re-home master and `main_grad` onto the param's device; *after backward, at the accumulation boundary* → all-reduce and step; *log* → surface the bank LR. | `callbacks.py:3347` |

### The AME-specific problem Tier 2 must solve

pkmf's banks are read by a **custom fused kernel** that `atomicAdd`s straight into
`main_grad`. AME's banks are `nn.Embedding` (`src/core/ame.py:269-270`), whose backward is
`embedding_dense_backward` writing to `.grad`. With `requires_grad=False` there is no
`.grad` and **no gradient at all** — autograd simply stops.

Three candidate routes, to be decided by a short spike before the full build:

| Route | Mechanism | Risk |
|---|---|---|
| **A** | Keep `requires_grad=True`, hide the banks from FSDP2 by excluding them from the sharded module set, and install a post-accumulation hook that folds `.grad` into `main_grad` | Smallest change; does not remove the dense `.grad` allocation, so it captures O2–O5 but only part of O1 |
| **B** | Custom `autograd.Function` wrapping the gather, whose `backward` scatters into `main_grad` via `index_add_` and returns no parameter gradient | Removes the dense per-micro-step grad; `index_add_` on a 2.576B fp32 buffer is the thing to benchmark |
| **C** | As B, with the scatter fused into a custom kernel | This is O14; **out of scope**, deferred |

**Route B is the expected choice**, with A as the fallback if the spike shows
`index_add_` is not competitive. Decide on measurement, not on preference.

### Measurement

Re-run the B1 micro-batch sweep at 4 GPUs, plus one 1-GPU run to compare directly against
pkmf's 1.43× claim on a shape where no NCCL traffic confounds it.

### Gates

- CPU toy-model equivalence (§1.4.1) — parameters identical after 3 steps.
- 30-step loss trajectory matching B1 within a stated tolerance, with grad-norm traces
  overlaid.
- Throughput delta reported at 1 and 4 GPUs, micro-batch held.

### Risk register

- **`index_add_` cost** (route B) — 2.576B fp32 target; benchmark standalone before
  committing.
- **Interaction with `compile_blocks_full`** — a custom `autograd.Function` inside a
  compiled region may graph-break. Measure with and without.
- **Interaction with the existing clip** — `src/core/trainer.py:649` computes a correct
  global norm over DTensor shards. Splitting the clip changes the math. Either clip
  jointly (gather both norms) or document the change as deliberate; do not let it happen
  silently.
- **The bank AdamW is a single fused/foreach step over 2.576B params.** pkmf hit an OOM
  here with foreach and solved it by chunking (O16), which we cut. If it OOMs, the
  minimal unblock is `foreach=False, fused=False` on that optimizer only — not a
  reintroduction of chunking.

---

## Tier 3 — Vertical bank sharding across the 4 GPUs (O6–O8)

**Goal:** cut per-rank bank VRAM and bank gradient comm by `mem_parallel_size`.
pkmf measured **~2.2×** for O1+O6+O7 together on their 4-GPU grid (215k vs 97k tok/s),
against ~1.00× for the hand-managed path *without* sharding — i.e. **on multi-GPU this
tier is where the win actually is.**

### What gets built

| | Mechanism | pkmf reference |
|---|---|---|
| **O6** | Two process groups at init. **Memory-parallel group**: contiguous blocks of `mem_parallel_size` ranks; each owns a column slice `[vocab, width/mp]` of every bank. **Memory-DP subgroup**: strided, size `world/mp`; `main_grad` is all-reduced *only* here. At `mp == 1` every helper is identity. | `memory_parallel_groups.py:43` |
| **O7** | Forward: all-gather token ids within the mp-group → local gather on the owned slice for all ranks' tokens → all-to-all so each rank ends with its own tokens at full width. Backward mirrors it. Autograd wrappers: all-gather fwd / reduce-scatter bwd; all-to-all with swapped splits on bwd. | `memory_parallel_groups.py:192, 229` |
| **O8** | **The divisor.** The all-reduce runs in the mem-DP subgroup (size `world/mp`), but the divisor stays the **global** `world_size` — the forward's id all-gather already summed `mp` ranks' tokens into each slice's local gradient, so `mp × (world/mp) = world`. | `memory_parallel.py:1024` docstring |

At `world=4, mp=4`: one mp-group `[0,1,2,3]`, four singleton dp-groups → the bank
all-reduce disappears entirely and each rank holds a quarter of every bank.

### Why the column split fits AME better than it fits pkmf

AME's bank is `[vocab, q_heads × d_head]`. A **column** slice is therefore a **head
subset**, and heads are independent until `o_proj`. Each rank can serve any token id from
its own slice, so no rank ever needs a row it does not own.

This is a different choice from the one `ame.md §7` anticipated ("shard them the way
Megatron shards a vocab-parallel embedding — rows split across ranks"). The row split
would need every rank to fetch arbitrary rows; the column split does not. **Record this
as a deliberate deviation from the design record.**

Open design question for the spike: whether to all-to-all the gathered rows back (pkmf's
scheme, keeps attention unchanged) or to let each rank run attention for its own heads and
all-gather only before `o_proj` (cheaper, but it is tensor parallelism of the attention
layer and interacts with FSDP2 and `flex_attention`). **Default to pkmf's scheme**; the
second is a separate investigation.

### Measurement

Sweep `mem_parallel_size ∈ {1, 2, 4}` at 4 GPUs with micro-batch held at B1's best,
reporting tokens/s, MFU/GPU and peak GiB/GPU for each.

### Gates

- Loss trajectory at `mp ∈ {1,2,4}` must coincide — this is the test that catches an O8
  divisor error, which otherwise shows up as a constant factor on the grad norm and
  nothing else.
- Checkpoint save/load at `mp=4` round-trips to an evaluable model.
- Throughput table across `mp` and micro-batch.

### Risk register

- **O8's divisor** — the single most likely silent bug in this plan. Assert it in a unit
  test comparing `mp=1` and `mp=4` gradient magnitudes on the toy model.
- **`overlap_comm` is a trap with vertical sharding** (O13). pkmf measured overlap
  *halving* VS throughput (215k → 123k). The FSDP2 instinct is to add overlap; measure
  before enabling anything of the sort.
- **Checkpointing.** 3.163B parameters is minutes, not seconds, to gather and write. The
  benchmark configs keep checkpointing off, so this only bites when a real training run
  follows. pkmf's answer is rank-local shards (O30) — out of scope here, flagged as the
  follow-up that real runs will need.
- **FSDP2 interaction.** The banks are excluded from FSDP2 from Tier 2 onward, so Tier 3
  shards them independently. The two sharding schemes must not both claim the same
  parameters; assert it at construction.

---

## 3. Experiment ledger

Every run recorded here as it completes: job id, config, wandb run, tokens/s, MFU/GPU,
peak GiB/GPU, step time. Cancelled runs recorded as "ignore" so later comparisons do not
pick them up.

| # | Tier | Config | GPUs | ga | mb | job | tokens/s | MFU/GPU | peak GiB | notes |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| | | | | | | | | | | *(empty — nothing submitted yet)* |

Reference points to carry into it: AME 1-GPU = 47,391 tok/s / 18.1% / 22.1 s (job 32360);
dense GPT-2-arch 4-GPU = 425,054 tok/s / 44.28% / 2.47 s.

---

## 4. Decision log

- **2026-11-04** — Three-tier scope fixed. O10, O11, O16, O17 cut as memory-only; any
  form of CPU offload ruled out outright. Former tier 4 (O14/O15 fused gather, O19/O20
  per-group LR, O21–O27 schedule machinery) deferred, not rejected.
- **2026-11-04** — Target platform fixed: entropy `h100` partition, 4×H100, AME h20.
- **2026-11-04** — Tier 1 configs written as **three separate files** rather than one
  `^gradient_accumulation_steps` grid. A caret key varies one subtree only, and the
  accumulation count (`trainer.`) and the wandb run name
  (`infrastructure.metric_logger.`) live in different subtrees — a grid would have
  produced three identically-named runs, which the §3 ledger cannot disambiguate.
- **2026-11-04** — **Preemption is a live hazard on `h100` and gates every 4-GPU
  submission.** Probed read-only with `sbatch --test-only --wrap="true"`:

  ```
  PreemptType = preempt/qos      PreemptMode = CANCEL
  GraceTime   = 0                PreemptExemptTime = 0
  sacctmgr: j321m_h100 | priority 0 | Preempt=lukaskoz_h100
  ```

  Our QOS is configured to preempt `lukaskoz_h100`, and the mode is **CANCEL with zero
  grace**. At the time of probing h100a had 5 of 8 GPUs allocated, so a 4-GPU job would
  have started immediately *by cancelling* jobs 32498 (`ch_0_2048`) and 32499
  (`ch_2048_4096`), both `lukaskoz`, both 2 h 13 m into a 24 h limit. A **3**-GPU job
  started immediately with no preemption; a 1-GPU job likewise.

  **Standing rule for this plan:** before every 4-GPU submission, run the `--test-only`
  probe and read the `Preempts:` line. If it names another user's job, do not submit —
  wait for free capacity, or get an explicit decision. The plan's test bed is fixed at
  4 GPUs, so dropping to 3 is not an option: 512/(3 × mb) is not an integer and the
  global token schedule would stop being held constant.
- **2026-11-04** — Earlier claim in [ame_optimization.md](ame_optimization.md) §8 that
  "none of the h20 baselines is at 4 GPUs" is **wrong**: `mfu_audit/h20_mgpu.yaml` carries
  measured 1/2/4-GPU numbers. The real gap is narrower — those numbers are the GPT-2-arch
  variant (4.122 GF/token), not the H-balanced h20 (3.97 GF/token) that AME's config
  builds on, and AME itself is 3.77 GF/token. Hence B2.
