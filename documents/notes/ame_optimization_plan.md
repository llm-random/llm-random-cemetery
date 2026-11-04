# AME training-efficiency plan — three tiers

**Status: Tier 1 COMPLETE. Tiers 2 and 3 STOPPED at the gate.** Written 2026-11-04
against branch `ame_optimize` (HEAD `f302ae12`). B1/B2 measured, O9 measured and
**rejected on this test bed** (−2.53%), O31 attribution produced. The bank
gradient/optimizer path is **~1.3–5% of step time, not a top-two term**, so by the Goal's
own instruction Tier 2 does not proceed and Tier 3 — gated on it — does not either.
See "Tier 1 result" below §3.
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

**Gate — corrected 2026-11-04.** The plan originally demanded a **bit-identical** loss
trajectory. That gate is unreachable at 4 GPUs and asking for it would have sent us
chasing a non-bug:

```
sync on   ->  rank r ends with   sum_i  shard_r( sum_ranks g_i )
sync off  ->  rank r ends with   shard_r( sum_ranks sum_i g_i )
```

Same terms, different association order. Floating-point addition is not associative, so
the two agree exactly only when there is no cross-rank sum to reassociate — i.e. at
world_size 1. The honest gate is therefore **two-level**:

| world_size | gate |
|---|---|
| 1 | **bit-identical** gradients — no reassociation is possible, so any difference is a bug |
| > 1 | equal to **fp32 tolerance**, with grad-norm traces overlaid |

**Landed 2026-11-04.** Opt-in flag `trainer.skip_grad_sync_during_accumulation`
(default `false`), implemented as `Trainer._accumulation_grad_sync`
(`src/core/trainer.py`), declared in `TrainerSchema` (`src/definitions.py`). Inert unless
the model is an FSDP2 module, the flag is set, `gradient_accumulation_steps > 1`, and the
model is in training mode.

**The memory cost, which the plan did not previously state.** O9 is free in FLOPs and
*not* free in memory. With sync suppressed, FSDP2 holds each group's **unsharded**
gradient between micro-steps and upcasts it to the fp32 reduce dtype
(`to_accumulated_grad_if_needed`, `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`).
Resident gradient per rank:

| | per-rank fp32 gradient at 3.163B params, 4 GPUs |
|---|---:|
| today (sharded) | 3.16 GB |
| with O9 (unsharded) | **12.65 GB** |
| difference | **+9.5 GB/GPU** |

Parameters still reshard after backward; only the gradient stays unsharded. At larger
micro-batches this may be what decides O9, not the saved communication — so **peak
GiB/GPU is reported next to tokens/s on every O9 run**, and `mb16` is the one at risk.

**Tests.** `tests/test_grad_sync_accumulation.py`, 6 tests, over a real FSDP2 CPU/gloo
mesh (the pattern `tests/test_distributed.py` established), at world_size 1 and 2, with
and without the production `MixedPrecisionPolicy(bf16, fp32)`.

A **negative control** was run to confirm the tests are not vacuous, and it found the
sharpest failure mode — which is not numeric drift. While sync is suppressed FSDP2 parks
the running sum in `unsharded_accumulated_grad` and leaves `param.grad` as **None**; only
the reducing micro-step materialises it. A loop that never re-enables sync therefore ends
with every `.grad` None — 7 of 7 parameters on the toy model. `clip_gradient`
(`src/core/trainer.py`) filters on `p.grad is not None`, so that state would clip nothing
and step on nothing while the loss kept being logged. The restore lives in a `finally`
for exactly this reason, and the test asserts gradients exist before comparing values.

**Measurement configs** (identical to their B1 twins except the flag, the job name and
the tags): `configs/ame/h20_4gpu_mb{4,8,16}_o9.yaml`. All pre-flighted.

**Scope note.** This is a framework fix, not an AME fix. It benefits every large model in
nano and should be reviewed as such. It is opt-in rather than default-on **because of the
memory cost** — flipping the default would change peak memory for every existing
multi-GPU config in the repo, and that is a separate decision with its own evidence.

### T1.3 — O31: per-submodule forward/backward attribution

**The original design does not work, and fails silently. Corrected 2026-11-04.**
O31 was specified as CUDA-event pairs around each AME sub-block — bank gather, QK-norm,
the KV-shift mix, `flex_attention`, `v_proj`/`o_proj`. Every one of those sits *inside* a
`TransformerBlock`, and `compile_blocks_full: true` compiles each whole block. Measured
with `torch._dynamo.explain` on torch 2.7.1:

```
plain      graphs=1 breaks=0 ops=2
annotated  graphs=1 breaks=0 ops=2
W... Profiler function <class 'torch.autograd.profiler.record_function'> will be ignored
```

The annotation does **not** break the graph — dynamo *drops* it. `ops` is unchanged, so
the annotation is simply absent from the compiled code. Annotating inside a compiled
block therefore yields no attribution at all, with no error and no missing-data signal:
it would have produced an empty table that looked like a broken profiler. A CUDA-event
pair inside the block would instead force a real graph break, measuring a model that is
not the one we ship.

**What replaces it**, all of it outside the compiled region and so compile-safe:

| instrument | what it gives | why it survives compile |
|---|---|---|
| `trainer.profile_timers` | `data_fetch`, `data_h2d`, **`forward`**, **`backward`**, **`clip`**, **`optim`**, `log`, vs `step_total`, as a `timers_ms:` line per step | `LoopTimers` wraps the *call sites* in `train()` and `calculate_loss`, not the model internals |
| `trainer.kernel_profile` | one `kprof_us:` line per CUDA kernel over steps 12–13 | kernel names are observed, not injected |

The gate's "≥90% of step time" is satisfied **by construction** rather than by hope: the
`LoopTimers` sections are a partition of the step, so coverage is a property of the
instrument.

Per-*path* attribution then comes from the kernel dump, because inductor fuses pointwise
work into opaque `triton_poi_fused_*` kernels but does **not** fuse away the four things
this tier must separate. `scripts/kprof_agg.py` gained two categories for exactly this
(`tests/test_kprof_agg.py` asserts they are purely additive — they only claim kernels that
previously fell through to `other`, so every table in `training_efficiency.md` still
reproduces):

| category | catches | answers |
|---|---|---|
| **`comm(nccl)`** | `ncclDevKernel_ReduceScatter*`, `*AllGather*`, `*AllReduce*` | gradient communication — O9's target, and Tier 3's |
| **`embedding/bank`** | `embedding_dense_backward`, `compute_grad_weight`, `krn_partial_segment_offset`, `sum_and_scatter`, `indexSelect*`, `index_add*` | **the bank gradient path** — what Tier 2 exists to remove |

The ordering of those two inside `CATS` is load-bearing and nearly went wrong: first match
wins, and inductor names fused kernels after the ops it fused, so
`triton_poi_fused_embedding_dense_backward_3` is a real kernel name. Placed ahead of
`compiled(triton)` the embedding pattern steals it and silently moves an existing
category's number. They sit *after* it, and a test pins that.

**Cross-check, because one instrument is not evidence.** The banks are 2.576B of 3.163B
parameters = **81.4%**, and both AdamW and reduce-scatter cost scale with element count.
So the measured `embedding/bank` + bank share of `comm(nccl)`/`optimizer` should land near
that analytic share. If the two readings disagree, neither is trusted until reconciled.

**Reading the results.** `scripts/run_summary.py <slurm log>` reduces a finished run to
the row this ledger needs — the `MFU accounting:` denominator, per-GPU MFU, global
tokens/s, step time, peak GiB, and the `timers_ms` section split averaged over a
steady-state window (rank 0 only, from step 15, which clears both warmup and the two
profiled steps). It prints the section coverage percentage directly, so the gate's
"≥90% of step time" is read off rather than argued. It **exits non-zero on any failure
signature** — OOM, traceback, `srun: error`, a cancellation — because a log that crashed
and a log that finished look identical otherwise. Tested in `tests/test_run_summary.py`.

**Still unavailable:** sub-block attribution *within* the compiled block (gather vs QK-norm
vs the mix vs the kernel). Getting it needs either a compile-off run — which measures a
different machine, compile being worth ~11 MFU points — or inductor provenance tracking.
Neither is needed for the Tier 2/3 decision, so neither is in scope. Recorded so the gap
is known rather than assumed covered.

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
| ~~B1/mb4~~ | 1 | `ame/h20_4gpu_mb4` | 4 | 32 | 4 | 32541 | — | — | — | **ignore** — cancelled 17:05, never ran |
| ~~B1/mb8~~ | 1 | `ame/h20_4gpu_mb8` | 4 | 16 | 8 | 32542 | — | — | — | **ignore** — cancelled 17:05, never ran |
| ~~B1~~ | 1 | `ame/h20_4gpu_mb16` | 4 | 8 | 16 | 32543 | — | — | — | **ignore** — cancelled 19:44, pre-fix snapshot, never ran |
| ~~B2~~ | 1 | `mfu_audit/h20_hbal_4gpu` | 4 | 8 | 16 | 32544 | — | — | — | **ignore** — cancelled 19:44, never ran |
| ~~O9~~ | 1 | `ame/h20_4gpu_mb16_o9` | 4 | 8 | 16 | 32555 | — | — | — | **ignore** — cancelled 19:44, never ran |
| ~~O31~~ | 1 | `ame/h20_4gpu_mb16_attrib` | 4 | 8 | 16 | 32559 | — | — | — | **ignore** — cancelled 19:44, never ran |
| **B1** ✅ | 1 | `ame/h20_4gpu_mb16` | 4 | 8 | 16 | **32576** | **281,748** | **26.84%** | **60.6** | 3.720 s/step, wandb `89nupukg` |
| **B2** ✅ | 1 | `mfu_audit/h20_hbal_4gpu` | 4 | 8 | 16 | **32577** | **443,432** | **44.44%** | **43.4** | 2.364 s/step, dense ceiling, wandb `oh5lepzx` |
| **O9** ✅ | 1 | `ame/h20_4gpu_mb16_o9` | 4 | 8 | 16 | **32578** | **274,627** | **26.16%** | **69.5** | 3.840 s/step, **slower than B1**, wandb `aoa5vzdj` |
| **O31** ✅ | 1 | `ame/h20_4gpu_mb16_attrib` | 4 | 8 | 16 | **32579** | 274,179 | 26.11% | 60.7 | attribution run; profiler costs ~2.7%, **not a baseline**, wandb `fztfn0lu` |
| **O31-dense** ✅ | 1 | `mfu_audit/h20_hbal_4gpu_attrib` | 4 | 8 | 16 | **32601** | — | — | 43.4 | dense attribution twin of 32579; 2360.3 ms/step |
| ~~B1-lem~~ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | 6236712 | — | — | — | **ignore** — died in compile, Triton cache race, 0 steps |
| ~~O9-lem~~ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | 6236713 | — | — | — | **ignore** — same, 0 steps |
| ~~B1-lem.2~~ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | 6236888 | — | — | — | **ignore** — triton fix held; died on the DTensor diagnostics bug |
| ~~O9-lem.2~~ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | 6236892 | — | — | — | **ignore** — same |
| **B1-lem** ✅ | 1 | `ame/h20_4gpu_mb16_lem` | 4 | 8 | 16 | **6237108** | **256,123** | n/a | **60.65** | **lem**, 4.092 s/step, wandb `jyvfozb7` |
| **O9-lem** ✅ | 1 | `ame/h20_4gpu_mb16_o9_lem` | 4 | 8 | 16 | **6237109** | **264,659** | n/a | **69.50** | **lem**, 3.960 s/step, wandb `jroouzw9` |

**Two clusters are in this table and their rows must not be compared across.** lem is
`gpu:hopper:4` — four GPUs per *node* — against entropy's 8×H100 SXM in one node. O9 is a
communication optimisation, so what it is worth is a property of the interconnect.
Compare **B1-lem with O9-lem**, and **B1 with O9**; never a lem row with an entropy row.
Each cluster yields its own self-consistent A/B. B2 (the dense ceiling) and O31 (the
attribution) exist only on entropy and are not duplicated.

All live runs sit at **ga 8 / micro-batch 16**, the shape `h20_mgpu.yaml` measured
the dense model at on 4 GPUs. 32543 vs 32555 is the O9 A/B; 32544 is the
same-architecture ceiling; 32559 carries `profile_timers` + `kernel_profile` and is an
attribution run, so **its throughput is not a baseline** — the profiler costs time. 32541 and 32542 were the deferred micro-batch sweep points,
cancelled before running — their configs remain on disk, so the sweep is one command away
if it is wanted later.

B1 and B2 were submitted from branch `ame_optimize` at `f302ae12` with the working tree
carrying only configs and documents — **deliberately before the O9 code change**, so the
baseline measures unmodified code. 32555 carries the O9 implementation. wandb project
`nano-ame`; run names match the config names in §T1.1.

**Queue note.** None of the four started on submission. The shared `j321m` account had
six 4-GPU jobs named `brrrrr` submitted at 15:55–15:56 from another session, filling
h100a (8/8 GPUs) and stacking the queue hourly, so the four baselines are scheduled
22:00 / 23:00 / 00:00 / 01:00. No preemption occurred: our QOS preempts only
`lukaskoz_h100`, which by then held a single GPU — not the four a baseline needs.

Reference points to carry into it: AME 1-GPU = 47,391 tok/s / 18.1% / 22.1 s (job 32360);
dense GPT-2-arch 4-GPU = 425,054 tok/s / 44.28% / 2.47 s.

### First result — O9 measured on lem, 4×H100, ga 8 / micro-batch 16

Both runs completed 30 steps on identical data from identical init.

| | step time | tokens/s (global) | tokens/s per GPU | peak GiB/GPU |
|---|---:|---:|---:|---:|
| **B1** (baseline) | 4.092 s | 256,123 | 64,031 | **60.65** |
| **O9** (sync suppressed) | **3.960 s** | **264,659** | **66,165** | **69.50** |
| **delta** | **−3.2%** | **+3.33%** | **+3.33%** | **+8.85** |

Steady-state means over steps 15–29.

**The memory prediction was right to within 1%.** §T1.2 predicted +9.5 GB/GPU from
`P × 4 bytes × (1 − 1/world)` = 3.163e9 × 4 × 0.75 = 9.49 GB = **8.84 GiB**. Measured:
**8.85 GiB**. The mechanism is understood, not merely observed.

**The throughput win is real but small, and that is the more interesting number.** At
ga 8 there are only 8 reduce-scatters per optimizer step and O9 removes 7 of them, yet
the step gets just 3.3% faster. **Gradient communication is therefore not a dominant term
at this shape** — which is a direct, if partial, answer to the question T1.3 exists to
settle, and it lowers the prior on Tiers 2 and 3. Note the plan's §T1.2 traffic estimate
assumed ga 32 (micro-batch 4); at ga 8 the saving was always going to be 4× smaller.

**Correctness gate: PASSED.**

| check | result |
|---|---|
| Step-0 grad norm, B1 vs O9 | 4.00139 vs 4.00137 — **4×10⁻⁶ relative** |
| Divisor error (a constant factor) | **none** — the ratio is 0.999995, not 4 or ¼ |
| Losses bit-identical | **19 of 30 steps** (0–18) |
| max \|Δloss\|/loss over 30 steps | 1.29×10⁻² |
| mean \|Δloss\|/loss | 2.45×10⁻³ |

Step 0 is the decisive one. §1.4.3 says a silent divisor error "shows up here as a
constant factor and nowhere else"; there is no constant factor, and agreement to 4×10⁻⁶
at the first optimizer step is fp32 reassociation and nothing else. Thereafter the
trajectories separate as chaos amplifies through `gradient_clipping=1.0`, which is
nonlinear and active for the first ~16 steps.

**One thing to keep watching:** the loss difference is **one-sided** — O9 is never above
the baseline, equal in 19 steps and below in 11. Pure rounding should be symmetric. The
benign reading is that the logged loss is quantised to bf16 (ulp 0.03125 at ≈7, and every
difference is exactly 1–3 ulps) over a monotonically decreasing series, so a vanishingly
small lead reads as "lower" rather than "sometimes higher". That is consistent with the
evidence, but it is an explanation, not a measurement. **If a longer O9 run ever shows a
persistent one-sided loss gap, revisit this.**

**MFU is not reported for these runs.** lem's GPUs identify as plain `NVIDIA H100`, which
matches no entry in `src/core/mfu.py:_PEAK_BF16` (`H100 80GB HBM3`, `H100 PCIe`,
`H100 NVL`), so `peak_flops_bf16` correctly returned `None` and the trainer logged
tokens/s without MFU — the module's documented behaviour, and the right one: a guessed
peak is worse than no MFU. Entropy's H100s are recognised, so the canonical MFU numbers
come from there. **Do not convert these tokens/s to MFU by assuming 989 TFLOP/s.**

### Tier 1 result — the gate says STOP, and O9 is a negative result on entropy

All four entropy runs completed 2026-11-04 20:05–20:25 on h100a, 4×H100 SXM
(`NVIDIA H100 80GB HBM3`, 989 TFLOP/s/GPU), ga 8 / micro-batch 16, 30 steps.

| run | MFU/GPU | tokens/s (global) | per GPU | step | peak GiB |
|---|---:|---:|---:|---:|---:|
| **B1** AME baseline | **26.84%** | 281,748 | 70,437 | 3.720 s | 60.6 |
| **B2** dense H-balanced | **44.44%** | 443,432 | 110,858 | 2.364 s | 43.4 |
| **O9** sync suppressed | **26.16%** | 274,627 | 68,657 | 3.840 s | 69.5 |

**AME reaches 60.4% of the dense ceiling** — 26.84% against 44.44%, a gap of 17.6 MFU
points. B2 also validates the instrument: 44.44% here against the 44.28% recorded for the
GPT-2-arch h20 at 4 GPUs, two different architectures landing within 0.2 points, each
against its own FLOPs/token denominator (3.964 vs 4.122 G).

#### O9 measures *worse* on entropy, and the opposite of lem

| | entropy (8×H100 SXM, one node) | lem (4 GPUs/node) |
|---|---:|---:|
| B1 tokens/s | 281,748 | 256,123 |
| O9 tokens/s | 274,627 | 264,659 |
| **O9 delta** | **−2.53%** | **+3.33%** |

**Same code, same config, opposite sign.** This is not noise — it is the mechanism, and
it is exactly why §3 forbids comparing rows across clusters.

O9 does not make communication free; it *trades NCCL traffic for HBM traffic*. With sync
suppressed FSDP2 accumulates into an unsharded fp32 buffer on every micro-step
(`unsharded_accumulated_grad += unsharded_param.grad`), which is a read-modify-write of
12.65 GB per micro-step — ~101 GB of extra HBM traffic per optimizer step at ga 8 —
in exchange for removing 7 of 8 reduce-scatters. On entropy's intra-node NVLink the
reduce-scatter was already cheap and well overlapped, so the trade loses. On lem's
topology it wins.

**Verdict: O9 is rejected on the plan's test bed.** Per the Goal, "a tier that measures
worse is a completed tier with a negative result, not a failure — record it and stop, do
not tune past the gate." The flag stays in the tree, default `false`, documented and
tested, because it is a real win on a slower interconnect — it is simply not one here.

#### O31 attribution — where the 3.84 s step actually goes

Section timers (`profile_timers`, rank 0, steps 15–29), which tile the step by
construction:

| section | ms | share |
|---|---:|---:|
| forward | 1515.4 | 39.4% |
| backward | 2308.6 | 60.1% |
| log (host, overlaps) | 217.0 | 5.6% |
| **optim** | **12.3** | **0.3%** |
| clip | 4.1 | 0.1% |
| data_fetch + data_h2d | 0.5 | 0.0% |
| **step_total** | **3842.6** | coverage **105.6%** |

Kernel attribution (`kernel_profile`, steps 12–13, via `scripts/kprof_agg.py`), with the
profiler's op-wrapper rows (`nccl:_*`, `section:*`, `FSDP::*`) excluded as double-counting:

| what | ms/step | note |
|---|---:|---|
| `triton_tem_fused_0` + `_zeros_1` | **1390.7** | flex-attention templates — 160 launches = 20 layers × 8 micro-steps |
| gemm (`nvjet_*`) | **1267.2** | `v_proj`, `o_proj`, FF, head |
| NCCL all-gather | **554.7** | **parameter** traffic — unaffected by O9 |
| NCCL reduce-scatter | **381.7** | **gradient** traffic — what O9 removes 7/8 of |
| optimizer (`multi_tensor_apply`) | ~164 | see the discrepancy below |
| **`embedding/bank`** (incl. fused) | **~50** | **the bank gradient path** |

#### The gate: the bank gradient/optimizer path is NOT a top-two term

Tier 2 (O1–O5) exists to stop FSDP2 and AdamW touching the 2.576B bank parameters
densely. Measured, that target is:

- **bank gradient path ≈ 50 ms/step ≈ 1.3%** of the step
- **optimizer ≈ 12–164 ms ≈ 0.3–4.3%** (instruments disagree, see below)
- **combined: at most ~5% of step time**

against attention templates at ~36%, GEMM at ~33% and NCCL at ~24%. The top two terms are
**attention compute** and **dense GEMM**; communication is third. The banks are **not**
in the top two — they are roughly 25× smaller than the smallest of them.

**So, per the Goal's instruction, Tier 2 does not proceed and Tier 3 (which is gated on
it) does not either.** This is the gate working as designed: it was put there precisely
to stop us building a 2.576B-parameter hand-managed optimizer path for a 1.3% term.

**Why pkmf's 1.43× does not transfer.** pkmf measured that speedup for the hand-managed
bank optimizer on a setup where the optimizer state was *not* sharded. Under FSDP2 each
rank already holds only 790M of the 3.163B parameters, and AdamW is bandwidth-bound, so
the work Tier 2 would remove has already been removed by sharding.

**One honest discrepancy, recorded rather than resolved.** `LoopTimers` puts `optim` at
12.3 ms (and `Optimizer.step#AdamW.step` agrees at 11.4 ms), while the `multi_tensor_apply`
kernels sum to ~164 ms. Those kernels are shared by AdamW, the foreach gradient clip and
FSDP2's own copies, so the 164 ms is an upper bound on the optimizer and the 12.3 ms a
lower one. **The conclusion is insensitive to which is right** — 4.3% is not a top-two
term either — but the gap should be closed before anyone quotes a single optimizer number.

**Known limit of this attribution:** ~52% of kernel time sits in `triton_tem_fused_*`
templates, which are compiled and do not resolve to a submodule. They are almost certainly
flex-attention forward and backward (launch count matches layers × micro-steps exactly),
but that is an inference from naming and arity, not a measurement. Closing it needs
inductor provenance tracking — see §T1.3's "still unavailable".

### Where AME's MFU actually goes — matched kernel comparison against dense

**Jobs 32579 (AME) and 32601 (dense H-balanced), the same run in every respect but
the attention type:** same node, 4×H100 SXM, FSDP2 + `MixedPrecisionPolicy(bf16, fp32)`,
`compile_blocks_full`, global batch 512×2048, ga 8 / micro-batch 16, profiler over steps
12–13.

The kernel tables already in [training_efficiency.md](training_efficiency.md) §4.2 could
**not** serve as the dense side: they are one GPU, plain module + autocast, **no FSDP2**
("nothing in this table measures the FSDP2 path"), and §8 states every number there is
single-GPU. Subtracting them would have charged AME for the whole distributed path.

| | AME | dense | delta |
|---|---:|---:|---:|
| **ATTENTION** | **1390.7** (flex) | **506.9** (flash) | **+883.8** · **2.74×** |
| NCCL all-gather (params) | 554.7 | 185.4 | +369.3 |
| NCCL reduce-scatter (grads) | 381.7 | 124.3 | +257.3 |
| bank gather + gradient | 147.5 | 0 | +147.5 |
| other triton (fused glue) | 472.7 | 332.9 | +139.8 |
| FSDP2 param packing | 163.4 | 32.4 | +131.0 |
| optimizer (AdamW/foreach) | 163.6 | 36.5 | +127.1 |
| other eager | 193.9 | 79.2 | +114.7 |
| softmax / CE | 57.4 | 57.7 | −0.3 |
| **GEMM** (proj + FF + head) | **1264.4** | **1298.4** | **−34.0 · 0.97×** |
| **step wall** | **3842.6** | **2360.3** | **+1482.3 · 1.63×** |

Section timers agree: forward 1515.4 vs 741.9, backward 2308.6 vs 1610.3, optim 12.3 vs
5.0, clip 4.1 vs 0.8.

#### The headroom, ranked

| rank | item | excess | share of AME step |
|---|---|---:|---:|
| **1** | **attention: flex vs flash** | **+883.8 ms** | **23.0%** |
| **2** | **communication (NCCL)** | **+626.7 ms** | **16.3%** |
| 3 | bank gather + gradient | +147.5 ms | 3.8% |
| 4 | FSDP2 param packing | +131.0 ms | 3.4% |
| 5 | optimizer | +127.1 ms | 3.3% |
| — | GEMM | −34.0 ms | −0.9% (AME is *cheaper*) |

**GEMM is exonerated.** 0.97× — AME's architecture costs nothing extra in matmuls. The
Q/K projections it removes are worth about as much as the shapes it changes. Every
remaining point of the 17.6-MFU gap is attention, communication, or bank bookkeeping.

#### Lever 1 — AME pays 2.74× for attention to keep a capability it is not using

AME uses `flex_attention` rather than flash, and `src/core/ame.py` says exactly why:

> ALiBi slopes. Fixed geometric by default; learnable is an ablation, **and it is the
> reason the flex backend exists** — flash's `alibi_slopes` [takes a non-differentiable
> tensor]

But `learnable_slopes=False` is the **default**, no config overrides it, and every number
in this document was measured with the slopes as a **frozen buffer**. The differentiability
that flex is paid for is not being used. Flash's `alibi_slopes` argument computes the same
`−m_h·(i−j)` bias, and the dense twin shows what that costs: **506.9 ms against 1390.7**.

**Not a drop-in, and the obstacle is specific:** `sink=True` is also default. The learned
sink is kv column 0 with *zero* distance bias and unconditional visibility
(`score_bias`), whereas flash applies its ALiBi slope uniformly across keys. Replacing
flex with flash therefore needs the sink handled — a separate term, a flash build with
attention-sink support, or dropping the sink and measuring the quality cost. That is a
real engineering question, but it is bounded, and the prize is **23% of the step**.

#### Lever 2 — the banks are 81% of parameters and FSDP2 all-gathers all of them

NCCL is 3.02× the dense model's. The mechanism: FSDP2 all-gathers each sharded block's
parameters once per micro-step, and AME's blocks carry two `[50304 × 1280]` banks — 257 MB
each in bf16 — of which a micro-batch reads a few thousand distinct rows. That is
**41 GB of all-gather traffic per step** for the banks alone.

Note **all-gather (554.7) exceeds reduce-scatter (381.7)**: the larger half is *parameter*
traffic, which neither O9 nor Tier 2 touches. The lever is keeping the banks sharded and
gathering rows on demand — the mechanism of pkmf's O6/O7, i.e. Tier 3's machinery, but
aimed at the parameter all-gather rather than at the gradient reduce it was scoped for.

**A prediction I recorded beforehand and got wrong, which is informative.** I predicted
dense NCCL would be AME's divided by the 5.10× parameter ratio = 183.6 ms. Measured:
**309.7 ms**, a ratio of 3.02× not 5.10×. Collectives have a **latency floor** — both runs
issue the same 352 all-gathers and 176 reduce-scatters, only the payload differs, so the
dense model's smaller messages are latency-bound rather than bandwidth-bound. Consequence:
shrinking bank traffic will recover **less** than proportionally, and ~310 ms of the 936 ms
is a floor that no amount of sharding removes.

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
- **2026-11-04** — **O9's "bit-identical" gate was wrong and is replaced.** Suppressing
  sync reassociates the sum (`sum_i shard_r(sum_ranks g_i)` becomes
  `shard_r(sum_ranks sum_i g_i)`), and fp32 addition is not associative. Bit-identity
  holds only at world_size 1. New gate: bit-identical at 1 rank, fp32 tolerance above it.
  Chasing the original gate at 4 GPUs would have been chasing a non-bug.
- **2026-11-04** — **O9 costs ~+9.5 GB/GPU** and the plan did not say so. FSDP2 keeps the
  *unsharded* gradient in fp32 between micro-steps when sync is off. Recorded in §T1.2;
  it is why the flag ships opt-in rather than default-on, and why `mb16` is the micro-batch
  at risk.
- **2026-11-04** — Negative control on the O9 tests found that a never-reducing loop
  leaves **every** `param.grad` as `None`, not merely wrong. `clip_gradient` filters on
  `p.grad is not None`, so such a run would clip nothing and step on nothing while still
  logging a loss. The `finally` restore and an explicit gradient-existence assertion both
  exist for this.
- **2026-11-04** — **lem and helios were evaluated as an escape from entropy's queue and
  both were rejected on evidence**, so the test bed stays entropy:

  | cluster | predicted 4-GPU start | what actually happened |
  |---|---|---|
  | **entropy** | 2026-11-04 21:50 | still pending at 18:08 |
  | helios | 2026-11-07 18:41 | not submitted |
  | **lem** | 2026-11-07 22:58 | **started within 60 s of submission** |

  > ### `sbatch --test-only` on lem was wrong by three days. Do not trust it alone.
  >
  > Both lem jobs (6236712, 6236713) were submitted at 18:06 and were **both RUNNING by
  > 18:08**, on nodes r12-2 and r12-13 — against a `--test-only` prediction of
  > **2026-11-07T22:58** made seven minutes earlier. Memory already records that
  > `squeue --start` is pessimistic because it assumes every running job uses its full
  > time limit; the same is true of `--test-only`, and on a partition with 93 pending
  > jobs and 3-day limits the error compounds to days.
  >
  > **The prediction is a lower bound on confidence, not an estimate of reality.** Where
  > a submission is cheap and cancellable, submit and observe rather than predict — a
  > 1-hour job that may start now is worth more than a perfect forecast that it will not.
  > The entropy-vs-lem decision was made on this prediction and was wrong.

  **lem's first probe was a false alarm and is corrected here.** At 16:35 every Slurm
  command returned `Unable to contact slurm controller` across four attempts over four
  minutes, and github.com would not resolve. Twenty minutes later `scontrol ping` reported
  *both* `Slurmctld(primary) at slurm-1` and `(backup) at slurm-2` UP, munge encoded and
  decoded cleanly, DNS resolved, and GitHub authenticated. It was a transient controller
  outage, most likely a failover, and it cleared on its own. **Do not conclude lem is
  down from a single round of failures** — re-probe twenty minutes later before acting.

  Two durable facts from the probe, both of which contradict what memory records:

  - The login node serves `ClusterName=bem2` and carries lem's partitions
    (`plgrid-lem-gpu-h100`, `plgrid-lem-cpu`, …) *within* that cluster. `sinfo -M all`
    works when a bare `sinfo -p plgrid-lem-gpu-h100` is flaky.
  - lem's `$HOME` quota problem is **stale**: 178 G free, 83% used. And
    `configs/_cluster/lem.yaml` on this branch already carries the cemetery-redirect and
    per-user `HF_HUB_CACHE` fixes that memory lists as uncommitted.

  What actually rules lem and helios out is **queue position, not health** — and the
  first reading of this was wrong in its mechanism. At 16:45 every one of lem's 73 H100
  nodes reported `AllocTRES=…gres/gpu=4`, which read as pure saturation. A re-check at
  17:59 found node `r12-13` fully **IDLE** with zero GPUs allocated, and still a
  2026-11-07 start prediction. The reason is **93 pending jobs against 34 running** on
  that partition: idle nodes sit in `PLANNED` while backfill holds them for
  higher-priority work ahead of us. The conclusion is unchanged, but the cause is a queue
  we are behind rather than a cluster with nothing free — which matters, because waiting
  does not help and neither would a smaller job shape.

  Note also that lem's nodes are `gpu:hopper:4`, so a 4-GPU job needs a *whole* node.
  Partially-free nodes are no use to this plan's shape.

  Worth stating plainly: moving this plan to another cluster would **not** have been a
  like-for-like substitution. O9 and Tier 3 are communication optimisations, and their
  measured value is a property of the interconnect. helios is GH200, a different topology
  from entropy's 8×H100 SXM node, so its numbers would not be comparable to B1 — the
  §1.3 invariant holds the workload fixed, not the hardware.
- **2026-11-04** — O9's FSDP2 import made defensive. `FSDPModule` is public under
  `torch.distributed.fsdp` only from torch 2.7 (entropy: 2.7.1+cu128); helios ships 2.6.0.
  A bare module-level import would have broken *every* run on that cluster, not just O9
  runs. Falls back to `torch.distributed._composable.fsdp`, then to `None`, in which case
  O9 logs a warning and degrades to a no-op.
- **2026-11-04** — **The B1 micro-batch sweep is deferred; mb16/ga8 is the working
  shape.** Decision by Maciej: do not vary the micro-batch yet, measure at the same
  micro-batch the dense model was measured at on 4 GPUs with FSDP2 — `h20_mgpu.yaml`'s
  **ga 8 / micro-batch 16**. So the live Tier 1 comparison narrows to three runs at one
  shape: **32543** (AME baseline), **32555** (AME + O9), **32544** (dense H-balanced
  ceiling). Jobs **32541** (mb4) and **32542** (mb8) were submitted before this decision
  and are the deferred points; they are held pending a decision on whether to cancel them,
  which would also move 32543 up the queue from 00:00 to 22:00.

  Note the tension this creates with §0's stated consequence: mb16 is exactly the
  micro-batch where O9's +9.5 GB/GPU is most likely to OOM. If 32555 OOMs while 32543
  does not, that is the finding — it bounds where O9 is usable — not a failed run.
- **2026-11-04** — **O31's CUDA-event-annotation design was abandoned on evidence.**
  `torch._dynamo.explain` shows `record_function` inside a compiled block is *dropped*,
  not broken (`ops` unchanged, `Profiler function ... will be ignored`), so the original
  plan would have produced an empty attribution table that looked like a profiler bug.
  Replaced by `profile_timers` (step-tiling sections, outside the compiled region) plus
  `kernel_profile` with two new `kprof_agg.py` categories, `comm(nccl)` and
  `embedding/bank`. Sub-block attribution *inside* the compiled block is now a known gap,
  not an assumed capability. Config: `configs/ame/h20_4gpu_mb16_attrib.yaml`.
- **2026-11-04** — **Both first lem runs died during compilation, before step 1, on a
  Triton cache race. Fixed in `main.py`.** Triton defaults `TRITON_CACHE_DIR` to
  `$HOME/.triton/cache` and publishes entries with `os.replace(temp, final)` — atomic on
  a local filesystem, **not on NFS**. Four ranks write identical cache keys at the same
  instant and tear each other's entries apart. Three symptoms, one cause, none of them
  naming it:

  | error | what it really was |
  |---|---|
  | `FileNotFoundError [Errno 2] … __triton_launcher.so` | another rank's rename removed the temp dir |
  | `OSError [Errno 26] Text file busy` | replacing a `.so` a rank already had mapped |
  | `ImportError … cannot open shared object file: Permission denied` | opening a half-written `.so` |

  `$HOME` is shared across *jobs* too, so 6236712 and 6236713 raced each other from
  separate nodes. Both produced **zero training steps**.

  Fix: `main.py:isolate_triton_cache()`, called from `setup_enviroment()` once `RANK` is
  known and before anything compiles. Points `TRITON_CACHE_DIR` at
  `$TMPDIR/triton-cache-<jobid>-rank<N>` — node-local *and* per-rank *and* per-job — and
  honours an explicitly set value. Guarded by `tests/test_triton_cache_isolation.py`.

  **This is latent on entropy too**, which also has an NFS `$HOME`. The earlier 4-GPU
  dense run there survived it, so the race is probabilistic rather than certain — which
  is the worst kind. The entropy jobs queued at 21:50 carry the pre-fix snapshot.
- **2026-11-04** — `scripts/run_summary.py` earned itself on its first real use: both lem
  logs end in a traceback with no metrics, and a reader skimming for numbers would have
  found none and assumed the run was still going. The tool exited 1 and named the
  signatures.
- **2026-11-04** — **AME has never run multi-GPU, and the first attempt found a real bug:
  a diagnostic killed the training run.** With the Triton race fixed, jobs 6236888/6236892
  trained correctly and then died at the first heavy-metric step:

  ```
  trainer.py:_log_layer_instrumentation -> ame.py:instrumentation
    -> ame.py:attention_diagnostics -> ame.py:_mixed_qkv
      q = self.q_bank(token_ids)
  RuntimeError: aten.embedding.default: got mixed torch.Tensor and DTensor
  ```

  `attention_diagnostics` re-runs part of the forward **outside FSDP2's forward hooks**,
  where the parameters are still sharded DTensors. `ame.py` already has `_full()` for
  exactly this — and it is applied to `tau`, `slopes` and the shift coefficients but not
  to the banks, because the diagnostics reuse `_mixed_qkv`, which gathers directly. The
  same hazard applies to every parameter that function touches: `q_bank`, `k_bank`,
  `v_proj`, `alpha`, `beta`, `tau`.

  **Fixed for now at the right altitude, not the convenient one:** a diagnostic must never
  kill a training run. `_log_layer_instrumentation` now catches per module, warns once
  naming the module and the cause, and **skips that module for the rest of the run** so
  the cost is not re-paid every heavy step. Five tests in
  `tests/test_instrumentation_is_not_fatal.py`; one of them caught that the first version
  of this fix warned "disabled for the rest of this run" while still calling the readout
  every step — the message was a lie.

  **Deliberately NOT fixed yet:** the underlying DTensor bug in `attention_diagnostics`.
  Fixing it means touching `_mixed_qkv`, which is the hot path this tier is *measuring*,
  and the entropy B1 baseline is already snapshotted against the current version. Changing
  it now would make the LEM and entropy runs incomparable. **Deferred to after Tier 1's
  numbers are in**, then fixed by routing the diagnostics through `_full()`-materialised
  parameters. Until then AME's per-layer diagnostics are unavailable on multi-GPU and the
  run says so in its log.
- **2026-11-04** — The four entropy jobs (32543/44/55/59) were **cancelled before running
  and resubmitted** as 32576/77/78/79. They carried the pre-fix snapshot, and the DTensor
  diagnostics bug is *deterministic* for AME on multi-GPU — it fires at the first
  heavy-metric step — so three of the four were certain to fail. (32544, the dense
  baseline, was safe: `RoPEAttention` defines no `instrumentation`, so the readout loop
  finds nothing to call. It was resubmitted anyway to pick up the Triton fix.) Queue
  position was not lost: the `brrrrr` block had drained, and B1's estimate actually
  improved from 21:50 to 20:32.
- **2026-11-04** — `run_summary.py` corrected: it treated `RuntimeError` inside the
  *handled* instrumentation WARNING as a failure, and reported two runs that trained 30
  steps each as failed. It now ignores WARNING-level lines and reports them separately.
  Both directions are tested — a handled warning must not fail, a real crash alongside one
  must still fail.
- **2026-11-04** — Earlier claim in [ame_optimization.md](ame_optimization.md) §8 that
  "none of the h20 baselines is at 4 GPUs" is **wrong**: `mfu_audit/h20_mgpu.yaml` carries
  measured 1/2/4-GPU numbers. The real gap is narrower — those numbers are the GPT-2-arch
  variant (4.122 GF/token), not the H-balanced h20 (3.97 GF/token) that AME's config
  builds on, and AME itself is 3.77 GF/token. Hence B2.
