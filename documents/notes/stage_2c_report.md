# Stage 2c — the best AME variant at h20: report

**Status: in progress (started 2026-11-08 00:40). Nothing of Stage 2c is submitted yet.**

The plan is [ame_experiments.md, "Stage 2c — the best AME variant at h20"](ame_experiments.md#stage-2c--the-best-ame-variant-at-h20).
This file is its record: every job, result and decision, each with its reason. Times are
CET. Stage 1c, which supplies the dense h20 baseline and n_d\*, is
[stage_1c_report.md](stage_1c_report.md). That report also holds what the two stages share:
the Helios compile fix, the pre-flight and the submission method. Stage 2b's record, the
same arm at h16, is [stage_2b_report.md](stage_2b_report.md).

## Result

Pending.

## Setup

| | |
|---|---|
| Question | How far is AME's best variant, ALiBi with the KV shift, from dense at h20 and 16B tokens? And does the gap keep shrinking with scale? |
| Arm | S2c-PS, one arm only (Maciej, 2026-11-07): S2b-PS on Stage 1c's model, recipe and hardware. `shift: true`, so α and β are learned per head and layer; the slopes are unset, so AME uses its ALiBi ladder, horizons 16 to 8,192 tokens over 20 heads; `bank_gate: none` |
| Code and configs | Commit **`683c316b`** on `ame_main`, the same as Stage 1c, submitted from `~/code/nano-stage1c2c` (branch `stage1c2c_helios`). Since Stage 2b's `89975bbf`, AME gained the bank-gates option (`157c3122`), off by default: with `bank_gate: none` the layer computes what it did in Stage 2b (only the q/k reshape moved after the gather) and draws nothing more from the RNG. The trainer gained readouts only (`train/clip_factor`, `train/grad_norm/banks`) and opt-in token counts, which are off; the clip itself is unchanged |
| Config | [configs/stage_2c/ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), `learning_rate: null`, set per run as `trainer.learning_rate=<n>`. One submit is one run |
| Cluster, recipe | Stage 1c's: Helios, one node of 4 GH200 per run, FSDP2 with `reshard_after_forward: false`, bf16 from FSDP2's mixed precision, micro-batch 16 × 8 per GPU, 15,259 steps = 16.00B tokens, warmup 152, seed 123, 24 h limit. The banks are column-sharded over the 4 GPUs (`mem_parallel_size: 4`) |
| Learning rate | n = 9 alongside Stage 1c; then n_d\* − 1, n_d\* and n_d\* + 1 around Stage 1c's best n, extended past an edge if needed |
| W&B | `ideas_cv/nano-ame`, runs `s2c_ps_h20_lr<n>.0`, tag `stage_2c` |

**Model**, built on the meta device ([stage_1c_report.md](stage_1c_report.md), pre-flight):
3,163,200,720 parameters, of which 2,575,564,800 are banks, 458,857,680 backbone and
128,778,240 embedding and head; 3.7683 GFLOPs per token at seq 2048. The banks are 5.1×
the dense model's whole parameter count; the FLOPs per token are 0.95× dense's.

**What the comparison with dense matches and what it does not:**

| Matched | Not matched |
|---|---|
| tokens, data order (4 GPUs) | total parameters: AME has 5.1× |
| recipe, precision path, hardware | FLOPs per token: AME does 0.95× |
| | positional signal: ALiBi with the shift, against RoPE |

**Weight decay differs between the arms.** AME's Q/K banks are exempt from weight decay,
while the dense W_Q and W_K are not
([archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) 2.6). Every AME-vs-dense
number in this file carries that difference.

**Readouts.** FSDP2 is on, so α₂, β₂, τ, the horizons and the attention entropy are not
logged: each AME layer's `instrumentation()` fails once with a warning and is dropped
([src/core/trainer.py](../../src/core/trainer.py), `_log_layer_instrumentation`). No fix had
reached `ame_main` by `683c316b`. So α₂, β₂ and τ are read from each final checkpoint, and
attention entropy and top-8 mass are left out, as the plan says.

## Jobs

| Job | What | n | W&B run | Node | Submitted | Started | Ended | State | `final_ce` |
|---|---|---:|---|---|---|---|---|---|---:|

## Decisions

None yet.

## For Maciej

Nothing yet.
