# Stage 2c — the best AME variant at h20: report

**Status: in progress (started 2026-11-08 00:40). Moved from Helios to LEM with Stage 1c
at 18:16, at Maciej's instruction; no Helios job had started
([stage_1c_report.md](stage_1c_report.md), decision 6). On LEM the AME smoke run (job
6274942) was submitted at 18:23. S2c-PS at n = 9 follows once it passes. The other two
runs wait for Stage 1c's best n.**

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
| Code and configs | Commit **`a8553b7e`** on branch `stage1c2c_lem`, the same as Stage 1c: `683c316b`'s code plus the config commit that moves both stages to LEM (`ame_main`'s `acb68ff5`, cherry-picked), submitted from `~/code/nano-stage1c2c` ([stage_1c_report.md](stage_1c_report.md), decision 7). Since Stage 2b's `89975bbf`, AME gained the bank-gates option (`157c3122`), off by default: with `bank_gate: none` the layer computes what it did in Stage 2b (only the q/k reshape moved after the gather) and draws nothing more from the RNG. The trainer gained readouts only (`train/clip_factor`, `train/grad_norm/banks`) and opt-in token counts, which are off; the clip itself is unchanged |
| Config | [configs/stage_2c/ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), `learning_rate: null`, set per run as `trainer.learning_rate=<n>`. One submit is one run |
| Cluster, recipe | Stage 1c's: LEM since 18:16 on 2026-11-08, one node of 4 H100 per run with r16-11 excluded, FSDP2 with `reshard_after_forward: false`, bf16 from FSDP2's mixed precision, micro-batch 16 × 8 per GPU, 15,259 steps = 16.00B tokens, warmup 152, seed 123, 24 h limit. The banks are column-sharded over the 4 GPUs (`mem_parallel_size: 4`) |
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
| 25862201 | Helios 4-GPU smoke: the S2c-PS file, 60 steps | 9 | — | — | 01:14 | never | 18:16 | cancelled, never started | — |
| 25862255 | Helios **S2c-PS** | 9 | — | — | 01:15 | never | 18:16 | cancelled, never started | — |
| 6274942 | LEM 4-GPU smoke: the S2c-PS file, 60 steps | 9 | | | 18:23 | | | pending | |

## Smoke run

**On Helios: cancelled before it started.** Job 25862201 was submitted at 01:14 with the
real runs, was still pending at 18:16, and was cancelled with the move to LEM.

**On LEM: job 6274942**, `smoke_s2c_ps_h20_lr9`, the S2c-PS file run from `a8553b7e`. It
uses `trainer.learning_rate=9 trainer.n_steps=60 trainer.eval_interval=20` and a 1.5 h
limit, with r16-11 excluded, and was submitted at 18:23, before the real run
([stage_1c_report.md](stage_1c_report.md), decision 8). It checks:
- that micro-batch 16 fits under FSDP2 with the banks sharded over 4 H100s (93.6 GiB each;
  64.63 GiB peak on entropy, §1.15);
- the s/step against the plan's 3.14–3.25, which was estimated for LEM;
- that `final_ce` is logged and the checkpoint written (backbone shards and each GPU's bank
  shard);
- that the readouts fail with a warning per layer while the run goes on.

Results: pending.

## Decisions

1. **The n = 9 run was submitted at 01:15, alongside Stage 1c's grid**, as the plan
   allows. n = 9 is in Stage 2c's grid whichever of 8, 9 or 10 Stage 1c picks, and it was
   n\* at h12 and h16, so it is the like-for-like point across the three scales. The other
   two runs wait for Stage 1c's best n, or for its leader to be clear mid-run.
2. **Moved to LEM with Stage 1c at 18:16** ([stage_1c_report.md](stage_1c_report.md),
   decisions 6–8). Helios jobs 25862201 and 25862255 never started and were cancelled. On
   LEM the smoke run goes first, then n = 9, which still runs alongside Stage 1c's grid
   for the reason in decision 1. Both stages stay on one cluster, as the plan requires.

## For Maciej

Nothing yet.
