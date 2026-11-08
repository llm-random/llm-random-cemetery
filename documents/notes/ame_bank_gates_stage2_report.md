# AME bank gates, stage 2 — report

**Status (2026-11-08 14:45): the four runs are training on LEM,** submitted 14:32-14:35
and expected to finish about 18:50-19:00. Scoring follows.

The plan is [ame_bank_gates_stage2_plan.md](ame_bank_gates_stage2_plan.md). This file is
its record: every job, check, result and decision, each with its reason. Times are CET.
Stage 1: [plan](ame_bank_gates_plan.md), [report](ame_bank_gates_report.md).

## Setup

**Questions** (the plan's):
1. Does σ = 0.01 alone, with no gate, lower final CE against production?
2. Does B3 add anything beyond the init?
3. Can B1 help if its trust doesn't start at 0.05?

**The seven cells.** The four new runs are in bold; the other three exist. Final CE is the
common evaluator's (`ame_ref32`).

| σ | No gate | B1, trust starts at 0.05 | B1, trust starts at 0.95 | B3 |
|---|---|---|---|---|
| 1 (production's) | `doi36p8k`: 3.125227 | **`b1_std1`** | — | **`b3_std1`** |
| 0.01 (stage 1's) | **`none_std001`** | stage-1 B1 (`621sjn71`): 3.141958 | **`b1open_std001`** | stage-1 B3 (`5w6p64bv`): 3.035837 |

| | |
|---|---|
| Recipe | Stage 1's [base.yaml](../../configs/bank_gates/base.yaml), unchanged: h12, 3,815 steps = 4.0B tokens, n = 8 (LR 2⁻⁸), warmup 38, seed 123, one LEM H100, micro-batch 32 × 16, Tier 2 banks, no FSDP2, autocast bf16, `compile_blocks_full`, 7 h limit |
| Configs | `configs/bank_gates/stage2_<X>.yaml`, each `defaults: [base, _self_]` plus the plan's options; W&B names `bank_gates_stage2_<X>` |
| Evaluator | Stage 1's, unchanged: `parity_numerics.py score` (`ame_ref32`, then `ame_prod`) and `bank_gates_eval.py`, through `bank_gates_lem_eval.sh` |
| Noise floor | ±0.019, stage 1's: the 95% half-width of one run against another. A gap inside it is a tie |

## Decisions

1. **`b1open_std001` is in (2026-11-08).** It is the plan's default, and Maciej's go kept
   it. It needs the stage's one code change, `confidence_init`.
2. **`confidence_init` refuses what it cannot mean (2026-11-08).**
   - A value outside (0, 1) has no logit, so the layer raises.
   - A non-default value on any gate but `confidence` raises too. Ignoring it would let
     a run believe it had changed B1.
   - The default, 0.05, stays legal everywhere, so no stage-1 config changes.
3. **The stage-2 plan stays uncommitted (2026-11-08).** It is Maciej's file, untracked in
   `~/code/nano` (last edited 14:08). The rules let me commit only my own edits. I
   updated its status line in place, as for stage 1's plan, and left the commit to
   Maciej. Until then, the link above works in the working tree only.

## Implementation and local checks

Committed as `f313760b` on `bank_gates_lem`, rebased onto `ame_main` as **`03317133`** and
fast-forwarded into `ame_main`. Between stage 1's last code (`69e31dc1`) and `03317133`,
only this commit touches code; everything else is other sessions' docs.

**The change** ([ame.py](../../src/core/ame.py), [attention.py](../../src/core/attention.py)):
- `AMEAttention(confidence_init=0.05)`, passed to `_with_trust_columns`, which writes
  `logit(confidence_init)` into the H trust columns of both banks.
- The logit is the expression the old module constant was, so the default writes the
  same fp32 value as stage 1's B1: −2.9444389 (0.95 gives +2.9444389).
- `confidence_init` is added to `_AME_OPTIONS`, so `attention.ame` accepts it.

**Tests:** 393 passed, 18 skipped (`-m "not gpu and not slow"`). That is stage 1's 381
plus 12 new cases:
- the main-build-path test (through the memory-parallel copy) and the at-init score test,
  both parametrized over 0.05 and 0.95. At 0.95 every score starts at 0.9025 of its
  ungated size;
- the default and an explicit 0.05 build the same state dict, bit for bit, and the trust
  columns hold logit(0.05) exactly;
- the readout and the layer's repr report 0.95;
- the rejections in decision 2.

The GPU-marked tests cannot run here (this machine's torch has no CUDA, which fails every
gate's GPU test alike); the runs themselves exercise that path.

**CPU smoke of `b1open_std001`'s path** through `main.py` (`smoke_local.yaml`, h4):
- every layer's trust read 0.9500 at step 0 and had moved by step 2;
- the run ended normally, with its final checkpoint and `token_counts.pt`.

**Composed as `run_exp.py` composes them**, each config passed `check_point` and was built
on the meta device. Each differs from its stage-1 counterpart only in the plan's options
and the names:

| Run | Parameters | = stage 1's | Differs from | In |
|---|---:|---|---|---|
| `none_std001` | 1,103,599,056 | none (`base.yaml`) | `base.yaml` | `bank_init_std: 0.01`, an explicit `bank_gate: none` |
| `b1_std1` | 1,118,086,608 | B1 | `b1.yaml` | `bank_init_std: 1.0` |
| `b3_std1` | 1,103,617,488 | B3 | `b3.yaml` | `bank_init_std: 1.0` |
| `b1open_std001` | 1,118,086,608 | B1 | `b1.yaml` | `confidence_init: 0.95` |

All four run 1.0529 GFLOP/token, as `doi36p8k` does.

## The wave — submitted 2026-11-08 14:32-14:35

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
stage2_<X> +infrastructure.slurm.exclude=r16-11 < /dev/null`, one at a time, from the
worktree at `03317133` (= `ame_main`, clean). Every cemetery commit has `03317133` as its
parent and changes only `exp.job`, `generated_configs/config_0.yaml` and the two pixi
files.

| Run | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---:|
| `none_std001` | 6272065_0 | `c5459c14` | `bank_gates_stage2_none_std001_2026-11-08_14-32-40` | `kf4q4t4s` | r14-5 | 1,103,599,056 |
| `b1_std1` | 6272075_0 | `430f9846` | `bank_gates_stage2_b1_std1_2026-11-08_14-33-32` | `1dmp8l8w` | r14-6 | 1,118,086,608 |
| `b3_std1` | 6272076_0 | `b1806db4` | `bank_gates_stage2_b3_std1_2026-11-08_14-33-56` | `5hzd2fte` | r16-1 | 1,103,617,488 |
| `b1open_std001` | 6272078_0 | `ff4100c5` | `bank_gates_stage2_b1open_std001_2026-11-08_14-34-18` | `jc9flzsz` | r16-1 | *pending* |

Each final checkpoint lands under `checkpoint/<job>/0/step_3814/` in its clone, with
`token_counts.pt` beside it.

### Checks

The plan's checks, at about 2 minutes after each start and at step 500:
- no NaN;
- s/step within 10% of stage 1's: 3.746 with no gate, 3.771 for B3, 3.880 for B1;
- B1's trust and B3's base norms moving.

**Start, 14:39-14:42.** All four were running within a minute of submission, logged the
composed parameter counts, and trained from 14:40. Taken at steps 30-34 (step-0 readouts;
the windows are steps 10-38, against `doi36p8k`'s same steps):

| Run | Step | loss | grad_norm | grad_norm/banks, steps 10-38 | clip factor | s/step | vs stage 1 |
|---|---:|---:|---:|---|---:|---:|---:|
| `none_std001` | 34 | 6.63 | 0.65 | 0.011-0.24 | 1.00 | 3.760 | +0.4% (3.746) |
| `b1_std1` | 30 | 7.16 | 1.32 | 9e-7 to 8e-5 | 0.76 | 3.901 | +0.5% (3.880) |
| `b3_std1` | 32 | 7.03 | 0.54 | 9e-5 to 0.035 | 1.00 | 3.787 | +0.4% (3.771) |
| `b1open_std001` | 30 | 7.09 | 1.28 | 0.010-4.8 | 0.78 | 3.900 | +0.5% (3.880) |

- **No NaN; s/step within 0.5% of stage 1's** for the same gate.
- **Loss tracks production's early steps:** a mean of 7.09-7.52 over steps 10-38, against
  `doi36p8k`'s 7.34. The warmup's `grad_norm` peaks (up to 46-50) are the backbone's, as in
  `doi36p8k` (46.5 in the same steps).
- **Every mechanism starts at its init** in every layer: trust 0.05 (`b1_std1`) and 0.95
  (`b1open_std001`), base norms 0 (`b3_std1`). Step 500 reads whether they move.
- **`b1_std1`'s banks get almost no gradient, as the plan feared.**
  - Its `grad_norm/banks` is 9e-7 to 8e-5, four orders of magnitude below
    `none_std001`'s. Stage-1 B1 (σ = 0.01) started at 4e-5.
  - Two factors shrink it. A σ = 1 row is 8 long per head, so the norm divides its
    gradient by 8 rather than 0.08. The trusts (0.05 × 0.05) scale the scores, and
    so the gradient, down by another 400.
  - The banks' AdamW uses the default ε = 1e-8. Spread over the hundreds of millions of
    bank entries a batch touches, a norm of 1e-5 leaves a typical entry's gradient far
    below ε. Adam's step is then about g/ε of the learning rate rather than the learning
    rate itself, so the rows and trusts may hardly train. This is a reading to check, not
    a fault: nothing is cancelled.

**`b1open_std001` spiked at steps 381-391, 15:05.** The step-500 watch flagged it at step
390, against a low of 4.13.
- **The loss:** 4.10-4.13 at steps 350-370, creeping to 4.34 by step 380. It then rose
  from step 381, peaked at 7.16 at step 391, and was back to 5.63 by step 398.
- **The gradient:** `grad_norm` peaked at 4.96 at step 388, and the clip was active
  (factor 0.20-0.80) at steps 382-394.
- **The backbone's, not the banks':** the banks were 0.06-0.21 of the norm throughout
  (at most 4.5% of the squared norm).
- **The banks' share before the spike was `none_std001`'s:** a median of 0.25 of the
  norm over steps 340-380, against 0.26. At trust 0.95, B1 passes nearly the whole score
  gradient, so its banks train like the ungated σ = 0.01 run's.
  - Stage-1 B1, at trust 0.05, had 0.014.
  - Both σ = 1 runs had about 0.01 (`b3_std1` 0.009, `b1_std1` 0.006).
- **Like stage 1's h16 spikes:** B2's at steps 345-359 (to 7.97) and B3's at 404-416 (to
  6.56) were of the same kind, and both runs recovered and finished normally.
- **Not a crash or a NaN,** so nothing is cancelled or resubmitted. Step 500 shows whether
  it recovered.

| Step | 370 | 380 | 382 | 385 | 388 | 391 | 394 | 398 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train loss | 4.13 | 4.34 | 4.88 | 5.88 | 6.63 | 7.16 | 6.19 | 5.63 |
| `grad_norm` | 0.18 | 0.40 | 1.33 | 2.54 | 4.96 | 2.45 | 1.25 | 0.65 |

**Step 500, 15:15.** Every check passes. `b1open_std001` has recovered from its spike.

| Run | Loss, steps 400-499 (min/mean/max) | vs `doi36p8k` (4.33) | `grad_norm/banks` | Clip factor | s/step | At step 500 (layer mean [range]) |
|---|---|---:|---:|---|---:|---|
| `none_std001` | 3.80/3.89/4.00 | −0.44 | 0.027 | 1.000 | 3.759 | entropy 0.49; τ 8.08 |
| `b1_std1` | 4.31/4.44/4.59 | +0.11 | 0.0015 | 1.000 | 3.900 | trust mean 0.0501 (q), p90 0.0505 (q) and 0.0501 (k); entropy 0.69; τ 8.72 [8.15, 9.95] |
| `b3_std1` | 4.16/4.30/4.44 | −0.03 | 0.0014 | 1.000 | 3.788 | base norms 1.87 [1.54, 2.24] (q) and 1.28 [1.11, 1.64] (k); entropy 0.51; τ 8.49 |
| `b1open_std001` | 4.16/4.48/5.41 (the spike's tail) | +0.15 | 0.032 | 1.000 | 3.900 | trust mean 0.950, p90 0.9522 (q and k); entropy 0.44; τ 8.24 |
| stage-1 B3 (`5w6p64bv`) | 3.78/3.88/3.98 | −0.45 | 0.025 | | | base norms 0.30 (q) and 0.29 (k) |
| stage-1 B1 (`621sjn71`) | 4.31/4.47/4.62 | +0.14 | 0.0027 | | | trust mean 0.0516, p90 0.0564 (q) |

- **No NaN; s/step steady** within 0.5% of stage 1's.
- **`b1open_std001` recovered.**
  - The clip was back at 1.0 from step 395.
  - The loss was 4.16 by step 520, against 4.10-4.13 before the spike.
  - Its 400-499 mean still carries the spike's tail. Before the spike it was already
    behind `none_std001`: 4.44 against 4.33 over steps 200-299.
- **B3's bases move fast.** 0 → 1.87 (q) and 1.28 (k) by step 500, six times stage-1
  B3's. That is the length they need to count against rows 8 long: a quarter of the
  row, where stage-1 B3's 0.3 already outweighed its 0.08-long rare rows.
- **B1's trust moves, but only in the third decimal place, from either start.**
  - `b1_std1`'s p90 went 0.0500 → 0.0505, stage-1 B1's 0.0500 → 0.0564, and
    `b1open_std001`'s 0.9500 → 0.9522.
  - At p90 that is a logit drift of +0.010, +0.047 and +0.127. That is the order of
    lr × √steps ≈ 0.09, the drift of a gradient with no consistent sign.
  - Starting the trust open does not let it learn. It mostly stays where it is put.
- **`b1_std1`'s banks are no longer starved.** Its `grad_norm/banks` grew a
  thousandfold from the start, to 0.0015 over steps 400-499. That is level with
  `b3_std1`'s 0.0014, at the same σ, so Adam's ε matters for the first steps at most.
  Its trust is still the slowest of the three B1 runs.
- **Early ordering, not a result:** the init alone (`none_std001`, 3.89) is level with
  stage-1 B3 (3.88) at steps 400-499, 0.44 below production. Both σ = 1 runs stay near
  production: `b3_std1` −0.03, `b1_std1` +0.11. Stage 1's early leads shrank but held to
  the end, so only final CE answers the questions.
