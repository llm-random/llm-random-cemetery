# AME bank gates — report

**Status: implemented and submitted (2026-11-07). Four h12 runs, one per gate, train on
LEM. An h16 wave of four more at 8B tokens was added the same evening (decision 7). The
evaluator is validated; nothing is scored yet.**

The plan is [ame_bank_gates_plan.md](ame_bank_gates_plan.md). This file is its record:
every job, check, result and decision, each with its reason. Times are CET.

## Setup

| | |
|---|---|
| Question | Which way of trusting a bank row gives the lowest final loss at h12 and 4B tokens, and which makes rarely updated rows least influential without costing loss? |
| Design | Four runs, one per mechanism, all with banks drawn at σ = 0.01 and otherwise identical: B1 per-row confidence, B2 context gate, B3 shared backoff vector, ST STEM gate. No control run (Maciej, 2026-11-07). The production reference is S2-PS, W&B `doi36p8k`, common evaluator 3.125227 |
| Code | Commit **`9d7ed99d`** on `ame_main` for every run: the mechanism in `157c3122`, the evaluator in `9d7ed99d`. Worktree `~/code/nano-bank-gates`, branch `bank_gates_lem`, equal to `ame_main` at every submit |
| Configs | [configs/bank_gates/](../../configs/bank_gates/): `base.yaml` (S2-PS at n = 8, a copy of the Stage 2 file), `b1.yaml`, `b2.yaml`, `b3.yaml`, `stem.yaml`, and `smoke_local.yaml` for the CPU smoke runs |
| Cluster, recipe | LEM, one H100 per run, node r16-11 excluded, 7 h limit. h12 AME as S2-PS: 3,815 steps × 1,048,576 tokens = 4.00B, seed 123, n = 8, warmup 38, micro-batch 32 × 16, Tier 2 banks, autocast bf16, `compile_blocks_full` |
| Evaluation | The common evaluator ([parity_numerics.py](../scripts/parity_numerics.py) `score`, `ame_ref32`) on the 8,192-sequence held-out set; the bucket analysis ([bank_gates_eval.py](../scripts/bank_gates_eval.py)); both on LEM through [bank_gates_lem_eval.sh](../scripts/bank_gates_lem_eval.sh) |

## Decisions

1. **Every gate starts every shared weight from the same numbers as `none` at the same
   seed.** Nothing an arm adds draws from the global RNG:
   - B1 widens the bank after the usual draw;
   - B2's and ST's gates are built on the meta device and then filled;
   - B3's base vectors are zeros;
   - ST's gate, the one random init, is drawn from a private generator seeded from a
     hash of the global RNG state, which it does not advance.

   **Why:** otherwise each arm would also start from a different draw of every later
   weight (the next blocks' projections, the head), as if it had another seed. The noise
   floor (±0.019) comes from same-seed replicates, so the arms must share the init for it
   to apply. A test pins it: after each arm's constructor the RNG state equals `none`'s,
   and every shared tensor is equal.
2. **B3's base vectors are cast to the rows' dtype before the add.** Under Tier 2 the
   rows arrive bf16, and an fp32 base would have promoted B3's whole prep (add, norm,
   mix) to fp32 while the other arms run it in bf16. The cast keeps the prep precision
   equal across arms.
3. **B2 and ST log their gate's weights, not its values.** Their gates read x, and the
   trainer's readout hook sees only token ids, so the plan lists no in-training readout
   for them. Each logs `gate_weight/rms`, and B2 also its bias mean, min and max. These
   are parameter readouts, as cheap as B1's and B3's. The gates' values on real
   activations come from the bucket analysis.
4. **The evaluator is a trimmed port, plus a LEM submit script.** `parity_numerics.py`
   on `ame_main` carries `parity_audit`'s `score` and `dump_data` checks and the
   helpers they use, unchanged except that a model is composed from any run config
   (`--config`, `--override`), so it builds every `bank_gate`. It needs none of
   `parity_audit`'s changes to `main.py`, `ame.py` or `model.py`, since `ame_ref32`
   uses only standard overrides. `parity_score_runs.sh` and `parity_submit.sh` target
   entropy's h100a; their LEM counterpart is `bank_gates_lem_eval.sh`.
5. **The held-out tensor is the one the references were scored on.** Entropy's parity
   dump (`~/parity/data/parity_batches.pt`, sha256 `3265f589…`) was copied to LEM, so
   every arm is scored on the same bytes as `doi36p8k`'s 3.125227. The validation job
   also rebuilds the held-out set from LEM's own FineWeb copy and compares the two.
6. **`black --check` passes on every file this plan adds or changes, but not
   tree-wide.** Nine files fail it on `ame_main` already. All predate this plan, and
   eight of them it does not touch:
   - `scripts/timers_by_rank.py`, `scripts/mfu_window_log.py`;
   - `tests/test_typed_config.py`, `tests/test_mfu.py`, `tests/test_repo_hygiene.py`,
     `tests/test_core_framework.py`;
   - `documents/scripts/stage_results.py`, `documents/scripts/ame_comparison_h12.py`;
   - `src/definitions.py`, whose only change here is a 3-line hunk that is itself
     black-clean.

   Reformatting them would rewrite other sessions' files and invite merge conflicts,
   so they are left for their owners.
7. **An h16 wave, by Maciej's request (2026-11-07, about 22:30).** Four more runs, one per
   gate: "Each is AME h16 at n = 8 (LR 2^-8) on 8B tokens, on LEM. Run them alongside the
   h12 wave without disturbing it, and keep finishing the h12 scoring and report as
   planned."
   - The plan's [h16 extension](ame_bank_gates_plan.md#h16-extension-maciej-2026-11-07)
     specifies them.
   - The reference is S2b-PS h16 at n = 8 (`hjw7znbz`).
   - The h12 noise floor does not transfer to h16 and 8B. h16 gaps are reported beside it
     as a stated proxy, with that caveat ([The h16 wave](#the-h16-wave)).

## Implementation and local checks

Commits `157c3122` (mechanism, trainer, tests, configs) and `9d7ed99d` (evaluator), both
fast-forwarded into `ame_main`.

- **`bank_gate: none` is bit-identical to the previous layer** (`7ae928a7`'s `ame.py`,
  loaded side by side). The state dict, the RNG state after construction, the fp32 and
  bf16 forward and every gradient are equal, at σ = 1, at σ = 0.01 and with the shift
  and sink off.
- **Tests:** 381 passed, 18 skipped (`pytest -m "not gpu and not slow and not network"`).
  The new ones cover:
  - every gate against a literal transcription of its formula;
  - B1's trust at 0.05 after `main.py`'s build path, its scores equal to s_q·s_k times
    the ungated scores, its gradient only for ids in the batch, and the fused
    memory-parallel path at width 780;
  - B2 at sigmoid(3), its scores equal to g_i times the ungated scores, and its bias
    undecayed;
  - B3 equal to the ungated forward at init, its base trained, undecayed and outside
    the FLOP count;
  - ST at about half the ungated logit spread;
  - the id-only invariants for `none`, `confidence` and `backoff`, and their break for
    `context` and `stem`;
  - the diagnostics guard, both backends, finite bf16 gradients, and every gate's
    readout through the trainer;
  - token counts, and the clip's bank norm and factor;
  - one `gpu`-marked test that each gate trains under `compile_blocks_full`. It is not
    run here (no GPU); the full runs' first steps exercise that path.
- **Startup parameter counts** (configs composed as `run_exp.py` composes them, model
  built on the meta device) match the plan's table exactly:

  | Arm | Parameters | GFLOP/token (seq 2048) |
  |---|---:|---:|
  | none (`base.yaml`, = `doi36p8k`) | 1,103,599,056 | 1.0529 |
  | B1 | 1,118,086,608 | 1.0529 |
  | B2 | 1,103,709,792 | 1.0536 |
  | B3 | 1,103,617,488 | 1.0529 |
  | ST | 1,110,676,944 | 1.0954 (+4.0%) |

- **CPU smoke run of each arm** (`smoke_local.yaml`: h4 at vocab 50,304, Tier 2 banks,
  4 steps of 4 × 32 tokens, final checkpoint on). Each one ran to completion:
  - it wrote `final_ce`, the model checkpoint, the bank sidecar and `token_counts.pt`;
  - the four `token_counts.pt` are identical: 512 occurrences = 4 steps × 4 × 32;
  - at step 0, B1's trust read 0.0500, B2's gate bias 3.0 with weight RMS 0, and B3's
    base norms 0;
  - entropy and top-k were logged for B1 and B3 and, by design, not for B2 and ST.
- **The ported evaluator on the smoke checkpoints** reproduces each run's own
  `final_ce` within 1e-6, through `ame_ref32` (banks loaded from the sidecar's fp32
  masters into `nn.Embedding`) and through `ame_prod` (through the controller). On CPU
  both paths run fp32, so this checks the loading of every gate's parameters, not the
  evaluator's precision:

  | Arm | trainer `final_ce` | `ame_ref32` | `ame_prod` |
  |---|---:|---:|---:|
  | B1 | 10.290774 | 10.290774 | 10.290774 |
  | B2 | 10.323029 | 10.323028 | 10.323028 |
  | B3 | 10.313798 | 10.313798 | 10.313798 |
  | ST | 10.314072 | 10.314072 | 10.314072 |

  `bank_gates_eval.py` ran on the same four checkpoints and its total CE equals the
  evaluator's.

## The wave — submitted 2026-11-07 22:16-22:18

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
<arm> +infrastructure.slurm.exclude=r16-11`, from the worktree at `9d7ed99d`. Each was
running within two minutes. Every cemetery commit has `9d7ed99d` as its parent and
changes only `exp.job`, `generated_configs/config_0.yaml` and `run_exp.py`'s usual
untracking of `pixi.toml` and `pixi.lock`.

| Arm | `bank_gate` | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---|---:|
| B1 | confidence | 6264723_0 | `49456e2` | `bank_gates_b1_2026-11-07_22-16-06` | `621sjn71` | r12-6 | 1,118,086,608 |
| B2 | context | 6264725_0 | `fda924b` | `bank_gates_b2_2026-11-07_22-16-34` | `kgk0prvf` | r12-7 | 1,103,709,792 |
| B3 | backoff | 6264726_0 | `12d2779` | `bank_gates_b3_2026-11-07_22-16-59` | `5w6p64bv` | r12-10 | 1,103,617,488 |
| ST | stem | 6264733_0 | `6af74f7` | `bank_gates_stem_2026-11-07_22-17-22` | `l54pyyqp` | r14-9 | *pending* |

W&B project `nano-ame`; run names `bank_gates_<arm>`. Each run's final checkpoint lands
in its clone under `checkpoint/<job>/0/step_3814/`, with `token_counts.pt` beside it.

**Evaluator validation, job 6264732** (`bg_eval_doi36p8k`, r12-10), submitted 22:17
with `bank_gates_lem_eval.sh` from `9d7ed99d`, while the four runs train:
1. `parity_numerics.py score` on `doi36p8k`'s final checkpoint (job 6256577, clone
   `s2_ps_h12_lr8.0_2026-11-06_13-52-40`), `ame_ref32` and `ame_prod`. The plan's
   gate: `ame_ref32` = 3.125227.
2. `dump_data`: the held-out set rebuilt from LEM's FineWeb copy and compared with the
   entropy dump that every score uses.

**The evaluator is validated (22:22).** On `doi36p8k`'s checkpoint the port gives
`ame_ref32` = **3.1252272**, the plan's 3.125227 to every digit it states, in 242 s.
- The production path (`ame_prod`) gives 3.1253729 in 49 s. The parity audit had
  3.125377 for it on entropy; the 4e-6 gap is bf16 nondeterminism in that path.
- The common evaluator, fp32 and eager, is the one the plan scores with, and it agrees
  exactly.
- LEM's own held-out set, rebuilt by `dump_data` from LEM's FineWeb copy, is identical
  to the entropy dump (`torch.equal` on the `[8192, 2049]` tensor; fingerprint
  `[21440, 25411, …, 48684]`).

So a score from LEM's dataloader and a score from the copied tensor are the same
measurement. Job 6264732 completed in 5 min 44 s.

### Checks

**Start, 22:18-22:21.** All four were running within two minutes of submission and logged
their first steps by 22:21. FineWeb opened in about 3 minutes, against the 10-21 the plan
warned of. The logged parameter counts equal the plan's table. The banks run Tier 2 in
bf16 ("Memory-parallel banks enabled: mem_parallel_size=1, compute dtype torch.bfloat16").

| Arm | Step | loss | grad_norm | grad_norm/banks | clip factor | s/step |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 12 | 7.69 | 0.87 | 0.00004 | 1.00 | 3.87 |
| B2 | 13 | 7.66 | 1.51 | 0.014 | 0.66 | 3.83 |
| B3 | 12 | 7.59 | 1.41 | 0.022 | 0.71 | 3.77 |
| ST | 3 | 9.25 | 1.98 | 0.013 | 0.50 | 3.88 |

- **The banks are a small share of the joint norm, not the bulk of it.** The plan's
  worry was that σ = 0.01 would make the bank gradients dominate the clip (the stop
  rule). In these first steps the banks contribute 0.00004-0.022 of a norm of 0.9-2.0.
  The clipping during warmup is the backbone's, as in `doi36p8k`, whose `grad_norm`
  averaged 2.5-2.8 before step 38.
- **B1's banks barely move at first:** every score starts at 0.0025 of its ungated size
  (the trust is 0.05 on both sides), so the rows' gradient is scaled down by the trust
  as well.

**Step 300, 22:42.** Every check passes, and the stop rule does not apply.

Train loss and `grad_norm` are window means (min/mean/max for the loss), and `doi36p8k`'s
are at the same steps. s/step is the median, excluding the eval steps.

| Arm | Loss, steps 200-299 | `grad_norm`, 100-199 / 200-299 | `grad_norm/banks`, 200-299 | Clip factor after step 76 | s/step | vs `doi36p8k`'s 3.746 |
|---|---|---|---:|---|---:|---:|
| B1 | 4.91/5.16/5.53 | 0.243 / 0.212 | 0.0016 | 1.000 | 3.880 | +3.6% |
| B2 | 4.25/4.43/4.69 | 0.251 / 0.197 | 0.045 | 1.000 | 3.836 | +2.4% |
| B3 | 4.12/4.35/4.66 | 0.286 / 0.237 | 0.043 | 1.000 | 3.773 | +0.7% |
| ST | 4.38/4.58/4.81 | 0.262 / 0.193 | 0.059 | 1.000 | 3.892 | +3.9% |
| `doi36p8k` | 4.81/5.03/5.28 | 0.293 / 0.182 | not logged | — | 3.746 | |

- **No NaN or inf** in any run.
- **s/step** is within the plan's 10% (ST: 15%) in every arm.
- **`grad_norm`** tracks `doi36p8k`'s.
- **The banks never dominate the norm.** At most about a fifth of it, so 4% of the
  squared norm. B1's banks are a hundredth of that. The clip has been inactive since the
  warmup in all four runs, as in production. So the stop rule's condition (a clip factor
  far below 1 after the warmup) is not met.
- **The readouts move** (layer means; the range over layers in brackets):

  | Arm | Step 0 | Step 300 |
  |---|---|---|
  | B1 trust, q / k | 0.0500 / 0.0500 | 0.0509 / 0.0504; p90 0.0534 [0.0517, 0.0605] / 0.0509; none above 0.5 |
  | B2 gate | bias 3.000, weight RMS 0 | bias mean 3.055, min 2.92 [2.49, 3.05]; weight RMS 0.044 |
  | B3 base norm, q / k | 0 / 0 | 0.246 [0.131, 0.313] / 0.220 [0.166, 0.265] |
  | ST gate weight RMS | 0.0317 | 0.0556; τ falling, 8.00 → 7.69 |

  B1's trust climbs as slowly as the plan predicts: reaching s = 0.5 takes at least 754
  of a token's own updates at the peak learning rate. B3's base vectors are already
  about three times the rows' initial length per head (0.08 at σ = 0.01).
- **An early lead, not a result.** Three arms are well ahead of production at the same
  step: 4.35-4.58 against 5.03 over steps 200-299. That fits σ = 0.01 turning the banks
  about 100× faster at the start, since QK-norm makes σ an inverse learning rate for
  the banks ([ame_implementation.md](ame_implementation.md), weight init). B1, whose
  trust mutes that, trails production slightly (5.16). The plan does not read a
  step-300 loss as quality: production's 220-step losses spread with an sd of 0.13.

**Step 500, 22:55.** Every check passes again.

| Arm | Loss, steps 400-499 | `grad_norm`, 400-499 | `grad_norm/banks` | Clip factor | s/step |
|---|---|---:|---:|---|---:|
| B1 | 4.31/4.47/4.62 | 0.155 | 0.0027 | 1.000 | 3.880 |
| B2 | 3.86/3.95/4.06 | 0.124 | 0.030 | 1.000 | 3.833 |
| B3 | 3.78/3.88/3.98 | 0.131 | 0.025 | 1.000 | 3.773 |
| ST | 3.91/4.01/4.12 | 0.130 | 0.039 | 1.000 | 3.880 |
| `doi36p8k` | 4.19/4.33/4.50 | 0.145 | not logged | — | 3.746 |

- **No NaN.** s/step and `grad_norm` are steady. The clip has stayed inactive since the
  warmup.
- **Readouts at step 500:**
  - B1's trust: mean 0.0516 (q) and 0.0514 (k), q p90 0.0564. The first two
    (token, head) entries of one layer crossed 0.5.
  - B2's bias: mean 3.00, minimum 2.36; weight RMS 0.047.
  - B3's base norms: 0.30 (q) and 0.29 (k).
  - ST's gate RMS: 0.065; τ 7.65.
- **The early lead over production is shrinking:** 0.32-0.45 over steps 400-499,
  against 0.45-0.68 over steps 200-299.

## The h16 wave

Four runs, one per gate, at h16 on 8B tokens (decision 7; the plan's
[h16 extension](ame_bank_gates_plan.md#h16-extension-maciej-2026-11-07)). They train
alongside the h12 wave.

| | |
|---|---|
| Configs | `configs/bank_gates/h16_base.yaml`, a copy of [configs/stage_2b/ame_h16_pe_shift.yaml](../../configs/stage_2b/ame_h16_pe_shift.yaml) (S2b-PS h16), plus `h16_b1.yaml`, `h16_b2.yaml`, `h16_b3.yaml`, `h16_stem.yaml` |
| Recipe | S2b-PS h16 as `hjw7znbz` ran it: one H100, micro-batch 16 × 32, Tier 2 banks, no FSDP2, autocast bf16, `compile_blocks_full`, 24 h limit. Changed: n = 8, 7,630 steps = 8.00B tokens with warmup 76 written out, `count_token_updates`, W&B names `bank_gates_h16_<arm>` |
| Reference | S2b-PS h16 at n = 8: job 6260570, W&B `hjw7znbz`, clone `s2b_ps_h16_lr8.0_2026-11-07_03-36-02`. Trainer `final_ce` 2.8594; 1,986,332,928 parameters; 6.810 s/step; peak 72.17 GiB; 15 h 06 min |
| Noise floor | None measured at h16 and 8B. The h12 floor (±0.019 at h12 and 4B) does not transfer; h16 gaps are reported beside it as a stated proxy, with that caveat |

### Before submitting

Configs committed in `c96c31a4` and fast-forwarded into `ame_main`.

**Composed as `run_exp.py` composes them**, each passed `check_point` and was built on the
meta device. The parameter counts equal the plan's arithmetic:

| Arm | Parameters | GFLOP/token (seq 2048) |
|---|---:|---:|
| none (`h16_base.yaml`, = `hjw7znbz`) | 1,986,332,928 | 2.1212 |
| B1 | 2,012,088,576 | 2.1212 |
| B2 | 1,986,595,328 | 2.1228 |
| B3 | 1,986,365,696 | 2.1212 |
| ST | 2,003,110,144 | 2.2219 (+4.75%) |

**The copy is the reference's recipe.** `h16_base.yaml`, resolved, was compared key by
key with `hjw7znbz`'s own generated config from its cemetery clone, also resolved. They
differ only in:
- names and tags;
- `count_token_updates`;
- `run_exp.py`'s bookkeeping keys;
- the exclude, which both pass at submit;
- the RAM request: 228G, the LEM cluster file's value since 2026-11-07, against the
  reference's 220G. It is host memory and changes nothing in the run.

So the data config (dataset, seed 123, sequence length 2049, batch 512) is the arms'
own. With world size 1 on both sides (W&B config), an arm's `token_counts.pt` describes
the reference's data stream too.

**Time.** `hjw7znbz` took 15 h 06 min at 6.810 s/step. ST, at +4.75% FLOPs, projects to
about 15.8 h. The limit is 24 h.

**Memory.** `hjw7znbz` peaked at 72.17 GiB; the H100 has 93.6 GiB.
- B1's 16 extra columns over 32 banks add 25.8M parameters, about 0.5 GiB of Tier 2 state
  at 18 bytes per parameter.
- ST's gate adds 16.8M parameters (about 0.3 GiB with AdamW) and at most about 3 GiB of
  saved activations at micro-batch 16 (three `[16, 2048, 1024]` bf16 tensors per layer).
- The heaviest arm is therefore at most about 76 GiB.

8B tokens fits both, so there was no reason to stop and ask.

**A test fixed on the way (`c96c31a4`).** `test_every_runnable_config_composes_and_resolves`
failed once `base.yaml` was tracked (`157c3122`). It resolves every config, and
interpolating the mandatory `bank_gates_arm: ???` raises even with missing values allowed.
The suite reported in "Implementation and local checks" ran before the configs were
tracked, so it never saw them. `bank_gates_arm` now defaults to `base`. At `c96c31a4` the
suite passes with every config tracked: 381 passed, 18 skipped. The running h12 jobs are
unaffected, since their arm files set the key.

### Jobs — submitted 2026-11-07 22:39-22:40

All four were submitted with `run_exp.py --config-path configs/bank_gates --config-name
h16_<arm> +infrastructure.slurm.exclude=r16-11`, one at a time, from the worktree at
`c96c31a4` (= `ame_main`). Every cemetery commit has `c96c31a4` as its parent and changes
only `exp.job`, `generated_configs/config_0.yaml` and the two pixi files.

| Arm | `bank_gate` | Job | Cemetery commit | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Parameters, as logged |
|---|---|---|---|---|---|---|---:|
| B1 | confidence | 6264776_0 | `02e1be80` | `bank_gates_h16_b1_2026-11-07_22-39-03` | `h85djd57` | r16-10 | 2,012,088,576 |
| B2 | context | 6264777_0 | `f8434114` | `bank_gates_h16_b2_2026-11-07_22-39-24` | `jwirkswo` | r16-13 | 1,986,595,328 |
| B3 | backoff | 6264778_0 | `92509a63` | `bank_gates_h16_b3_2026-11-07_22-39-46` | `3yc7haqu` | r16-14 | 1,986,365,696 |
| ST | stem | 6264779_0 | `b6dd7a5f` | `bank_gates_h16_stem_2026-11-07_22-40-08` | `ug8g2i2x` | r16-15 | 2,003,110,144 |

Each final checkpoint lands under `checkpoint/<job>/0/step_7629/` in its clone, with
`token_counts.pt` beside it.

### Checks

**Start, 22:39-22:46.** All four were running within about two minutes of submission,
logged the plan's parameter counts, and trained from 22:44. FineWeb opened in about 3
minutes.

| Arm | Step | loss | grad_norm | grad_norm/banks | clip factor | s/step | vs `hjw7znbz`'s 6.810 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 34 | 7.06 | 0.75 | 0.00014 | 1.00 | 7.03 | +3.2% |
| B2 | 30 | 6.91 | 0.96 | 0.060 | 1.00 | 6.86 | +0.7% |
| B3 | 26 | 7.09 | 1.07 | 0.099 | 0.94 | 6.81 | 0.0% |
| ST | 4 | 8.75 | 2.12 | 0.018 | 0.47 | 7.01 | +3.0% |

As at h12, the banks are a small share of the norm, and any clipping in the warmup is the
backbone's.

**Step 300, 23:24.** All four passed the step-300 checks. B2 then spiked at steps 345-359,
the first loss spike in this plan.

| Arm | Loss, steps 200-299 | `grad_norm`, 100-199 / 200-299 | `grad_norm/banks`, 200-299 | Clip factor, steps 100-299 | s/step | vs 6.810 |
|---|---|---|---:|---|---:|---:|
| B1 | 4.81/5.05/5.34 | 0.325 / 0.204 | 0.0026 | 1.000 | 7.011 | +3.0% |
| B2 | 4.09/4.28/4.53 | 0.316 / 0.188 | 0.034 | 1.000 | 6.867 | +0.8% |
| B3 | 4.12/4.30/4.50 | 0.354 / 0.233 | 0.041 | 0.96-1.00 | 6.804 | −0.1% |
| ST | 4.12/4.34/4.56 | 0.325 / 0.207 | 0.055 | 1.000 | 7.023 | +3.1% |
| `hjw7znbz` | 4.84/5.10/5.41 | 0.282 / 0.237 | not logged | — | 6.810 | |

- **Up to step 300 the h16 runs look like the h12 runs.**
  - No NaN.
  - s/step within the plan's limits; B1's and ST's overhead matches h12's.
  - `grad_norm` tracks the reference.
  - The banks are a small share of the norm, and the clip has been inactive since about
    step 100.
  - The readouts move: B1's trust mean 0.0506 (q); B2's bias mean 3.08, minimum 2.64,
    weight RMS 0.045; B3's base norms 0.25 (q) and 0.18 (k); ST's gate RMS
    0.027 → 0.055, τ 7.74.
  - The same early lead as at h12: B2, B3 and ST at 4.28-4.34 over steps 200-299 against
    the reference's 5.10, with B1 level with it.
- **B2 spiked at steps 345-359.**
  - The train loss rose from 4.0 at step 335 to 4.28 at 345 and 5.09 at 347, peaked at
    7.97 at step 354, and was back to 6.69 by step 359.
  - `grad_norm` peaked at 5.49 at step 348, and the clip was active (factor 0.18-0.97)
    through the spike.
  - The banks were 0.14-0.58 of that norm. So the spike is mostly the backbone's, and the
    stop rule's condition (the banks dominating the norm after the warmup) does not hold.
  - It is not a crash or a NaN, so nothing is cancelled or resubmitted. Earlier AME runs
    recovered from spikes like it: Stage 2's S2-S at steps 555-581 (to 9.50) and S1-A0
    at 609-635 (to 14.69).
  - **Recovered by 23:32.** The clip was back at 1.0 by step 360, and `grad_norm` had
    settled to 0.15-0.3 by step 380. The 20-step mean loss was 4.59 at step 414: back
    under 4.6, though still above the pre-spike 4.0.

    | Step | 335 | 345 | 350 | 354 | 360 | 370 | 380 | 390 | 400 | 414 |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | Train loss | 4.03 | 4.28 | 5.72 | 7.97 | 6.41 | 5.38 | 5.06 | 4.75 | 4.69 | 4.50 |

    The final CE will show whether the detour cost B2 anything lasting at h16.

**Step 500, 23:48.** Every check passes, with a second transient spike, in B3.

| Arm | Loss, steps 400-499 (min/mean/max) | `grad_norm`, 400-499 | `grad_norm/banks` | Clip factor | s/step |
|---|---|---:|---:|---|---:|
| B1 | 4.25/4.38/4.56 | 0.152 | 0.0031 | 1.000 | 7.013 |
| B2 | 4.06/4.28/4.69 | 0.155 | 0.026 | 1.000 | 6.863 |
| B3 | 4.00/4.53/6.56 | 0.481 | 0.070 | 0.30-1.00 (spike) | 6.816 |
| ST | 3.72/3.82/3.97 | 0.126 | 0.032 | 1.000 | 7.023 |
| `hjw7znbz` | 4.19/4.34/4.53 | 0.150 | not logged | — | 6.810 |

- **No NaN.** s/step is steady.
- **B2's recovery held:** 4.28 over steps 400-499, level with the reference.
- **B3 spiked at steps 404-416 and recovered by step 430.**
  - The loss went from 4.28 at step 400 to a peak of 6.56 at step 408, was back to 4.66
    by step 430, and reached 3.92 at step 560, below its pre-spike 4.0-4.2.
  - `grad_norm` peaked at 3.35, and the clip was active (down to 0.30) through steps
    404-416.
  - The banks were a larger share of the norm than in B2's spike, at most 0.95 of 3.35
    (8% of the squared norm). The backbone still dominated, so again the stop rule does
    not apply.
  - At step 500, B3's layers 0 and 1 have nearly collapsed attention (normalized entropy
    0.24 and 0.13, top-8 mass 0.86 and 0.93), and the largest base vectors (norms
    0.45-0.61, against 0.20-0.39 in layers 2-15). The shared vector is strongest where
    attention is sharpest; the bucket analysis looks at it with real activations.
- **The readouts:**
  - B1's trust: mean 0.0510 (q) and 0.0508 (k); the first entries cross 0.5.
  - B2's bias: mean 3.14, minimum 2.76; weight RMS 0.065.
  - B3's base norms: 0.33 (q) and 0.28 (k), layer means.
  - ST's gate RMS: 0.064; τ 7.68.
