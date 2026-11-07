# AME bank gates — report

**Status: implemented and submitted (2026-11-07). Four runs, one per gate, train on LEM;
nothing is scored yet.**

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

### Checks
