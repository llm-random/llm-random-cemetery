# AME init follow-up: pkmf-scale `Linear`s and head under the old embedding (plan and log)

**Status:** RUNNING, 2026-11-10: both runs submitted to LEM at 10:03-10:04 (jobs
6295253 and 6295256) and pending. Maciej asked for them on 2026-11-10.
Background:
- [ame_init_embed_ablation.md](ame_init_embed_ablation.md), its *Analysis, step 1*;
- [ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md).

**Question:** take the best old-init h12 runs and change only the `Linear`s' and the
head's init to pkmf's √(0.1/fan_in). Does that help or hurt?
- AME is compared with `kf4q4t4s`: σ = 0.01 banks, no gate.
- Dense is compared with `x0xlqroj`.
- Both run at n = 8.

## Why

- **AME.** In ame_init_embed_ablation.md, the new init's head, `Linear`s and banks
  together came out neutral for AME (−0.0001, from `oldinit_embnowd` to `oldemb`). Yet
  σ = 0.01 banks alone gain −0.105 in paired train loss (bank-gates stage 2).
  - The inference there: the new head and `Linear`s cost about +0.10 and cancel the
    banks' gain.
  - No run isolated them. This one does, against `kf4q4t4s`, which already has the
    σ = 0.01 banks.
- **Dense.** The new head and `Linear`s cost +0.2325 there, but that run had a Xavier
  head and an undecayed embedding. This run prices pkmf-scale `Linear`s and head with
  the embedding as `x0xlqroj` had it.

## The runs

Maciej's spec (2026-11-10). One change from each reference: the `Linear`s' and the
head's init.

| Weight | Both runs | The references |
|---|---|---|
| Token embedding | N(0, 1) cut at ±2 (std 0.88), decayed at 0.1 | the same |
| AME banks (`q_bank`, `k_bank`) | N(0, 0.01²), not cut, not decayed | the same (`kf4q4t4s`) |
| Attention `Linear`s: AME's V and O (its Q and K are the banks), dense's Q, K, V and O | **√(0.1/fan_in), cut at ±2σ** | 1/√fan_in, cut at ±2σ |
| FF `Linear`s: in, gate and out | **√(0.1/fan_in), cut at ±2σ** | 1/√fan_in, cut at ±2σ |
| Head | **√(0.1/768) = 0.0114, cut at ±2σ** | 1/√768 = 0.0361, cut at ±2σ |
| Everything else | as the reference | |

- **`pkmflin_ame_h12`.** S2-PS (AME with the KV shift and ALiBi), no gate, h12.
  - Recipe: 3,815 steps = 4.0B tokens, n = 8 (LR 2⁻⁸), warmup 38, seed 123; one LEM
    H100, micro-batch 32 × 16, Tier 2 banks, no FSDP2, autocast bf16,
    `compile_blocks_full`; 7 h limit.
  - Reference: `kf4q4t4s` (`bank_gates_stage2_none_std001`), in-training final_ce
    3.0352, 3.758 s/step, 70.40 GiB.
- **`pkmflin_dense_h12`.** S1-D (GQA with RoPE): the same recipe without banks, with a
  5 h limit.
  - Reference: `x0xlqroj` (`s1_d_h12_lr8`), final_ce 2.9824, 3.272 s/step.
- **The cut is relative (±2σ), as asked.** A weight's actual std is therefore 0.88σ,
  as our `Linear`s always had. pkmf's own draw cuts at a fixed ±0.2. At these σ
  (0.0057-0.0114) that is 17-35σ out, so it never bites
  ([pkmf_init_comparison.md](pkmf_init_comparison.md)).

## How to analyze

- **Each run:**
  - final_ce and its Δ against the reference;
  - the paired same-seed train-loss gap per quarter and over the last 500 steps
    ([scripts/paired_gap.py](../../scripts/paired_gap.py), on the branch);
  - health: spikes, the grad-norm max after step 100, s/step and peak memory;
  - block 0's embedding norm against its attention update norm.
- **The paired gap decides.** A gap inside ±0.022 is a tie: that is AME's floor, and
  dense borrows it.
- **AME, against `kf4q4t4s`:** the gap is the price of pkmf-scale `Linear`s and head,
  given σ = 0.01 banks.
  - Near +0.10: confirms the inference above. The `Linear`s and head cancel the banks'
    gain.
  - Inside ±0.022: they are neutral here. The cancellation in
    ame_init_embed_ablation.md then came from something this run keeps at the old
    values: the Xavier head (0.0063, against 0.0114 here) or the Xavier banks (0.0063,
    against 0.01).
  - Below −0.022: they add a gain on top of the banks'.
- **Dense, against `x0xlqroj`:** the gap is the price of pkmf-scale `Linear`s and head
  with the embedding decayed. Set it against ame_init_embed_ablation.md's +0.2325,
  which had a Xavier head and an undecayed embedding.

## Log

### Implementation

2026-11-10, on branch `ame_initialization` (worktree `~/code/nano-ame-init`).

- **Code: `model.embedding.weight_decay`**
  ([src/core/model.py](../../src/core/model.py), `TransformerEmbedding`).
  - The branch exempts the embedding from decay by a class attribute, so no config
    could decay it again.
  - `weight_decay: true` sets an instance attribute that shadows the exemption.
  - The default is false, so no other config changes. Anything but a bool is refused.
- **Tests:** two new ones in
  [tests/test_init_defaults.py](../../tests/test_init_defaults.py).
  - With the knob on, the embedding is back in the decayed group, and a dense model is
    back to the one group holding exactly `model.parameters()`, as stage 1-2's dense
    runs had.
  - A non-bool value is refused.
  - The four test files that touch `parameter_groups` pass on CPU: 177 passed, 13 GPU
    tests deselected.
- **Configs.** Each writes out every init, so the branch's defaults don't reach it:
  - [pkmflin_ame_h12.yaml](../../configs/ame_init/pkmflin_ame_h12.yaml) is a copy of
    `configs/bank_gates/base.yaml` with `stage2_none_std001`'s two attention options;
  - [pkmflin_dense_h12.yaml](../../configs/ame_init/pkmflin_dense_h12.yaml) is a copy
    of `oldinit_embnowd_dense_h12.yaml`.
- **Against the reference's W&B-logged config** (composed as `run_exp.py` does, with
  `check_point`, then resolved), the only training keys that differ are the init's:
  - `model.embedding.weight_decay`: absent → true (the references were decayed by
    default);
  - `attention_fn.init_scale`, the scales of the FF's `ff_pre_act_fn`, `gate_fn` and
    `ff_post_act_fn`, and the head's scale: 1 → 0.3162.

  Everything else that differs is names, tags and runtime keys (job id, paths). For
  dense it also includes Slurm's `mem_per_gpu` (220G → 228G, LEM's default since) and
  the time limit (7 h → 5 h).

- **Built on CPU with the real init** (seed 123), each against its reference's
  config: the new runs on the branch, the references on `ame_main`.
  - `ame_main` hasn't changed `src/` or any config these runs read since `03317133`,
    `kf4q4t4s`'s commit.
  - Dense reference: `configs/stage_1/dense_h12_lr_grid.yaml` set to n = 8 (the init
    doesn't read n).
  - AME reference: `configs/bank_gates/stage2_none_std001.yaml`.

  | | `pkmflin_dense_h12` | `pkmflin_ame_h12` |
  |---|---|---|
  | Parameters | 111 (183,454,464), as the reference | 171 (1,103,599,056), as the reference |
  | Bit-identical to the reference | 26: the embedding and the 25 norm weights | 110: the embedding, the banks, τ, α, β, the sinks and the norms |
  | Exactly √0.1 × the reference's draw | 85: Q, K, V, O, FF in, gate and out in 12 blocks, and the head | 61: V, O, FF in, gate and out in 12 blocks, and the head |
  | Embedding std, max \|w\| | 0.8798, 2.000 | 0.8798, 2.000 |
  | Head, attention, FF in and gate (fan_in 768): std, max \|w\| | 0.0100, 0.0228 (reference 0.0317, 0.0722) | the same |
  | FF out (fan_in 3072): std | 0.00502 (reference 0.0159) | the same |
  | Banks: std | — | 0.0100, as the reference |
  | Weight decay 0.1 / 0 (numel) | 183,454,464 / 0: one group, as the reference | 176,376,576 / 927,222,480, as the reference, with the embedding decayed |

  The worst element is 1.9e-7 off the common ratio √0.1, which is float32 rounding.
  The cut is relative, so the new draws use the same uniform samples as the
  reference's. Every parameter drawn after them, the banks included, therefore
  starts exactly as in the reference.

### Runs

Submitted from `~/code/nano-ame-init` at `3c55ed99` (code, configs and this doc),
with each config's `run_exp.py` command plus `+infrastructure.slurm.exclude=r16-11`,
one after the other. Each clone is under
`/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`.

| Run | Job | Clone | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|---|
| `pkmflin_ame_h12` | 6295253 | `ame_init_pkmflin_ame_h12_2026-11-10_10-03-09` | | | 11-10 10:03 | | PENDING |
| `pkmflin_dense_h12` | 6295256 | `ame_init_pkmflin_dense_h12_2026-11-10_10-03-56` | | | 11-10 10:04 | | PENDING |

### Checks

- **Submit, 11-10 10:15:** both PENDING (Priority) behind our four other LEM jobs: the
  ame_init step-2 pair and the doc-mask pair. The worktree was clean after each
  `run_exp.py`.
- **LEM's Slurm controller went unreachable, 11-10 from about 11:09.**
  - Both jobs were PENDING at 11:07. Both watchers then exited with "left the
    queue": a watcher bug that read a failed `squeue` as every job gone.
  - Fixed: the watcher accepts only rc=0 (details in ame_init_embed_ablation.md,
    *Checks*).
  - **Back at 11:24, after about 15 min.** Both jobs survived, still PENDING
    (Priority). The watchers were re-armed with the fix at 11:25.

### Results

| Run | final_ce | Reference | Δ final_ce | Paired gap, last 500 | Verdict |
|---|---:|---|---:|---:|---|
| `pkmflin_ame_h12` | | `kf4q4t4s` 3.0352 | | | |
| `pkmflin_dense_h12` | | `x0xlqroj` 2.9824 | | | |

### Analysis

### Decisions and issues

1. **Names: `pkmflin`**, for pkmf-scale `Linear`s, with the head counted among them.
   W&B names `ame_init_pkmflin_<model>_h12`; tag `ame_init_pkmflin`.
2. **The banks' σ = 0.01 is `kf4q4t4s`'s code path:** `bank_init_std: 0.01` draws
   `normal_(0, 0.01)` in both trees. The branch changed only the default (null, which
   means Xavier).
