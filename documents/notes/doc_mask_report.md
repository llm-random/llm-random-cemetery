# Document masking for packed rows — implementation and h12 runs (plan and log)

**Status (2026-11-10 09:00): implemented and tested on CPU; GPU test job and the two
LEM runs about to be submitted.**

Maciej set the task by `/goal` on 2026-11-10: implement document masking on a branch
from `ame_main` in its own worktree, test it on CPU, then run dense h12 and AME
(KV shift + positional encoding, S2-PS) for 4B tokens at n = 8 on LEM, to measure the
mask's effect on both methods. Times are CET.

## Why

The packer (`src/core/datasets.py`, `sample_packer`) fills each 2049-token row with
documents back to back, each ending in `<|endoftext|>`, and cuts the row at the
length. Until now nothing masked across them: dense attention was SDPA with
`is_causal=True` and no mask, and AME's flex mask was causal plus the sink, the same
for every row. So every token also attended to the earlier documents in its row, and
AME's KV shift mixed each document's first key and value with the previous
document's EOS. That was a deliberate, documented choice: both methods saw the same
cross-document context, so the comparison was fair
([ame_implementation.md](ame_implementation.md), "Repo-specific findings"). These
runs measure what the choice costs or gains, for each method.

## The feature

`common.attention.document_mask: true` (default false). Branch `doc_mask`, worktree
`~/code/nano-doc-mask`, cut from `ame_main` 3ef6e363.

| Piece | What it does | Where |
|---|---|---|
| Documents | A position's document = number of EOS tokens strictly before it. An EOS belongs to the document it ends | `src/core/document_mask.py` `document_ids` |
| Mask | Query i sees key j iff j ≤ i and same document. Every query sees itself, so no row is empty | `allowed_keys` (materialized), `document_score_mod` (flex) |
| Dense (mha/mqa/gqa) | Mask off: the old SDPA call, unchanged. Mask on: the layer asks for token ids (`wants_token_ids`, as AME does); flex with a cached causal block mask and a score_mod that sends cross-document scores to −inf on CUDA, SDPA with the `[B, 1, n, n]` mask on CPU | `src/core/model.py` `RoPEAttention` |
| AME | The same score_mod wraps the ALiBi one in the flex kernel; the reference path masks the materialized scores. The sink stays visible to every query, and the LSE merge is unchanged. The KV shift's predecessor term is zero at every document start, as at position 0 of a row | `src/core/ame.py` |
| AME diagnostics | `score_matrix` and `attention_diagnostics` mask too; normalized entropy divides by the keys a row can see in its own document | `src/core/ame.py` |
| Positions | Not reset per document, and not needed: RoPE's score depends on q_pos − k_pos only, ALiBi's bias on i − j only | — |
| Loss | Unchanged: every position trains, including the EOS that predicts the next document's first token | — |
| Config | `common.eos_token_id: 0` (GPT-NeoX `<|endoftext|>`) in the h_balanced model config, passed to `build_attention`. `validate` refuses the mask without an EOS id before anything is submitted; a planned type refuses the mask at build | `configs/_model/h_balanced/base.yaml`, `src/definitions.py`, `src/core/attention.py` |

**Why a score_mod and not a mask_mod.** A mask_mod would need a block mask built per
batch from the document ids: either traced inside the compiled block, where
`create_block_mask` breaks the graph, or built outside and passed in, where the
BlockMask becomes a graph input (AME's `_block_mask` docstring records both failures).
The score_mod keeps the cached causal block mask, so the only new input is the
document ids, computed in the graph from the token ids it already has. The cost:
cross-document blocks are computed and masked rather than skipped.

## Verification

**CPU** (`tests/test_document_mask.py`, 36 tests, plus the full CPU suite):

- Every document of a packed row gets exactly the output it gets alone. Checked per
  layer type (mha, mqa, gqa, gqa NoPE, AME S2-PS, AME without sink), and through a
  whole h4 model (dense, AME, and a dense/AME mix): each document's logits do not
  depend on what precedes it in the row. Two rows with different boundaries,
  including a document that is only its EOS. Unmasked, the same check fails, so it
  has teeth.
- The AME shift stops at a boundary: pinned fully shifted, every document's first key
  and value are zero.
- Off is the old layer bit for bit: same parameters, same outputs, for every type.
- Flex (uncompiled on CPU) matches the reference path, dense and AME with and
  without the sink.
- AME's diagnostic scores are −inf exactly outside a row's own document; the sink
  column stays finite.
- Config: the switch reaches every layer; default off; no EOS id fails `validate`;
  the EOS id is the tokenizer's (`tokenize_fn` appends exactly one).
- The composed run configs equal the baselines' logged W&B configs except the mask
  keys, schema defaults (all equal to the Trainer's), names, tags and LEM's newer
  `mem_per_gpu` (228G against 220G).

**GPU** (`pytest -m gpu`, LEM): the compiled flex kernel against the CPU reference,
forward and backward, eager and inside `torch.compile`, at head width 64 with
boundaries on and across flex's 128-wide blocks; bf16 autocast within bf16
resolution; AME's existing GPU tests. Result: pending.

## Runs

| Run | Config | Baseline (same seed, data order, steps, schedule) | Baseline `final_ce` | Baseline s/step |
|---|---|---|---:|---:|
| `dm_dense_h12_lr8` | [dense_h12_docmask.yaml](../../configs/doc_mask/dense_h12_docmask.yaml) | S1-D n = 8, [`x0xlqroj`](https://wandb.ai/ideas_cv/nano-ame/runs/x0xlqroj), job 6252398_0 | 2.9824 | 3.272 |
| `dm_ame_h12_ps_lr8` | [ame_h12_ps_docmask.yaml](../../configs/doc_mask/ame_h12_ps_docmask.yaml) | S2-PS n = 8, [`doi36p8k`](https://wandb.ai/ideas_cv/nano-ame/runs/doi36p8k), job 6256577 | 3.1250 | 3.746 |

- **Recipe**, both: LEM, 1 H100, r16-11 excluded; h12; 3,815 steps of 512 × 2049
  tokens (4B); LR 2⁻⁸; seed 123; micro-batch 32; no FSDP2; autocast bf16;
  `compile_blocks_full`; final checkpoint only; 7 h limit. AME: Tier 2 banks
  (`mem_parallel_size: 1`), bank init N(0, 1), banks at the backbone's LR, as the
  baseline. W&B project `ideas_cv/nano-ame`.
- **The only difference from each baseline is `document_mask: true`.**

## How the verdict is made

As in [ame_improve_runs_report.md](ame_improve_runs_report.md): each pair shares the
seed, data order and every setting but the mask, so the train losses pair step by
step. Two measures:

- `final_ce` difference. Each run's value is a mean of 16 bf16 batch losses, so it can
  be off by up to 0.008, a difference by up to 0.016
  ([reference: final_ce precision](ame_improve_runs_report.md#results)).
- The paired train-loss gap (masked − baseline at the same step), averaged over the
  last 500 steps and over each part of training. This is the finer measure.

AME's same-seed spread at h12 4B is sd 0.0072, so a difference under 0.022 is a tie.
Dense runs spread 6–24× more, so a dense difference is less certain than an AME one.

Both runs are evaluated as trained: the masked model with the mask, on the same
packed held-out rows.

## Check procedure

- **Start (about 2 min after training begins):** job running, first steps logged, the
  W&B config has `document_mask: true`, s/step within 1.5× the baseline's, first
  losses level with the baseline's at the same steps.
- **Then one check per decision point:** mid-run (paired gap so far, s/step, NaN or
  spikes: a train loss more than 0.5 above the median of the previous 50 steps), and
  the end.
- Cancelled or failed runs are recorded below with their W&B ids as "ignore".

## Log

| Time | Event |
|---|---|
| 08:45 | LEM busy: 21 nodes in maintenance, 12 allocated; the other session's two AME jobs pending on Priority since 08:23; `sbatch --test-only` forecasts a 1-GPU start on 11-14 (forecasts have been days off before) |
