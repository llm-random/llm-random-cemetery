# Task: PC vs HP training-time overhead vs batch size (paper Fig. 2)

Started 2026-10-06. Session "pc_time".

## Goal
Empirically back Fig. 2 ("Amortized relative computational overhead of PC across increasing batch
sizes"): PC's extra work (materialize W_c, projection backward) is a fixed cost per optimizer step,
independent of batch size b, so its relative overhead vs HP should shrink as b grows.
Plan (Maciej): Llama 1B, HP vs PC, on 1 H100, batch 100k → 2M tokens (rounded to seq-len multiples),
grad_acc = 1, ~10 steps per point for timing. First find max batch that fits on 1 GPU for PC 50%,
then 70% and 90%.

## Code facts (verified)
- No activation checkpointing anywhere in src/ → all activations for the full batch must fit.
- PC overhead runs ONCE PER OPTIMIZER STEP, outside the grad-accum loop:
  `prepare_compressed_weights()` at start of each step (projected_compression/trainer.py:59),
  `pass_gradient_to_projections()` after accumulation (:72). So grad accumulation still measures
  amortization correctly (overhead per step / tokens per step).
- Step time is only logged as a 100-step average (`steps/100/time`) → for 10-step probes use wandb
  per-row `_timestamp` deltas.
- Vocab 128,256 → logits dominate activation memory: ~4 GB bf16 per 16k tokens just for logits,
  more with fp32 loss/grad copies. This caps tokens/GPU regardless of compression ratio.

## Probe setup
Configs `configs/pc_project_mas/llama_1B_pc_bs_probe{,_70p,_90p}_entropy.yaml`: plain PC (no distill),
1×H100 (`gpu:h100:1`), grad_acc 1, n_steps 10, warmup 2, no checkpoint, wandb `ideas_cv/pc_time`.
Grid over `common.batch_size` (sequences of 1024). NB run_exp resolves interpolations before grid
expansion → need a plain `batch_size` placeholder next to `^batch_size` (caret wins).

## Results
### PC 50% (job 23350) — DOES NOT FIT on 1 GPU at any batch
- batch 16 seqs (16k tokens) OOM in forward (calculate_loss): 73.6 GiB already allocated, tried +3.9 GiB (logits).
- Static memory ≈ 70 GB: ~3B trainable projection+W_b params (ff P ≈ 8192×5376 per matrix) in fp32 ×
  (weight+grad+2 Adam) ≈ 48 GB, + frozen 1B source + compressed model. On 4 GPUs FSDP2 shards this.
- Larger tasks cancelled (would OOM too).

### PC 90% (job 23357), PC 70% (job 23361) — 1×H100, grad_acc 1, 10 steps (wandb ideas_cv/pc_time)
Max batch that fits: **90% → 32 seqs (32,768 tok)**, OOM at 48. **70% → 24 seqs (24,576 tok)**, OOM at 32.

| ratio | seqs | tokens | s/step (median) | tok/s |
|---|---|---|---|---|
| 90% | 4 | 4,096 | 0.373 | 10,991 |
| 90% | 8 | 8,192 | 0.382 | 21,437 |
| 90% | 16 | 16,384 | 0.437 | 37,531 |
| 90% | 24 | 24,576 | 0.479 | 51,280 |
| 90% | 32 | 32,768 | 0.544 | 60,198 |
| 70% | 4 | 4,096 | 0.690 | 5,938 |
| 70% | 8 | 8,192 | 0.713 | 11,497 |
| 70% | 16 | 16,384 | 0.776 | 21,126 |
| 70% | 24 | 24,576 | 0.951 | 25,840 |

- s/step = median of wandb `_timestamp` deltas after dropping 2 warm-up steps (~9 deltas) — rough.
- Peak GPU mem from wandb system metrics is sampled ~15 s → under-sampled, not reliable (e.g. 70% bs24 shows 51 GiB < bs16 65 GiB).
- Linear fit (rough): 90% ≈ 0.35 s fixed/step + 6.1 ms/seq; 70% ≈ 0.64 s fixed/step + 13 ms/seq. The fixed
  part grows with projection size (70% has larger P than 90%) — consistent with a per-step PC overhead,
  but it also contains optimizer step / launch costs; needs an HP baseline to attribute.
- Illustrative only (extrapolation): with grad accumulation at 32-seq microbatches, 2M tok ≈ 61 microbatches
  × ~0.195 s ≈ 12 s compute + ~0.35 s fixed → fixed ≈ 3% at 90%.

## Conclusions
- grad_acc=1 on 1 GPU caps batch at ~24–32k tokens (70/90%) and PC 50% does not fit at all → the planned
  100k–2M sweep is impossible as specified. Logits (128k vocab) + no activation checkpointing are the limit.
- Options: (1) grad accumulation with a fixed microbatch (valid: PC overhead is once per optimizer step);
  (2) multi-GPU FSDP (makes 50% fit, but adds comm to timings; still needs accum for 2M);
  (3) cpu_offload_projections (fits 50% on 1 GPU but timing dominated by PCIe — not representative).
- Awaiting Maciej's decision.
