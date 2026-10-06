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

## n_blocks = 1 timing model (agreed with Maciej 2026-10-06)
Idea: T(n, b) ≈ F0 + F1·n + (c0 + c1·n)·b — measure n=1 and n=2 blocks at feasible batch sizes, fit,
extrapolate to n=16 and b=100k..2M. Head/embedding/loss are NOT negligible (≈26/30/35% of fwd FLOPs at
50/70/90% due to the 128k vocab) → captured by the n-independent terms, not ignored.

Truncated source checkpoints (no code change needed; get_sharded_sd iterates source keys, so a 16-block
checkpoint crashes a 1-block model): `/storage_nvme_4/nano/j321m/pc/trunc/llama1b_{1,2}blk_in_nano.pt`
(first n blocks + embedding + head) and `importances_{1,2}blk.pt` (dff_importances[:n]). Projections take
n_blocks from len(target_model.encoder.blocks) → consistent.

Configs `configs/pc_project_mas/llama_1B_pc_1blk_bs_probe_{50,70,90}p_entropy.yaml` (common.n_blocks: 1).
Jobs 23381 (50%), 23385 (70%), 23386 (90%). 1×H100, grad_acc 1, 10 steps.

**Max batch with n_blocks=1: 64 seqs = 65,536 tokens for ALL three ratios; OOM at 80 seqs (81,920).**
Same limit for every ratio → the 128k-vocab logits/loss are the binding constraint (~1 MB/token), not width.
(Caveat learned: wandb prints "View run" even on crash and the OOM trace can be flushed later — judge
success by wandb state=finished + step count, not by grepping the slurm log mid-run.)

| ratio | seqs | tokens | s/step | tok/s |
|---|---|---|---|---|
| 50% | 32 | 32,768 | 0.220 | 149,139 |
| 50% | 48 | 49,152 | 0.267 | 183,843 |
| 50% | 64 | 65,536 | 0.329 | 199,066 |
| 70% | 32 | 32,768 | 0.163 | 201,303 |
| 70% | 48 | 49,152 | 0.207 | 237,146 |
| 70% | 64 | 65,536 | 0.246 | 266,314 |
| 90% | 32 | 32,768 | 0.105 | 311,055 |
| 90% | 48 | 49,152 | 0.157 | 314,048 |
| 90% | 64 | 65,536 | 0.182 | 359,660 |

Next: n_blocks=2 PC at {32,48,64}; same for HP (n_blocks 1/2) to get F_HP; then fit + extrapolate.

## Single-block synthetic benchmark — no embedding / no head (2026-10-06, Maciej's idea)
Script `benchmarks/bench_block.py` (repo classes, same dims as the probe configs; random weights/data):
- PC step: materialize W_c=P_out·W·P_in+W_b (fp32, no_grad) → block fwd/bwd under bf16 autocast (≈ FSDP2
  param_dtype=bf16 used by PC training) → projection backward (fp32, = backward_compressed_weights) →
  clip_grad_norm(proj, 0.25) + AdamW(proj); clip(norms, 1.0) + AdamW(norms). W_c not stepped (as in code).
- HP step: block fwd/bwd (bf16 autocast) → clip 1.0 → AdamW(block).
- Input x: random bf16 (b, 1024, d_c), requires_grad=True (middle layer also computes dL/dx). Dummy loss = out.mean().
- Exact peak memory via torch.cuda.max_memory_allocated; median of timed steps; ascending grid, stop at first OOM;
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Run with PYTHONPATH=repo root.
- Validation (50%, 3 steps): PC 8→32 seqs 0.053→0.067 s; HP 0.007→0.022 s → same per-seq slope (~0.6 ms/seq),
  PC fixed ≈ 0.045 s/step/block. Activations ≈ 108 KB/token (50%).
- Full sweep (1×H100): 50/70/90 × pc/hp over seqs 98,196,391,586,782,977,1172,1368,1563,1758,1954,2442,2930,3907
  (= 100k..4M tokens rounded up to 1024 multiples). Log: entropy ~/bench_sweep1.log

### Full sweep results (srun job 23422, 1×H100, 1 block, 10 timed steps)
**Max batch (same for PC and HP): 50% → 586 seqs = 600k tok (OOM 800k); 70% → 782 seqs = 800k tok (OOM 1M); 90% → 1563 seqs = 1.6M tok (OOM 1.8M).** 2M fits for none at grad_acc=1.
PC peak memory ≈ HP + ~2 GiB (projections + AdamW state); e.g. 50%@600k: PC 65.8 vs HP 63.8 GiB.

| ratio | tokens | HP s/step | PC s/step | PC−HP (s) | PC overhead |
|---|---|---|---|---|---|
| 50% | 100,352 | 0.0634 | 0.1098 | +0.0464 | 73.1% |
| 50% | 200,704 | 0.1261 | 0.1723 | +0.0462 | 36.6% |
| 50% | 400,384 | 0.2496 | 0.2951 | +0.0454 | 18.2% |
| 50% | 600,064 | 0.3747 | 0.4169 | +0.0422 | 11.3% |
| 70% | 100,352 | 0.0464 | 0.0757 | +0.0293 | 63.2% |
| 70% | 200,704 | 0.0918 | 0.1208 | +0.0290 | 31.6% |
| 70% | 400,384 | 0.1829 | 0.2106 | +0.0277 | 15.1% |
| 70% | 600,064 | 0.2743 | 0.3021 | +0.0277 | 10.1% |
| 70% | 800,768 | 0.3674 | 0.3956 | +0.0282 | 7.7% |
| 90% | 100,352 | 0.0312 | 0.0424 | +0.0111 | 35.7% |
| 90% | 200,704 | 0.0612 | 0.0724 | +0.0112 | 18.3% |
| 90% | 400,384 | 0.1181 | 0.1294 | +0.0113 | 9.6% |
| 90% | 600,064 | 0.1787 | 0.1896 | +0.0109 | 6.1% |
| 90% | 800,768 | 0.2383 | 0.2491 | +0.0109 | 4.6% |
| 90% | 1,000,448 | 0.3002 | 0.3103 | +0.0100 | 3.3% |
| 90% | 1,200,128 | 0.3442 | 0.3554 | +0.0112 | 3.2% |
| 90% | 1,400,832 | 0.4023 | 0.4128 | +0.0105 | 2.6% |
| 90% | 1,600,512 | 0.4596 | 0.4705 | +0.0108 | 2.4% |

- PC−HP is CONSTANT per ratio, independent of batch: ≈0.045 s (50%), ≈0.028 s (70%), ≈0.011 s (90%) per block per step → direct empirical confirmation of the Fig. 2 claim (fixed per-step overhead, amortized by batch). HP throughput flat (~1.60M / 2.18M / 3.45M tok/s), PC approaches it.
- Extrapolated to 2M tokens (HP time = 2M / HP tok/s): 50% 0.045/1.25 s ≈ 3.6%; 70% 0.028/0.92 s ≈ 3.1%; 90% 0.011/0.58 s ≈ 1.9%.
- Scope: per-BLOCK numbers. Full model adds head/embedding: HP-equal per-token compute (lowers the relative overhead) + PC head/embedding projection fixed cost (raises it) — to measure separately if the paper figure should be whole-model.

## 100-step sweep @ 90% (2026-10-06) — the Fig. 2 data
16 batch sizes every ~100k tok (98..1563 seqs = 100k..1.6M), PC and HP, each a separate process on the SAME
1×H100 (srun job 23428), PC/HP interleaved per size. 100 steps, **mean step time over last 50**.
Raw: `benchmarks/results/bench_block_90p_100step.jsonl` (incl. per-step times). Plots: `mastest_pc_time.ipynb`
(plotly express; run in conda env `nanollmr`).

Result: PC − HP ≈ **10–12 ms per step, constant** across 100k → 1.5M tokens; relative overhead
35.0% @100k → 9.5% @400k → 4.6% @800k → 3.4% @1.0M → **2.5% @1.5M**. Step-time std < 1.1 ms everywhere.
HP throughput ~3.2–3.5M tok/s (small kernel-selection jump between 1.0M and 1.1M, same for PC and HP).

**31/32 runs OK. PC @ 1563 seqs (1.6M tok) fails reproducibly** (HP @1.6M OK; PC @1.6M passed in the earlier
10-step in-process sweep): `CUDA error: illegal memory access` in loss.backward() after the warm-up steps
(x2), and without expandable_segments `CUBLAS_STATUS_EXECUTION_FAILED` in a bf16 cublasGemmEx. Activation
tensors there exceed 2^31 elements (attention q ≈ 3.3e9) → likely a PyTorch/cuBLAS large-tensor edge case,
not our code. Not pursued; curve uses 15 complete pairs (100k–1.5M) + HP-only point at 1.6M.

## Paper Fig. 2 updated locally (2026-10-06) — NOT pushed, awaiting Maciej's review
- Original generator: `~/code/nano_local/nano/mastest.ipynb` cell 1 (analytic 6rd(d+r)/(6br^2), d=2048, r=960, plotly gold).
- New generator: `benchmarks/plot_overhead_fig2.py` = same analytic gold curve + measured blue (90%, 1 block,
  31 PC/HP pairs 50k..1.55M from both sweeps). Output `pc_paper/<id>/Figures/projection_compression_overhead_measured.png`
  (old PNG kept). mainAAAI.tex: \includegraphics → new PNG, new caption (both curves described), +1 sentence in
  "Amortization at Scale" paragraph citing ≈11 ms constant / 2.5% at 1.5M. Local PDF builds clean (19 pages).
- Midpoint sweep (srun 23449): 32/32 OK incl. PC @1.55M (only PC @1.6M fails, see above).
- Caveat for the paper: blue is 90% (r=448) while gold is r=960 (70%); measured ≈4–5× above analytic at ≥1M
  (likely fp32 projection math vs bf16 per-token compute — not profiled).

## Measurement window decision (Maciej, 2026-10-06)
Per-step times over the measured window (steps 51–100) are flat for all runs → method accepted:
100 steps, mean over steps 51–100. The 300-step fully-timed warm-up check (job 23453) was cancelled — not needed.
`bench_block.py` now also stores `all_step_times` (warm-up included) for future runs.

## 70% sweep — matches the analytic Fig. 2 setup (d=2048, r=960) (2026-10-06)
srun job 23454, 1×H100, 1 block, 19 batch sizes every 50k (49..928 seqs = 50k..950k), PC+HP, 100 steps,
mean of steps 51–100. **38/38 OK** — 70% one block fits ≥950k tokens (the earlier 10-step probe OOM was at 1M).
Raw: `benchmarks/results/bench_block_70p_100step.jsonl`.
PC − HP ≈ **25–29 ms/step, ~constant** (drifts 29→25 ms); overhead 121% @50k, 61.8% @100k, 30.5% @200k,
14.6% @400k, 9.4% @600k, 7.3% @800k, **6.0% @950k**. Analytic (r=960) at 950k = 0.68% → measured ≈ 9× analytic.

Paper Fig. 2 (local only, not pushed): now matched — gold analytic r=960 + blue measured 70%, ymax 20%.
`Figures/projection_compression_overhead_measured.png` = 70% version (included); `..._measured_90p.png` = earlier
90% version (not included). Caption + text sentence updated to 70% numbers (≈26 ms, 62% @100K → 6.0% @950K).
Notebook `mastest_pc_time.ipynb` now loads 70% + 90% (101 runs), plots faceted/colored by ratio.

## Fig. 2 switched to 90% (Maciej, 2026-10-06) — local only, not pushed
Gold analytic now r=448 (Llama 1B at 90%), blue measured 90% (31 pts, 50k–1.55M), ymax 20%.
Included file: `Figures/projection_compression_overhead_90p.png`. Kept for switching back:
`..._70p.png` (gold r=960 + measured 70%) and the original `projection_compression_overhead.png`.
Caption + in-text sentence reverted to 90% numbers (≈11 ms constant; 35% @100K → 2.5% @1.5M).
At 90%, measured ≈ 3× analytic at ≥1M (vs ≈9× at 70%). 70% vs 90% comparison: 70% relative overhead ≈1.6×
higher (fixed cost 2.4× larger ≈ analytic 2.6×; HP per-token time only 1.5× larger vs r² model's 4.6×).
