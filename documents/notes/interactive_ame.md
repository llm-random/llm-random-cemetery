# AME MFU improvement loop — working log

**Target: AME h20 training MFU > 35% per GPU on entropy 4xH100.**
Standing instruction (Maciej, 2026-11-05): keep trying strategies until it
clears 35%; log what was tried and what is planned here so progress is legible.

This file is the *working* log — hypotheses, attempts, dead ends, next steps.
The durable record of finished results stays in
[ame_optimization_plan.md](ame_optimization_plan.md).

---

## Where we start

| configuration | job | MFU/GPU | step | peak GiB |
|---|---|---:|---:|---:|
| AME baseline B1, mb16 | 32576 | 26.82% | 3.723 s | 60.6 |
| AME: FSDP2 + mem-par + fused, mb16 | 32666 | 27.41% | 3.643 s | 61.90 |
| **AME best: + sink merge, mb16** | **32687** | **30.23%** | **3.302 s** | **63.50** |
| AME, FSDP2 + mem-par + fused, mb4 | 32674 | 23.49% | 4.250 s | — |
| AME, DDP + mem-par + fused, mb8 | 32680 | 25.23% | 3.957 s | 53.53 |
| AME, DDP + mem-par + fused, mb4 | 32664 | 23.02% | 4.337 s | 39.75 |
| **dense H-balanced h20 (the ceiling)** | 32577 | **44.44%** | 2.364 s | 43.4 |

**Gap to target: 27.41% -> 35% needs a 27.7% relative speedup**, i.e. the step
has to come down from 3.643 s to **2.853 s — 790 ms to find.**

## Where the 790 ms could come from

From the Tier 1 matched comparison against the dense twin (jobs 32579/32601,
and the forward/backward split in 32623/32624):

| term | AME | dense | AME excess |
|---|---:|---:|---:|
| **attention kernel** | **1390.7 ms** | 506.9 ms | **+883.8 ms (2.74x)** |
| — of which forward | 730.4 | 139.2 | **+591.2 (5.25x)** |
| — of which backward | 661.0 | 366.0 | +295.0 (1.81x) |
| GEMM | 1264.4 | 1298.4 | -34 (exonerated) |
| NCCL | 936.4 | 309.7 | +626.7 — **mostly removed by Tiers 2+3** |

**One term dominates and it is attention.** +883.8 ms against the 790 ms needed:
closing even two thirds of the attention gap clears the target on its own.
Everything else on the list is small change by comparison.

**Why AME's attention is 2.74x dense:** it runs `flex_attention` rather than
FlashAttention, because it needs an ALiBi distance bias and a learned sink key,
and PyTorch's flash backend accepts neither. The twist recorded in Tier 1 is
that `learnable_slopes=False` is the **default** — so AME is paying for
differentiable slopes it is not using.

---

## Strategy backlog

Ordered by expected value. Status: PLANNED / RUNNING / DONE / REJECTED.

| # | strategy | expected | status |
|---|---|---|---|
| S1 | Measure flex vs an SDPA/flash floor at the real h20 shape | sizes the prize exactly | **DONE — found the sink, see below** |
| S2 | **LSE sink merge** — keep flex, take the sink OUT of the kv axis and merge it afterwards through the log-sum-exp | **-876 ms predicted** | **IMPLEMENTED, measuring** |
| S3 | Tune the flex path in place — block-mask caching, `kernel_options` (BLOCK_M/BLOCK_N), a cheaper `score_mod` | unknown, cheap to try | PLANNED |
| S4 | `reshard_after_forward=False` on the 587M backbone now that the banks are out of FSDP2 | removes the backward all-gather, small | PLANNED |
| S5 | Get the bank path back under compile without the side-effecting backward | recovers what is left of the graph break | PLANNED |
| S6 | Hoist the id all-gather (320 NCCL launches -> 8) | ~95 ms, overlapped so likely invisible | PLANNED |
| — | Activation checkpointing (O10), fused-linear-CE (O11), chunked AdamW (O16), CPU offload (O17) | — | **REJECTED** — cut by the original goal |

## Log

### 2026-11-05 00:35 — entering the loop
Best is 27.41%. DDP explored and parked: it fits at micro-batch 8 (53.53 GiB,
25.23%) and 4 (39.75 GiB, 23.02%), OOMs at 16, and loses to FSDP2 at every
shape where both run. The CE-inside-autocast bug found on the way is fixed and
benefits any future autocast run.

Next: S1, to size the attention prize before engineering against it.


### 2026-11-05 00:36 — S1 result: it is the sink, not flex, and not ALiBi

`scripts/bench_attention.py` on one H100 at the real shape (B=16, H=20, n=2048,
D=64, bf16), scaled by 20 layers x 8 micro-steps to one optimizer step:

| kernel | fwd ms | bwd ms | **per step** |
|---|---:|---:|---:|
| AME flex, ALiBi + sink as kv column 0 | 4.629 | 3.693 | **1331.6 ms** |
| flex, ALiBi fp32, **no sink** | 0.752 | 2.097 | **455.8 ms** |
| flex, ALiBi bf16, no sink | 0.748 | 2.096 | 455.1 ms |
| flex, no score_mod at all, causal only | 0.545 | 1.940 | 397.5 ms |
| SDPA flash, causal, no bias (the floor) | 0.629 | 2.168 | 447.5 ms |

**Three things fall out, and two of them overturn what the plan assumed.**

1. **The sink costs 876 ms per step** — more than everything else in attention
   put together. One extra key.
2. **ALiBi costs 58 ms.** The distance bias, which the whole flex backend exists
   to support, is nearly free.
3. **flex with ALiBi (455.8) already beats the flash floor (447.5) to within
   2%.** Flex is not slow. The Tier 1 conclusion that "AME pays 2.74x for
   attention to keep a capability it is not using" was pointing at the wrong
   capability: it is not the differentiable slopes, it is the sink.

Also settled: **`flash_attn` is not installed** on entropy, so the S2 I had
planned (flash with native `alibi_slopes`) was never available. It does not
matter — flex already matches flash once the sink is out.

**Why one key costs 876 ms.** Two structural reasons, neither arithmetic:
`_with_sink` makes `KV_LEN` 2049, one past flex's 128-wide block alignment, so
every block goes ragged; and the sink needs a `torch.where` inside `score_mod`,
which flex evaluates **once per score** — n^2 per head — to special-case a
single column.

### 2026-11-05 00:40 — S2 implemented: merge the sink through the log-sum-exp

Attention over `{sink} u sequence` is the softmax-weighted merge of attention
over each part, and flex returns the log-sum-exp needed to do it exactly:

    numerator   = out_seq * exp(lse_seq) + exp(q . k_sink) * v_sink
    denominator = exp(lse_seq)           + exp(q . k_sink)

shifted by the running max. O(B*H*n*D) elementwise, no n^2 term. The sink keeps
its exact semantics — no distance bias, visible to every query, `tau` already
folded into `q`.

`KV_LEN` is now 2048 and `score_mod` is two ops with no branch.

**Gate:** the repo's own backend-equivalence tests
(`test_the_two_backends_compute_the_same_function`, parametrised on
`sink` x `shift`, and `test_the_backends_still_agree_under_bf16`) pass at
rtol 1e-5 / atol 1e-6 — flex against the reference path, which materialises the
sink the old way. 64 AME tests green.

**Projected:** 3643 ms - ~870 ms = ~2773 ms/step -> **~36.0% MFU**, which would
clear the target. Measuring now.


### 2026-11-05 00:41 — S2 measured and tuned: 1331.6 -> 679.5 ms/step

Three iterations on the same idea, each measured on one H100:

| version | fwd | bwd | **per step** |
|---|---:|---:|---:|
| original: sink as kv column 0 | 740.7 | 590.9 | **1331.6** |
| LSE merge, written as the ratio | 277.2 | 732.3 | 1009.5 |
| **LSE merge, written as a lerp** | **171.2** | **508.3** | **679.5** |
| (reference: sinkless + `return_lse`) | 120.8 | 335.0 | 455.8 |

**Diagnostics that made the second iteration possible** (same job):

| | ms/step |
|---|---:|
| ragged `KV_LEN = n+1`, no branch | 949.8 — so raggedness alone is **+495** |
| ragged `KV_LEN = n+1`, with branch | 1208.9 — the `torch.where` is **+261** |
| aligned, `return_lse=True` | 455.8 — asking flex for the LSE is **free** |

So the LSE route was sound and the ratio form was the mistake: upcasting the
full `[B, H, n, D]` output to fp32 and building five tensors that size cost 156
ms forward and 403 ms backward. The lerp identity

    out = out_seq + alpha * (v_sink - out_seq),   alpha = sigmoid(s - lse)

is the same arithmetic over `[B, H, n]` for the weight — 1/64th the elements —
and leaves the big tensor in its own dtype. Then written as a broadcast
multiply-add rather than `torch.lerp`, because lerp wants matching shapes and
would expand `v_sink` to full size first.

**652 ms saved so far.** The remaining 224 ms over the sinkless floor is the
merge's own forward (+50) and backward (+173, the sigmoid and the
`bhnd,hd->bhn` einsum that feeds gradients back to `q` and `k_sink`). Note the
benchmark calls `_attend_flex` **uncompiled**, so those ops are separate eager
kernels there; under `compile_blocks_full` inductor should fuse them into the
tail of the attention kernel, so the training number should beat 679.5.

Naive projection from 652 ms: 3643 -> 2991 ms/step = **33.4% MFU**. Short of 35,
but the projection ignores the fusion above. Measuring for real now.

**Next if 35% is not cleared:** S3 (the remaining 224 ms of merge overhead — the
einsum's backward is the biggest piece and could fold into the score_mod),
then S4/S5/S6 from the backlog.


### 2026-11-05 00:47 — S2 measured in training: **27.41% -> 30.23%**

Job **32687**, the pinned test bed, sink merge on top of the fused mem-parallel banks.

| | MFU/GPU | tokens/s | step | peak GiB |
|---|---:|---:|---:|---:|
| B1 baseline | 26.82% | 281,541 | 3.723 s | 60.6 |
| + mem-parallel + fused | 27.41% | 287,729 | 3.643 s | 61.90 |
| **+ sink merge** | **30.23%** | **317,389** | **3.302 s** | **63.50** |

**+12.73% over B1 cumulatively**, and the loss trajectory is unchanged
(11.1875 -> 7.0938, the same values B1 logged). 341 ms off the step, against
652 ms the uncompiled benchmark suggested — the gap is the merge fusing into
the compiled block, and the benchmark's shape differing from the real one (see
below).

**Short of target by 4.77 points.** 35% needs the step at 2.852 s, so there are
**450 ms** left to find.

### 2026-11-05 00:48 — S3 hypothesis: the real run's Q_LEN is not 128-aligned

The benchmark has been running at `--seq 2048`. **The training run does not.**
`common.sequence_length: 2048` gives a batch width of 2048, and
`_preprocess_input` takes `batch[:, :-1]` for next-token targets, so the model
sees **n = 2047** — and `2047 = 15*128 + 127`, one short of flex's block grid.

That is the same class of defect the sink had, on the other axis: the sink made
`KV_LEN` 2049 and cost 495 ms of ragged blocks; `Q_LEN` 2047 is ragged too and
nothing has measured it.

**If it is real the fix is nearly free and arguably more correct:**
`sequence_length: 2049` makes n = 2048, which aligns the grid *and* makes the
model train on exactly the 2048 tokens per sequence the plan says it does.
Job 32688 measures seq 2047 against 2048 to size it before changing anything.

### 2026-11-05 08:10 — S3 confirmed: one token costs **480 ms per step**

Job **32688**, `scripts/bench_attention.py` at both lengths on one H100, B=16
H=20 D=64 bf16, scaled to the 160 attention calls in one optimizer step:

| | fwd | bwd | **total/step** |
|---|---:|---:|---:|
| AME flex, ALiBi + sink merge, **n = 2047** | 482.1 | 674.8 | **1156.9 ms** |
| AME flex, ALiBi + sink merge, **n = 2048** | 171.4 | 505.6 | **677.0 ms** |
| SDPA flash floor, n = 2047 | 118.4 | 352.9 | 471.3 |
| SDPA flash floor, n = 2048 | 101.7 | 347.5 | 449.2 |

**479.9 ms**, and 2.8x of it is in the forward. The hypothesis was right and the
size of it was not a guess worth making — the ragged row-block is 1/16th of the
rows and costs more than the other fifteen together in the forward.

Note the flash floor barely moves (471.3 -> 449.2, 22 ms). This is not a generic
alignment tax on attention; it is specific to how flex handles a partial
row-block, which is the same mechanism that made the sink's `KV_LEN = 2049`
cost 495 ms. Two instances of one bug class, on the two axes of the same matrix.

**480 ms against the 450 ms that 35% needs.** It is on the boundary, so it has
to be measured in training, not projected.

#### The fix, and what it costs

`sequence_length: 2049`. The model sees `batch.shape[-1] - 1 = 2048`.

It is safe to do at config level and nowhere else: `sequence_length` never
reaches the model (grep over `src/` and `main.py` finds it only in
`definitions.py` and `datasets.py`), the MFU accounting already derives its
seq-len from `batch.shape[-1] - 1` (`trainer.py:437`), and `processed_tokens`
counts `input_ids.numel()` — so both sides of the MFU ratio stay honest and the
gain is wall-clock, not accounting.

What it is not: free of consequence. Each example now spans 2049 dataset tokens
instead of 2048, so the token stream is **not** the one B1 saw and the loss
trace will be comparable in shape but not step-for-step identical. That is a
data change, not a numerics change — the section 1.4 gates are untouched by it.

The honest framing for later: `sequence_length` in the config now means batch
width, not model sequence length. The cleaner fix is for the dataloader to
fetch `sequence_length + 1`, which makes every config in the repo correct
rather than this one. Deferred on purpose — not refactoring shared dataloader
code on the strength of an unmeasured end-to-end result.

### 2026-11-05 08:16 — the 4-GPU test bed is occupied; measuring at 1 GPU

`sbatch --test-only` for the 4-GPU job returns:

```
Preempts: 32611,32736,32737,32738,32739
```

All five are **lukaskoz**'s running jobs on h100a — `j321m_h100` preempts
`lukaskoz_h100` with `PreemptMode=CANCEL` and `GraceTime=0`, so submitting
would destroy them outright. 32611 is 11 h 14 m in. **Not submitted.**

h100a has 8 H100s and lukaskoz holds 5 of them at one GPU each, time limits
24-30 h with 1-11 h elapsed; the three free GPUs take a 1-, 2- or 3-GPU job
with no `Preempts:` line at all. Waiting out the 4th GPU means waiting until
tomorrow.

So the alignment fix gets measured where it can be measured honestly: a
**1-GPU A/B**, `h20_1gpu_mb4_mp1` with `common.sequence_length` the single
overridden variable.

| job | width | model n | role |
|---|---:|---:|---|
| **32771** | 2048 | 2047 | control, current behaviour |
| **32772** | 2049 | 2048 | aligned |

This transfers cleanly rather than by luck: at 1 GPU the step does all four
ranks' micro-steps (ga 128 at micro-batch 4 = 2560 attention calls at B=4,
four times the 160-at-B=16 a single rank of the 4-GPU run does), so the
alignment saving scales with it and the *ratio* to step time is the same
~14.5% in both. What 1 GPU cannot show is whether the saving survives contact
with FSDP2's collectives, which is the only reason the 4-GPU run still matters.

`configs/ame/h20_4gpu_mb16_mp4_align.yaml` is written and ready to submit the
moment a 4th GPU frees.

### 2026-11-05 08:30 — alignment measured in training: **28.75% -> 33.39%** at 1 GPU

Jobs **32771** (control) and **32772** (aligned), `h20_1gpu_mb4_mp1` with
`common.sequence_length` the only overridden variable.

| 1 GPU, ga 128 | MFU/GPU | step | tokens/s | peak GiB |
|---|---:|---:|---:|---:|
| control, model n = 2047 | 28.75% | 13.890 s | 75,457 | 64.71 |
| aligned, model n = 2048 | **33.39%** | **11.964 s** | **87,642** | 64.72 |

**1.925 s/step saved against 1.920 s predicted** by the kernel bench (4 x 479.9,
because one GPU at ga 128 / micro-batch 4 does 2560 calls at B=4, four times the
2560 B-units a single rank of the 4-GPU run does). A **100.3% transfer** — the
whole kernel saving reaches the step.

That is worth saying plainly because the sink merge did not: S2 transferred 52%
of its benchmark gain and I expected something similar here. The difference is
that the sink merge's benefit was partly already being captured by inductor
fusing the old form into the compiled block, so the benchmark over-stated what
was left to win; raggedness is inside the flex kernel where no amount of
surrounding fusion touches it. Memory is unchanged (64.71 vs 64.72 GiB), which
is the expected signature: the ragged block was costing time, not space.

#### What this projects to on 4 GPUs -- and why it is still a projection

Per rank the saving is the same 479.9 ms (160 calls at B=16 is the same 2560
B-units), so the step goes 3.302 -> 2.822 s:

  30.23% x (3.302 / 2.822) x 1.00049  =  **35.38%**

and the 1-GPU MFU ratio taken directly gives 30.23% x 1.1616 = **35.11%**.
Both clear 35%, by 0.1-0.4 points. That is too thin to call, and this session
has already produced one projection that was wrong by 20x (the 1-GPU Tier 2
estimate). **It gets measured before it gets claimed.**

### 2026-11-05 08:34 — the full decomposition at n = 2048 (job 32774)

The earlier bench_seq grep truncated the table at three rows. The whole thing,
one H100, B=16 H=20 D=64 bf16, scaled to the 160 attention calls one rank makes
in one optimizer step:

| n = 2048 | fwd | bwd | total |
|---|---:|---:|---:|
| AME flex (ALiBi + sink merge) | 212.9 | 489.3 | **702.2** |
| flex: ALiBi, no sink, `return_lse=True` | 122.2 | 338.8 | 461.0 |
| flex: ALiBi bf16, no sink | 120.6 | 335.4 | 456.0 |
| flex: no score_mod, causal only | 89.4 | 312.3 | 401.7 |
| SDPA flash, causal, no bias (floor) | 100.8 | 346.6 | 447.4 |
| flex: KV = n+1, sink, no branch | 451.7 | 508.2 | 959.8 |
| flex: KV = n+1, sink, with branch | 668.7 | 558.2 | 1226.9 |

Three things fall out of it.

**1. ALiBi is nearly free: 59.3 ms** (461.0 over 401.7), and flex with ALiBi
(456-461) sits at the SDPA flash floor (447.4). The score_mod was never the
problem, which is now measured rather than assumed.

**2. The sink merge costs 241.2 ms** (fwd +90.7, bwd +150.5) and that is the
next target. It should not cost this. Per call the forward moves roughly 255 MB
-- read q (83.9 MB) for the einsum, read out_seq (83.9) and write out (83.9),
plus the small alpha and lse tensors -- which is 0.085 ms at H100 bandwidth
against 0.567 ms measured. **6.7x off the bandwidth floor**, so this is a shape
or fusion problem, not work that has to be paid.

Two candidate causes, not yet separated: `torch.einsum("bhnd,hd->bhn")` lowering
to a degenerate matmul (a reduction over D=64 producing one scalar per element
-- a matrix-vector product per head, the worst case for tensor cores), and a
gradient flowing back through `lse_seq` into flex's backward. The bench's
`return_lse=True` row does NOT measure the second: it returns the LSE but never
backprops through it, so its +3.4 ms of backward is the cost of handing the
tensor back, not of differentiating it.

**3. The KV-column alternative is dead, and now by measurement.** Making the
sink a real KV entry needs a branch in score_mod to stop ALiBi applying to it,
and the branch costs **+267.1 ms** (1226.9 over 959.8) because flex evaluates
score_mod for every score in the matrix. Even padding KV to 2176 to buy back
the raggedness could not pay for that. The merge is the right shape; it is the
arithmetic inside it that is wrong.

**Next: S3b**, the 241.2 ms. On a 2.822 s step that is worth ~3.3 MFU points,
which turns a 35.1-35.4% boundary call into real margin.

### 2026-11-05 08:45 — in flight, and an idea AME's own structure hands us

Running: **32776** (merge variants bench), **32777** (1-GPU aligned run with
`profile_timers` + `kernel_profile`, to find where the other ~2.1 s of the step
goes rather than guessing at it). A background watcher polls `sbatch
--test-only` for a non-preempting 4-GPU slot.

Why a profile now: at 35% the step is 2.822 s and the ideal-FLOPs content is
0.999 s (262,144 tokens/GPU/step x 3.768 GFLOPs at 989 TFLOP/s). Attention is
702 ms of that wall time and the bank path was measured at ~8%. So roughly half
the step is neither attention nor banks and nothing has looked at it.

#### S3b candidates being measured (32776)

Four merge forms, with `out_seq`/`lse_seq` as leaf tensors so flex is out of
the picture: einsum vs a mul+sum reduction over D, each eager and compiled.
The compiled pair matters beyond S3b — **every bench row so far has been eager
while the real model runs this inside a compiled block**, which is the most
likely explanation for why the sink merge transferred 52% and the alignment fix
transferred 100%. Plus one diagnostic row with `lse` detached, which is not
valid math (it drops a real gradient term) but is the only way to separate the
merge's own arithmetic from what flex pays to differentiate its LSE.

#### S3c — sink_score is a property of the token id, not the position

If the measured fixes do not close the 241 ms, there is a structural one that
only exists because this is AME.

`q` is not computed, it is **gathered**: `q = q_bank(token_ids)`. So

    sink_score[b, h, n] = q_bank[ids[b, n]] . k_sink[h]

depends on `ids[b, n]` and nothing else. The whole `[B, H, n, D]` einsum is
recomputing, 16 x 2048 times per block, a quantity that has only `vocab`
distinct values per head.

Precompute `sink_proj = q_bank_weight.view(vocab, H, D) @ k_sink` -- a
`[50304, 20]` table, 4 MB fp32 -- and `sink_score` becomes a gather of 655K
elements (1.3 MB) instead of a reduction over 83.9 MB of q.

The arithmetic on when it pays: building `sink_proj` for one block reads that
block's whole q_bank, 128.8 MB, which is *more* than the 83.9 MB of q it
replaces -- for a single micro-step. But the bank is constant across an
optimizer step, so it is built **once per step and reused across all 8
gradient-accumulation micro-steps**: 128.8 MB once against 83.9 x 8 = 671 MB.
A 5.2x reduction in traffic for this term, and it grows with ga.

The catch, stated now so it is not discovered later: the backward has to route
`d(sink_score)` back into the bank as an index_add over token occurrences
rather than into a dense `dq`. That is the same scatter the bank already does
every step, so it folds into existing machinery -- but it is real work and it
is the reason this is S3c and not S3b. It also only holds while the merge
reads q straight from the bank; any transform between gather and attention
(there is none today) would break it.
