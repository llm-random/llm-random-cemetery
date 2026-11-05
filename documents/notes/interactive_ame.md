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
