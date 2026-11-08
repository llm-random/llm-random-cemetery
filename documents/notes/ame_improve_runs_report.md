# AME with the banks at 4× the learning rate — report

**Status (2026-11-08 01:20): both runs train with no NaN or loss spike, and both are ahead
of their baselines at the same step: h12 by 0.038, h16 by 0.071 and shrinking (mean train
loss over the last 100 steps). h12 ends about 03:20, h16 about 13:45. Checks every 2 h.**

Two runs test the `ame_improve` features on Stage 2's winning AME recipe (S2-PS: ALiBi and
the KV shift). The point of the test is `trainer.bank_lr_ratio: 4.0`, which trains AME's Q/K
banks at 4× the learning rate for the whole schedule. Each run is compared with the same
recipe at ratio 1. Maciej set the task by `/goal` on 2026-11-07. Times are CET.

## Setup

| | |
|---|---|
| Question | Does training the banks at 4× the learning rate lower `final_ce` at h12 (4B tokens) and h16 (8B tokens)? |
| Runs | h12: [ame_h12_ps_bank4.yaml](../../configs/ame_improve/ame_h12_ps_bank4.yaml). h16: [ame_h16_ps_bank4.yaml](../../configs/ame_improve/ame_h16_ps_bank4.yaml). Both are on branch `ame_improve_runs`, not on `ame_main` |
| Baselines (ratio 1) | h12: S2-PS n = 8, W&B `doi36p8k`, job 6256577, `final_ce` 3.1250. h16: S2b-PS n = 8, W&B `hjw7znbz`, job 6260570, `final_ce` 2.8594 |
| What differs from the baselines | `bank_lr_ratio` 4.0 (the baselines predate the key, so 1), and the length: `n_tokens_b` 4 and 8 resolve to 3,812 and 7,625 steps at 512 × 2049 tokens per step, against 3,815 and 7,630. Warmup is 38 and 76 steps in both. Everything else equals the baselines' logged W&B configs |
| Recipe | LEM, 1 H100 per run, r16-11 excluded. n = 8 (LR 2⁻⁸; banks at 2⁻⁶), seed 123, batch 512 × 2049, micro-batch 32 (h12) and 16 (h16), no FSDP2, Tier 2 banks (`mem_parallel_size: 1`), autocast bf16, `compile_blocks_full`, final checkpoint only |
| Code | Branch `ame_improve_runs` in `~/code/nano-ame-improve-runs`, cut from `ame_improve` 7345eebc: configs in `cb7704cd`, a `run_exp.py` fix in `83ebaa85` (decision 1) |
| Time limits | h12 7 h (the run takes about 4.5 h); h16 24 h (about 15 h) |

## Decisions

1. **`run_exp.py` could not submit a config with `n_tokens_b`; fixed in `83ebaa85`.** The
   first h12 submit (23:07) failed on the laptop, before anything was pushed or queued.
   `run_exp.py` first calls `OmegaConf.missing_keys(cfg)`, which resolves every
   interpolation it passes. With `n_steps` still null, the warmup
   `${eval:'int(${trainer.n_steps} * 0.01)'}` fails on `None * 0.01`. The check now runs on
   a copy with the token budget resolved, and the config keeps `n_tokens_b` as written. A
   config without `n_tokens_b` is checked exactly as before. The feature's tests went through
   `check_point` and `main.run` but never through this step. Two tests now cover it. The fix
   is on `ame_improve_runs` only: `ame_improve` is checked out in another worktree, which this
   task leaves alone ("For Maciej").

## Jobs

| Run | Job | Cemetery clone (under `.../nano/plgmstefaniak/llmrandom_cemetery/`) | W&B run | Node | Submitted |
|---|---|---|---|---|---|
| h12 | 6264870 | `ai_h12_ps_bank4_lr8_2026-11-07_23-10-29` | [`y5ogvt3k`](https://wandb.ai/ideas_cv/nano-ame/runs/y5ogvt3k) | r12-11 | 23:10 |
| h16 | 6264872 | `ai_h16_ps_bank4_lr8_2026-11-07_23-11-17` | [`hn1qal9z`](https://wandb.ai/ideas_cv/nano-ame/runs/hn1qal9z) | r16-3 | 23:11 |

Both jobs started within a minute of submission. Training began at 23:13 (h12) and 23:14
(h16); FineWeb opened in under 2 min.

## Checks

Each check reads W&B. s/step is the median of `train/step_time`, the stage reports'
convention. The loss gap is new − baseline at the same step: both runs see the same batches
at the same step, so batch difficulty cancels. A spike is a train loss more than 0.5 above
the median of the previous 50 steps. Train loss and `final_ce` are bf16 values: they move
in steps of 1/64 = 0.0156 between 2 and 4, so a single step's gap means little, and window
means are used instead.

| Time | Run | Step | s/step (baseline) | Loss gap, mean over window | NaN / spikes | Notes |
|---|---|---:|---|---|---|---|
| 23:15 | h12 | 27 | 3.755 (3.746) | −0.002 (steps 0–27) | none | **Start check passed.** The log resolves `n_tokens_b` 4 to 3,812 steps; 1.053 GFLOP/token as the baseline; `bank_lr`/`lr` = 4.000000 at every step; alpha/beta/tau/entropy readouts logged |
| 23:15 | h16 | 7 | 6.777 (6.810) | 0.000 (steps 0–7) | none | **Start check passed.** 7,625 steps; 2.121 GFLOP/token as the baseline; `bank_lr`/`lr` = 4.000000; readouts logged |
| 01:18 | h12 | 1,888 of 3,812 | 3.753 (3.746) | −0.038 (steps 1,789–1,888) | none | Behind over steps 0–250 (+0.084; eval +0.094 at steps 100 and 200), ahead since: −0.045 to −0.049 per 250-step window until step 750, then flat at −0.036 to −0.038 from step 1,000. Eval gap −0.031 (two bf16 steps) at every eval from 1,200 to 1,800. Grad norm after step 100 as the baseline's: max 0.691 at step 143 (baseline 0.650 at 130). `bank_lr`/`lr` = 4 throughout |
| 01:18 | h16 | 1,043 of 7,625 | 6.782 (6.810) | −0.071 (steps 944–1,043) | none | Ahead from the start, but the lead is shrinking: −0.068, −0.160, −0.113, −0.081 and −0.067 per 250-step window; eval gap −0.19 at step 400, −0.047 at 1,000. h12's lead shrank the same way until about a quarter of its training, then held. Grad norm as the baseline's: max 0.708 at step 140 (baseline 0.715 at 121). `bank_lr`/`lr` = 4 throughout |

## Results

## For Maciej

1. **The `run_exp.py` fix (`83ebaa85`) is not on `ame_improve` or origin yet.** It is one
   commit on top of 7345eebc and can be cherry-picked as it is. Pushing waits for your OK.
