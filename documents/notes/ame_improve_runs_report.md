# AME with the banks at 4× the learning rate — report

**Status (2026-11-08 03:40): h12 finished, and ratio 4 beats ratio 1 by 0.031 in
`final_ce` (3.0938 against 3.1250), well outside AME's run-to-run spread. h16 trains with no
NaN or loss spike, 0.035 ahead of its baseline at the same step, narrowing slowly as the
learning rate anneals (11:40, step 6,351 of 7,625); it ends about 14:10. Checks every 2 h.**

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
| 03:35 | h12 | 3,811 of 3,812 | 3.753 (3.746) | −0.033 (steps 3,712–3,811) | none | **Finished at 03:23** (COMPLETED after 4 h 12 min). `final_ce` 3.09375 against 3.1250; checkpoint and bank state saved at step 3,811. Gap −0.034 to −0.038 per 500-step window from step 1,000 on. See Results |
| 03:35 | h16 | 2,213 of 7,625 | 6.782 (6.810) | −0.046 (steps 2,114–2,213) | none | The lead stopped shrinking: −0.052, −0.047 and −0.047 per 250-step window since step 1,500. Eval gap −0.047 at 10 of the 13 evals from step 1,000 to 2,200 (−0.031 to −0.063 at the rest). Grad norm last 0.042. Effective 7.05 s/step with evals, so it ends about 14:10 |
| 05:38 | h16 | 3,270 of 7,625 | 6.783 (6.810) | −0.042 (steps 3,171–3,270) | none | The lead still shrinks, slowly: −0.050, −0.047, −0.043 and −0.042 per 500-step window since step 1,500. Eval gap −0.031 or −0.047 at every eval since step 2,000. Grad norm last 0.042. Still ends about 14:10 |
| 07:38 | h16 | 4,299 of 7,625 | 6.783 (6.810) | −0.039 (steps 4,200–4,299) | none | The lead stopped shrinking: −0.041 per 500-step window since step 3,000, −0.040 over the last 500 steps. Eval gap −0.031 or −0.047 at every eval since step 2,000. Grad norm last 0.041. Still ends about 14:10 |
| 09:39 | h16 | 5,328 of 7,625 | 6.783 (6.810) | −0.036 (steps 5,229–5,328) | none | The lead narrows slowly as the learning rate anneals: −0.040, −0.038 and −0.037 per 500-step window since step 4,000; h12's narrowed the same way in its last quarter (−0.037 to −0.034). Eval gap −0.031 at 8 of the last 10 evals, −0.047 at the other 2. Grad norm last 0.039. Still ends about 14:10 |
| 11:40 | h16 | 6,351 of 7,625 | 6.782 (6.810) | −0.035 (steps 6,252–6,351) | none | Still narrowing slowly: −0.036 and −0.034 in the last two 500-step windows, −0.035 over the last 500 steps. Eval gap −0.031 at every eval since step 5,100. Grad norm last 0.036. The next check is the end of the run (about 14:10), not a 2-hour mark 30 min before it |

## Results

**How the verdict is made.** Each pair shares the seed, the data order and every setting
except `bank_lr_ratio`. A difference counts only if it exceeds AME's run-to-run spread at
one seed: three runs of one configuration at h12 and 4B tokens end with sd 0.0072, and all
14 AME runs of that study end within 0.015 of each other
([ame_efficiency_parity_report.md](ame_efficiency_parity_report.md)). A difference under
3 sd = 0.022 is a tie. No spread is measured at h16, so the h12 value stands in for it.
`final_ce` is a bf16 value that moves in steps of 0.0156 near 3, which leaves one run's
`final_ce` difference uncertain by about one step. The paired train-loss gap (the same
batches at the same step, averaged over hundreds of steps) is the finer measure.

### h12, 4B tokens: ratio 4 wins by 0.031

| | Ratio 4 (`y5ogvt3k`) | Ratio 1 (`doi36p8k`) | Ratio 4 − ratio 1 |
|---|---:|---:|---:|
| `final_ce` | **3.0938** | 3.1250 | **−0.031** |
| Last 10-batch eval (step 3,800) | 3.0938 | 3.1250 | −0.031 |
| Loss spikes, NaN | none | none | |
| Grad norm, max after step 100 | 0.691 | 0.650 | |
| s/step | 3.753 | 3.746 | +0.2% |
| MFU | 29.8% | 29.8% | |
| Peak memory | 70.40 GiB | 70.40 GiB | |

Paired train-loss gap (ratio 4 − ratio 1 at the same step), mean over each part of training:

| 0–10% | 10–25% | 25–50% | 50–75% | 75–100% | Last 500 steps |
|---:|---:|---:|---:|---:|---:|
| +0.046 | −0.050 | −0.037 | −0.037 | −0.035 | −0.034 |

Readouts at the end of training, averaged over the 12 layers, as in
[stage_2_report.md](stage_2_report.md). α₂ weighs the key shift, β₂ the value shift.

| Run | τ, mean | Shortest horizon | Attention entropy, normalized | Top-8 mass | α₂, mean \|·\| | β₂, mean \|·\| |
|---|---:|---:|---:|---:|---:|---:|
| Ratio 4 | 8.35 | 14.9 | 0.484 | 0.630 | 0.663 | 0.636 |
| Ratio 1 | 8.47 | 15.1 | 0.523 | 0.572 | 0.504 | 0.708 |

- **Ratio 4 wins.** −0.031 in `final_ce` is 4 sd of AME's spread and twice the range of
  its 14 runs. The paired train loss agrees and is steadier: −0.034 to −0.037 over the
  last three quarters of training. The eval gap is −0.031 at 25 of the 27 evals from step
  1,200 on.
- **A slower start.** The first 250 steps ran 0.084 behind (eval +0.094 at steps 100 and
  200). The run led from step 250, by most over 10–25% of training (−0.050), then settled at
  −0.037. No spikes, and grad norms like the baseline's.
- **No cost.** s/step +0.2%, the same MFU and the same peak memory: only the banks'
  learning rate changes.
- **The readouts move as they do under a higher learning rate**
  ([stage_2b_report.md](stage_2b_report.md): sharper attention, larger α₂). Normalized
  entropy falls from 0.523 to 0.484, top-8 mass rises from 0.572 to 0.630, and α₂ rises
  from 0.504 to 0.663. Unlike under a higher learning rate, β₂ drops (0.708 to 0.636). So
  the key shift is now used about as much as the value shift: α₂ > β₂ in 7 of 12 layers,
  against 2 of 12 at ratio 1. τ and the shortest horizon barely move.
- **Against dense** (S1-D, [stage_1_report.md](stage_1_report.md)): the best dense run of
  the h12 grid ends at 2.9824 (n = 8, `x0xlqroj`; n = 9 ends at 3.0244). The gap to it
  shrinks from 0.143 to 0.111, by 22%. Dense's runs spread 6–24× more than AME's, so this
  gap is less certain than the AME-to-AME difference.

### h16, 8B tokens

Pending: the run ends about 14:10.

## For Maciej

1. **The `run_exp.py` fix (`83ebaa85`) is not on `ame_improve` or origin yet.** It is one
   commit on top of 7345eebc and can be cherry-picked as it is. Pushing waits for your OK.
2. **`n_tokens_b` gives 3 and 5 fewer steps than the baselines** (3,812 and 7,625 against
   3,815 and 7,630). It rounds down, and the hand-written baseline configs rounded up. That
   is 0.08% fewer tokens, with the same warmup and a cosine that ends at the same floor. It
   can only handicap ratio 4, and by far less than the differences here.
