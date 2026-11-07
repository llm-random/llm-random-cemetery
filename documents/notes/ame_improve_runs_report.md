# AME with the banks at 4× the learning rate — report

**Status (2026-11-07 23:15): h12 submitted; h16 next. Nothing scored yet.**

Two runs test the `ame_improve` features on Stage 2's winning AME recipe (S2-PS: ALiBi and
the KV shift). The point of the test is `trainer.bank_lr_ratio: 4.0`, which trains AME's Q/K
banks at 4× the learning rate for the whole schedule. Each run is compared with the same
recipe at ratio 1. Maciej set the task by `/goal` on 2026-11-07. Times are CET.

## Setup

| | |
|---|---|
| Question | Does training the banks at 4× the learning rate lower `final_ce` at h12 (4B tokens) and h16 (8B tokens)? |
| Runs | h12: [ame_h12_ps_bank4.yaml](../../configs/ame_improve/ame_h12_ps_bank4.yaml). h16: [ame_h16_ps_bank4.yaml](../../configs/ame_improve/ame_h16_ps_bank4.yaml) |
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
| h12 | 6264870 | `ai_h12_ps_bank4_lr8_2026-11-07_23-10-29` | | | 23:10 |
| h16 | | | | | |

## Checks

## Results

## For Maciej

1. **The `run_exp.py` fix (`83ebaa85`) is not on `ame_improve` or origin yet.** It is one
   commit on top of 7345eebc and can be cherry-picked as it is. Pushing waits for your OK.
