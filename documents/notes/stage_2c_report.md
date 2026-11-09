# Stage 2c — the best AME variant at h20: report

**Status: in progress (started 2026-11-08 00:40). Moved from Helios to LEM with Stage 1c
at 18:16, at Maciej's instruction; no Helios job had started
([stage_1c_report.md](stage_1c_report.md), decision 6). On LEM the AME smoke run passed
every check (18:23–18:34). Stage 1c finished at 07:24 on 2026-11-09 with n_d\* = 9, so this
stage's grid is n = 8, 9 and 10 (decision 8). n = 9 has run since 02:43 and n = 10 since
06:02. n = 8, submitted at 06:06, waits in the queue. n = 11, queued at 02:50 while Stage
1c's n = 10 led, was cancelled at 07:29, before it started.**

The plan is [ame_experiments.md, "Stage 2c — the best AME variant at h20"](ame_experiments.md#stage-2c--the-best-ame-variant-at-h20).
This file is its record: every job, result and decision, each with its reason. Times are
CET. Stage 1c, which supplies the dense h20 baseline and n_d\*, is
[stage_1c_report.md](stage_1c_report.md). That report also holds what the two stages share:
the Helios compile fix, the pre-flight and the submission method. Stage 2b's record, the
same arm at h16, is [stage_2b_report.md](stage_2b_report.md).

## Result

Pending.

## Setup

| | |
|---|---|
| Question | How far is AME's best variant, ALiBi with the KV shift, from dense at h20 and 16B tokens? And does the gap keep shrinking with scale? |
| Arm | S2c-PS, one arm only (Maciej, 2026-11-07): S2b-PS on Stage 1c's model, recipe and hardware. `shift: true`, so α and β are learned per head and layer; the slopes are unset, so AME uses its ALiBi ladder, horizons 16 to 8,192 tokens over 20 heads; `bank_gate: none` |
| Code and configs | Commit **`a8553b7e`** on branch `stage1c2c_lem`, the same as Stage 1c: `683c316b`'s code plus the config commit that moves both stages to LEM (`ame_main`'s `acb68ff5`, cherry-picked), submitted from `~/code/nano-stage1c2c` ([stage_1c_report.md](stage_1c_report.md), decision 7). Since Stage 2b's `89975bbf`, AME gained the bank-gates option (`157c3122`), off by default: with `bank_gate: none` the layer computes what it did in Stage 2b (only the q/k reshape moved after the gather) and draws nothing more from the RNG. The trainer gained readouts only (`train/clip_factor`, `train/grad_norm/banks`) and opt-in token counts, which are off; the clip itself is unchanged |
| Config | [configs/stage_2c/ame_h20_pe_shift.yaml](../../configs/stage_2c/ame_h20_pe_shift.yaml), `learning_rate: null`, set per run as `trainer.learning_rate=<n>`. One submit is one run |
| Cluster, recipe | Stage 1c's: LEM since 18:16 on 2026-11-08, one node of 4 H100 per run with r16-11 excluded, FSDP2 with `reshard_after_forward: false`, bf16 from FSDP2's mixed precision, micro-batch 16 × 8 per GPU, 15,259 steps = 16.00B tokens, warmup 152, seed 123, 24 h limit. The banks are column-sharded over the 4 GPUs (`mem_parallel_size: 4`) |
| Learning rate | n = 9 alongside Stage 1c; then n_d\* − 1, n_d\* and n_d\* + 1 around Stage 1c's best n, extended past an edge if needed. Stage 1c gave n_d\* = 9, so n = 8, 9 and 10 |
| W&B | `ideas_cv/nano-ame`, runs `s2c_ps_h20_lr<n>.0`, tag `stage_2c` |

**Model**, built on the meta device ([stage_1c_report.md](stage_1c_report.md), pre-flight):
3,163,200,720 parameters, of which 2,575,564,800 are banks, 458,857,680 backbone and
128,778,240 embedding and head; 3.7683 GFLOPs per token at seq 2048. The banks are 5.1×
the dense model's whole parameter count; the FLOPs per token are 0.95× dense's.

**What the comparison with dense matches and what it does not:**

| Matched | Not matched |
|---|---|
| tokens, data order (4 GPUs) | total parameters: AME has 5.1× |
| recipe, precision path, hardware | FLOPs per token: AME does 0.95× |
| | positional signal: ALiBi with the shift, against RoPE |

**Weight decay differs between the arms.** AME's Q/K banks are exempt from weight decay,
while the dense W_Q and W_K are not
([archive/ame_review_and_fixes.md](archive/ame_review_and_fixes.md) 2.6). Every AME-vs-dense
number in this file carries that difference.

**Readouts.** FSDP2 is on, so α₂, β₂, τ, the horizons and the attention entropy are not
logged: each AME layer's `instrumentation()` fails once with a warning and is dropped
([src/core/trainer.py](../../src/core/trainer.py), `_log_layer_instrumentation`). No fix had
reached `ame_main` by `683c316b`. So α₂, β₂ and τ are read from each final checkpoint, and
attention entropy and top-8 mass are left out, as the plan says.

## Jobs

| Job | What | n | W&B run | Node | Submitted | Started | Ended | State | `final_ce` |
|---|---|---:|---|---|---|---|---|---|---:|
| 25862201 | Helios 4-GPU smoke: the S2c-PS file, 60 steps | 9 | — | — | 01:14 | never | 18:16 | cancelled, never started | — |
| 25862255 | Helios **S2c-PS** | 9 | — | — | 01:15 | never | 18:16 | cancelled, never started | — |
| 6274942 | LEM 4-GPU smoke: the S2c-PS file, 60 steps | 9 | `yss8c50l` | r16-15 | 18:23 | 18:23 | 18:34 | completed | 7.3535 (60 steps) |
| 6275194 | **S2c-PS** | 9 | `g3gw9wru` | r18-9 | 18:35 | 11-09 02:43:01 | | running | |
| 6277615 | **S2c-PS** | 10 | `5xclzzfy` | r10-11 | 20:23 | 11-09 06:02:52 | | running | |
| 6282280 | **S2c-PS** | 11 | — | — | 11-09 02:50 | never | 11-09 07:29 | cancelled, never started (decision 8) | — |
| 6284092 | **S2c-PS** | 8 | | | 11-09 06:06 | | | pending | |

## Smoke run

**On Helios: cancelled before it started.** Job 25862201 was submitted at 01:14 with the
real runs, was still pending at 18:16, and was cancelled with the move to LEM.

**On LEM: job 6274942**, `smoke_s2c_ps_h20_lr9`, the S2c-PS file run from `a8553b7e`. It
uses `trainer.learning_rate=9 trainer.n_steps=60 trainer.eval_interval=20` and a 1.5 h
limit, with r16-11 excluded, and was submitted at 18:23, before the real run
([stage_1c_report.md](stage_1c_report.md), decision 8). It checks:
- that micro-batch 16 fits under FSDP2 with the banks sharded over 4 H100s (93.6 GiB each;
  64.63 GiB peak on entropy, §1.15);
- the s/step against the plan's 3.14–3.25, which was estimated for LEM;
- that `final_ce` is logged and the checkpoint written (backbone shards and each GPU's bank
  shard);
- that the readouts fail with a warning per layer while the run goes on.

**Results: every check passes.** The job completed with exit code 0 in 10 min 30 s on
r16-15 (W&B `yss8c50l`):

| Check | Result | Plan |
|---|---|---|
| Peak memory | **64.63 GiB**, the same as on entropy (§1.15) | under LEM's 93.6 |
| s/step, median over steps 15–59 | **2.964**, steady at 2.94–2.97 between evals | 3.14–3.25 |
| Eval cost | about 16–21 s each: steps 21 and 41 took 24.3 and 18.9 s | — |
| `final_ce` after 60 steps | 7.3535 | logged |
| Checkpoint | 37,964,561,848 bytes: 4 FSDP2 backbone shards (1.76 GB each), each GPU's bank shard `memory_parallel_rank<k>.pt` (7.73 GB each), the metadata and the training state | about 37 GB |
| Readouts | 80 warnings, one per layer on each of the 4 ranks ("instrumentation … failed and is disabled for the rest of this run"); no AME readout keys in W&B; the run went on | one warning per layer |

- **2.964 s/step is 354k tokens/s, 33.7% MFU** against 4 × 989 TFLOP/s, computed offline. It
  is 1.21× dense's 2.443 on the same nodes.
- **Whole-run estimate on LEM: about 13.4 h**, which fits the 24 h limit with 1.8× margin:
  - 15,259 steps × 2.964 s = 12.6 h;
  - 152 evals × about 18 s = 0.75 h;
  - about 6 min to start, then `final_ce` and the checkpoint.
- The checkpoint was deleted at 18:35:05.

## Decisions

1. **The n = 9 run was submitted at 01:15, alongside Stage 1c's grid**, as the plan
   allows. n = 9 is in Stage 2c's grid whichever of 8, 9 or 10 Stage 1c picks, and it was
   n\* at h12 and h16, so it is the like-for-like point across the three scales. The other
   two runs wait for Stage 1c's best n, or for its leader to be clear mid-run.
2. **Moved to LEM with Stage 1c at 18:16** ([stage_1c_report.md](stage_1c_report.md),
   decisions 6–8). Helios jobs 25862201 and 25862255 never started and were cancelled. On
   LEM the smoke run goes first, then n = 9, which still runs alongside Stage 1c's grid
   for the reason in decision 1. Both stages stay on one cluster, as the plan requires.
3. **S2c-PS at n = 9 was submitted at 18:35, as soon as the AME smoke run passed.** It
   waits in LEM's queue with Stage 1c's grid ([stage_1c_report.md](stage_1c_report.md),
   decision 9).
4. **S2c-PS at n = 10 was submitted at 20:23, before Stage 1c's best n is known.**
   - **n = 10 is in the grid whichever of 9, 10 or 11 Stage 1c picks.** The grid is
     n_d\* − 1, n_d\* and n_d\* + 1. Stage 1c's n = 8 can no longer be best: at step 2,000
     it is 2.1 nats behind n = 9 ([stage_1c_report.md](stage_1c_report.md), decision 10).
     This is the plan's reason for starting n = 9 early, applied once n = 8 dropped out.
   - **Only n_d\* ≥ 12 would leave n = 10 out.** For that, n = 10 must beat n = 9, and then
     n = 11 must beat n = 10. Dense's best was n = 8 at h12 and n = 9 at h16. If it
     happens anyway, the run is one more grid point, about 58 GPU-h.
   - **Waiting would cost queue age.** On LEM, age adds about 59 priority points an hour
     (weight 10,000, full after about 7 days; n = 9 had 105 after 1.8 h). So five hours of
     age equal our whole fairshare term, 303. Meanwhile the queue moves: between 18:40 and
     20:20, Slurm moved n = 9's forecast from 2026-11-09 08:28 to 2026-11-10 04:11.
   - **The third run waits:** n = 8 if n_d\* = 9, n = 11 if n_d\* = 10.
   - **It runs on the same code.** It was submitted with the same script from the worktree
     at `a8553b7e`, adding `trainer.learning_rate=10`. Its clone's `src` and `configs`
     trees match those of `a8553b7e` and of the n = 9 clone, and its config differs from
     n = 9's only in `learning_rate`.
5. **The third run still waits (2026-11-09 00:30).** Stage 1c's n = 10 leads n = 9 by
   0.012 a window, steadily since step 3,600 ([stage_1c_report.md](stage_1c_report.md),
   decision 12).
   - **That lead is enough for Stage 1c's early start of n = 11**, which costs one dense
     run if it is wrong.
   - **It does not yet make the leader clear for Stage 2c.** At h16, the gaps between
     neighbouring n moved by up to 0.014 after 40% of the run, about the size of this lead.
   - **A wrong guess here costs a 58 GPU-h AME run on the wrong side of the grid.** The
     third run is n = 11 if n_d\* = 10, and n = 8 if n_d\* = 9.
   - **Waiting costs little.** The run would queue behind Stage 2c's n = 9 and 10 and
     Stage 1c's n = 11, which are all older.
   - **Next check:** at about 03:00, when n = 10 is about 60% through. Otherwise when n = 9
     and n = 10 finish, at about 06:00 and 07:30.
   - **The snapshots of n = 9 and 10 carry two stale grid configs**, which are never
     read ([stage_1c_report.md](stage_1c_report.md), decision 12).
6. **S2c-PS at n = 11 was queued at 02:50 on 2026-11-09, as job 6282280, before Stage 1c's
   best n is certain.**
   - **Stage 1c's n = 10 is likely to end best.** At 58% of its run it leads n = 9 by
     0.010, and it has led in every 200-step window since step 200. The lead narrows by
     about 0.0005 per 1,000 steps, so at that pace n = 10 ends about 0.007 ahead
     ([stage_1c_report.md](stage_1c_report.md), Progress, 02:50).
   - **It is not certain.** At h16, late in the run, the higher-LR run of a neighbouring
     pair gained up to 0.012 on the lower one. Here the higher-LR run is n = 9, 0.010
     behind.
   - **Queue position matters more than decision 5 assumed.** It expected the run to queue
     only behind our own older jobs. In fact 146 tasks of other users, 197 GPUs in all,
     wait between our n = 10 (priority 895) and Stage 1c's n = 11 (646). A job queued now
     starts at 515 and gains about 59 an hour. So each hour of waiting lets the jobs
     others queue in that hour go ahead of it.
   - **A pending job can be cancelled at no cost.** n = 11 queues behind all of those, so
     it should still be pending when Stage 1c's n = 9 and n = 10 finish, at about 06:05
     and 07:25.
   - **If Stage 1c's n = 9 ends best, n = 11 is cancelled and n = 8 submitted.** The grid
     is then 8, 9 and 10, and n = 11 would be needed only if AME's best landed on n = 10,
     that grid's edge. At h16, AME's best n was one below dense's
     ([stage_2b_report.md](stage_2b_report.md)).
   - **n = 8 may be needed even if Stage 1c's n = 10 ends best**, as the edge extension,
     if AME's best lands on n = 9. Stage 2c's own results decide that, one run at a time.
   - **It runs on the same code.** Its clone is `s2c_ps_h20_lr11.0_2026-11-09_02-49-43`,
     snapshot `499879a`:
     - its `src` and `configs` trees are those of `a8553b7e` and of n = 10's clone;
     - its config differs from n = 10's only in `learning_rate`, and its `exp.job` not at
       all;
     - its snapshot holds only `config_0.yaml`, since the submit now clears the directory
       first.
7. **S2c-PS at n = 8 was queued at 06:06 on 2026-11-09, as job 6284092.**
   - **Stage 1c's n = 10 lost its lead as the LR decayed.** It fell from 0.010 at 57% of
     its run to nothing at 83%, and at 86–87% n = 9 led by 0.0002–0.0004
     ([stage_1c_report.md](stage_1c_report.md), Progress, 06:05). So n = 9 may end best,
     and then this stage's grid is 8, 9 and 10.
   - **n = 8 is likely needed either way.** If Stage 1c's n = 10 ends best, n = 8 is the
     edge extension should AME's best land on n = 9. At h16, AME's best n was one below
     dense's ([stage_2b_report.md](stage_2b_report.md)).
   - **Queuing now is free and keeps its place**, as in decision 6. It ranks behind Stage
     1c's n = 11, this stage's n = 11 and about 150 tasks of other users.
   - **At Stage 1c's n = 10 finish, about 07:25:**
     - if n = 9 has the lower `final_ce`, n = 11 (6282280) is cancelled while pending and
       n = 8 stays;
     - if n = 10 has, n = 11 stays. n = 8 then stays queued while AME's n = 9 leads its
       n = 10 mid-run, the plan's early start for an edge, and is cancelled while pending
       once AME's n = 10 leads clearly.
   - **It runs on the same code.** Its clone is `s2c_ps_h20_lr8.0_2026-11-09_06-05-44`,
     snapshot `8c39f4e`. Its `src` and `configs` trees are `a8553b7e`'s, its config differs
     from n = 10's only in `learning_rate`, its `exp.job` not at all, and its snapshot
     holds only `config_0.yaml`.
8. **The grid is n = 8, 9 and 10 (07:29 on 2026-11-09).** Stage 1c's n = 10 finished at
   07:24 with `final_ce` 2.5098, above n = 9's 2.5059, so n_d\* = 9
   ([stage_1c_report.md](stage_1c_report.md), decision 14).
   - **n = 11, job 6282280, was cancelled at 07:29, before it started**, as decision 7 set
     out. It is outside the grid, and would be needed only if AME's best landed on n = 10,
     the grid's edge.
   - **n = 8, job 6284092, stays queued**, since it is now in the grid. Slurm forecasts its
     start at 13:15.
   - **All three runs use the same code**, from the worktree at `a8553b7e` (decisions 4, 6
     and 7).
   - **Next: the edge rule.** If n = 10 ends best, n = 11 is added; if n = 8 does, n = 7.
     As in Stage 1c, each extension starts early, once the edge leads mid-run.

## For Maciej

Nothing yet.
