# AME init follow-up: the new init with the old embedding, then n ± 1 (plan and log)

**Status:** RUNNING, 2026-11-09: step 1 and its extra arm are on LEM; step 2 waits
on step 1.
Background: [ame_init_exp.md](ame_init_exp.md), its Results and Analysis.

**Question:** is the Xavier embedding the whole problem with the new init? If the new
init minus that one change ties or beats the old init at h12, does it win at its own
best learning rate?

## Why

- **The new init lost at the references' n.** In ame_init_exp.md, the bundle is:
  - Xavier embedding, head and banks;
  - pkmf's √(0.1/fan_in) `Linear`s;
  - no weight decay on the embedding.

  Its last-500 paired gaps against the old init: dense h12 +1.271, AME h12 +0.014
  (a tie), dense h16 +0.213, AME h16 +0.032.
- **The embedding is the prime suspect.** Its vectors start at norm 0.17, against 24
  before, and block 0's attention output drowns them: 1,800-2,000:1 by step 100 at
  h12, against 0.3:1.
- **One global LR can't compensate.** Adam moves every weight by about lr per step.
  The new init raised that step's ratio to the init std 140× for the embedding and
  only 3-5× for the head and the `Linear`s.
- **The bundle hides a known gain.** In bank-gates stage 2, σ = 0.01 banks alone beat
  `doi36p8k` by −0.0895 at h12
  ([ame_bank_gates_stage2_report.md](ame_bank_gates_stage2_report.md)). The bundle's
  Xavier banks (σ ≈ 0.006) are close to that, yet AME only tied.

## Step 1: the old embedding, at n = 8

| Run | Model | Config copied | n | Old-init reference, final_ce | New-init run (Xavier embedding), final_ce |
|---|---|---|---|---|---|
| `oldemb_ame_h12` | AME with KV shift and ALiBi (S2-PS) | [configs/ame_init/ame_h12.yaml](../../configs/ame_init/ame_h12.yaml) | 8 | `doi36p8k`, 3.1250 | `ycwqa7k9`, 3.1504 |
| `oldemb_dense_h12` | Dense GQA with RoPE (S1-D) | [configs/ame_init/dense_h12.yaml](../../configs/ame_init/dense_h12.yaml) | 8 | `x0xlqroj`, 2.9824 | `355fr6rn`, 4.2188 |

- **One change from the `ame_init` runs:** the embedding's draw goes back to the old
  one, a cut N(0, 1) with std 0.88, set in each run config:

  ```yaml
  model:
    embedding:
      init_fn:
        _target_: src.core.model.trunc_normal_ # N(0, 1), cut at +-2, as ame_main
        _partial_: true
  ```

- **Everything else stays the new init:**
  - the head and the banks are Xavier;
  - the `Linear`s are √(0.1/fan_in), cut at ±2σ;
  - the embedding is still not decayed. That exemption is a code attribute on this
    branch (`TransformerEmbedding.no_weight_decay_parameters`), not part of the init.
- **Two paired comparisons per run,** both same-seed:
  - **against the old-init reference:** is the rest of the bundle a gain once the
    embedding is normal? The rest is the head, banks, `Linear`s and the undecayed
    embedding.
  - **against the `ame_init` run:** how much did the Xavier embedding cost?
- **Where:**
  - branch `ame_initialization`, worktree `~/code/nano-ame-init`;
  - configs `configs/ame_init/oldemb_<model>_h12.yaml`;
  - W&B project `nano-ame`, run name `ame_init_oldemb_<model>_h12`;
  - seed 123.
- **Submitting:**
  - Submit both at once, the same way as ame_init_exp.md, *Submitting*.
  - Commit everything on the branch first.
  - The two `ame_init` h12 runs took 4 h 14 min (AME) and 3 h 42 min (dense).

**A model survives step 1** if its run's last-500 paired gap against the old-init
reference is below +0.022: a tie or a win. The threshold is AME's tie floor; dense has
no replicate floor of its own and borrows it.

### Step 1's extra arm: the old init without the embedding's decay

Added 2026-11-09 on Maciej's call: the decay may be the embedding's problem. In the
old-init references, the decay alone shrinks the embedding from norm 24.35 to about
10.8 over the run (*Decisions and issues*).

| Run | Model | Config | n | Reference, final_ce |
|---|---|---|---|---|
| `oldinit_embnowd_ame_h12` | AME with KV shift and ALiBi (S2-PS) | [configs/ame_init/oldinit_embnowd_ame_h12.yaml](../../configs/ame_init/oldinit_embnowd_ame_h12.yaml) | 8 | `doi36p8k`, 3.1250 |
| `oldinit_embnowd_dense_h12` | Dense GQA with RoPE (S1-D) | [configs/ame_init/oldinit_embnowd_dense_h12.yaml](../../configs/ame_init/oldinit_embnowd_dense_h12.yaml) | 8 | `x0xlqroj`, 2.9824 |

- **One change from the reference:** the embedding is not decayed, which is this
  branch's default. Everything else is the old init, restored in the config:
  - the embedding draw;
  - scale 1 for the attention and FF `Linear`s;
  - the head's `llm_random_weight_init` at scale 1;
  - AME's `bank_init_std: 1.0`.
- **Two paired comparisons per run,** both same-seed:
  - **against the reference:** the decay alone, on the tuned old init;
  - **against `oldemb_<model>_h12`:** the rest of the new init (the head, banks and
    `Linear`s), with both embeddings the old draw, undecayed.
- **Outside the plan's rules:** the arm decides neither survival nor step 2. The
  plan's rule still judges `oldemb_<model>_h12` against the reference. If the arm
  clearly beats the reference (a gap below −0.022), the decay hurt the old embedding,
  and the survival gap carries that gain; the Analysis then says so.
- **Submitting:** as step 1, beside the pending pair.

## Step 2: n ± 1 for what survives

- **Grid:** for each surviving model, two runs at n = 7 and n = 9. Each is its step-1
  config with only `learning_rate` and the names changed (`..._n7`, `..._n9`), at the
  same seed.
- **Edge rule:** if the best of n = 7, 8 and 9 is an edge (7 or 9), add one run a step
  further (6 or 10). That matches how the references were tuned: each one's n = 8 is the
  best of its own grid (S1-D n = 7-13, S2-PS n = 6-10). Under the old init, dense n = 7
  broke down, so it may here too; it stays a grid point.
- **Nothing survives:** there is no step 2. The Log says so, and the plan ends at
  step 1.
- **Size:** at most 2 runs per surviving model, plus 1 at an edge. Submit each
  model's runs together.

## Checks

As in ame_init_exp.md, *Checks*:
- **About 2 minutes after each start:**
  - the job is running;
  - the first loss is ≈ ln V = 10.83. The Xavier head and the final norm set the
    logits, whatever the embedding;
  - s/step is within 10% of the reference's (AME 3.746, dense 3.272).
- **At step 500 and mid-run:**
  - no NaN;
  - loss spikes and grad norm;
  - the paired gap.
- **The watchers also stop on a loss spike** (a step more than 0.1 above the median of
  the 50 before it, after step 100). The `ame_h16` spike went unseen until its mid-run
  check.
- **Never cancel a job without asking.**

## How to analyze

- **Each run:**
  - final_ce and its Δ against the reference;
  - the paired train-loss gap per 500 steps and over the last 500 steps
    ([scripts/paired_gap.py](../../scripts/paired_gap.py), on the branch);
  - health: spikes, the grad-norm max after step 100, s/step, peak memory;
  - block 0's embedding norm against its attention update norm, the failure seen in
    ame_init_exp.md.
- **The paired gap decides.** AME ties inside ±0.022; dense borrows that threshold.
- **Step 1 verdict, per model:**
  - A tie or a win means the embedding was the whole problem for that model, and the
    rest of the bundle is neutral or a gain.
  - Still clearly worse means it is not the whole problem: the rest hurts too.
- **For AME, also set the gap against bank-gates' −0.0895** for σ = 0.01 banks alone:
  - near −0.09: the head, `Linear`s and undecayed embedding add nothing;
  - well above it: they cost;
  - well below it: they gain.
- **The extra arm:** its gap against the reference is the decay's effect. Below
  −0.022, the decay hurt; inside ±0.022, it made no difference. Its gap against
  `oldemb_<model>_h12` prices the rest of the new init with neither embedding
  decayed.
- **Step 2:** the best n for each surviving model, its gap against the reference
  (n = 8), and whether the new init, at its best n, beats the old.

## Log

### Implementation

2026-11-09, on branch `ame_initialization` (worktree `~/code/nano-ame-init`). No code
change; two run configs:
[oldemb_ame_h12.yaml](../../configs/ame_init/oldemb_ame_h12.yaml) and
[oldemb_dense_h12.yaml](../../configs/ame_init/oldemb_dense_h12.yaml).
- **Each is its `ame_init` h12 config plus the plan's `model.embedding.init_fn`
  block,** with the names and tags changed (W&B tag `ame_init_oldemb` on both).
- **Composed as `run_exp.py` does** (grid point, `check_point`), each differs from its
  `ame_init` original in exactly four keys:
  - `model.embedding.init_fn._target_`, from `torch.nn.init.xavier_normal_` to
    `src.core.model.trunc_normal_`;
  - the W&B name, the tags and the Slurm job name.

  No missing keys.
- **Built on CPU with the real init** (seed 123):

  | | `oldemb_dense_h12` | `oldemb_ame_h12` | Target |
  |---|---:|---:|---:|
  | Parameters | 183,454,464 | 1,103,599,056 | as the `ame_init` runs |
  | Embedding std, max \|w\| | 0.8798, 2.000 | 0.8798, 2.000 | 0.8796, cut at ±2 |
  | Head std | 0.00626 | 0.00626 | Xavier 0.00626 |
  | `q_bank`, `k_bank` std | — | 0.00626, 0.00626 | Xavier 0.00626 |
  | Attention and FF-in `Linear` std | 0.0100 | 0.0100 | 0.0100 |
  | FF-out std | 0.00502 | 0.00502 | 0.00502 |
  | Weight decay 0.1 / 0 (numel) | 144,820,992 / 38,633,472 | 137,743,104 / 965,855,952 | as the `ame_init` runs; the embedding is in the 0 group |
  | Logits std at init | 0.173 | 0.173 | |
  | CE against random targets | 10.840 | 10.831 | ln V = 10.826 |

- **No test run:** nothing in `src/` changed. The init itself is covered by
  `tests/test_init_defaults.py`, and the old embedding draw is `ame_main`'s.

**The extra arm, 2026-11-09:**
[oldinit_embnowd_ame_h12.yaml](../../configs/ame_init/oldinit_embnowd_ame_h12.yaml)
and [oldinit_embnowd_dense_h12.yaml](../../configs/ame_init/oldinit_embnowd_dense_h12.yaml).
Each is its `oldemb_` config with the old init also restored for the `Linear`s, the
head and AME's banks. Checked against the reference's own composition on `ame_main`:
`configs/bank_gates/base.yaml`, which composes `doi36p8k`, and
`configs/stage_1/dense_h12_lr_grid.yaml` set to n = 8 (its grid now lists 10-13; the
init doesn't read n).
- **Composed config:** identical except the names and tags. AME's `bank_init_std` is
  set explicitly to 1.0, the default it had on `ame_main`.
- **Init, built on CPU with seed 123:** every parameter is bit-identical to the
  reference's (dense 111 of 111, AME 171 of 171).
- **Optimizer groups:** the only difference is `embedding.weight`, decay 0.1 → 0.

### Runs

Step 1 was submitted from `~/code/nano-ame-init` at `7a060dd1` (configs plus this
doc), with the plan's command, one `run_exp.py` after the other. Each clone is under
`/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/llmrandom_cemetery/`.

| Run | Job | Clone | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|---|
| `oldemb_ame_h12` | 6287404 | `ame_init_oldemb_ame_h12_2026-11-09_12-14-39` | | | 11-09 12:14 | | PENDING |
| `oldemb_dense_h12` | 6287406 | `ame_init_oldemb_dense_h12_2026-11-09_12-15-01` | | | 11-09 12:15 | | PENDING |

### Checks

### Results

**Step 1:**

| Run | final_ce | Reference | Δ final_ce | Paired gap vs reference, last 500 | Paired gap vs the `ame_init` run, last 500 | Verdict | Survives |
|---|---:|---:|---:|---:|---:|---|---|
| `oldemb_ame_h12` | | 3.1250 | | | (vs `ycwqa7k9`) | | |
| `oldemb_dense_h12` | | 2.9824 | | | (vs `355fr6rn`) | | |

**Step 1, extra arm:**

| Run | final_ce | Reference | Δ final_ce | Paired gap vs reference, last 500 | Paired gap vs `oldemb_<model>_h12`, last 500 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `oldinit_embnowd_ame_h12` | | 3.1250 | | | | |
| `oldinit_embnowd_dense_h12` | | 2.9824 | | | | |

**Step 2:**

| Run | n | final_ce | Paired gap vs reference, last 500 | Verdict |
|---|---:|---:|---:|---|

Best n per surviving model: —.

### Decisions and issues

- **The embedding's weight decay, 2026-11-09 (raised by Maciej).** Step 1 restores
  only the draw; the embedding stays undecayed. The old-init references decayed it at
  0.1, and that decay sets its size. Block 0's input norm (the embedding rows looked
  up), with the old draw's norm shrunk by the decay alone, 24.35 × ∏(1 − lr_t · 0.1):

  | Step | 0 | 1,000 | 2,000 | 3,800 |
  |---|---:|---:|---:|---:|
  | `x0xlqroj`, dense, old init | 24.35 | 17.08 | 13.00 | 10.86 |
  | `doi36p8k`, AME, old init | 24.35 | 16.83 | 12.82 | 10.72 |
  | The decay alone | 24.35 | 16.87 | 12.83 | 10.74 |
  | `355fr6rn`, dense, new init (undecayed) | 0.17 | 3.97 | 4.25 | 4.27 |
  | `ycwqa7k9`, AME, new init (undecayed) | 0.17 | 1.61 | 2.17 | 2.39 |

  - The old embedding's norm follows the decay to within 1%. Undecayed, the step-1
    embedding should stay near 24 all run, 2.2× the references' at the end. Its size
    against block 0's output is the mechanism ame_init_exp.md diagnosed.
  - So the comparison against the `ame_init` run still isolates the draw (both
    undecayed). The one against the old-init reference carries the decay along with
    the head, banks and `Linear`s, and so does the AME check against bank-gates'
    −0.0895: those runs decayed the embedding.
  - **Proposed:** a third arm per model, the old draw plus decay 0.1, i.e. the old
    embedding exactly. Step 2 would then tune, per model, whichever variant has the
    lower gap, if it survives. It needs a flag on `TransformerEmbedding`: the
    exemption is a class attribute (`src/core/model.py`), with no config knob.
    Not run: Maciej chose the arm in the next entry.
- **Maciej's call, 2026-11-09:** "run also old embedding without weight decay, the
  weight decay might be the problem for the embedding". The step-1 runs already are
  the old draw without decay, on the new init. So the new arm is the old init with
  only the embedding's decay removed: a one-change test of the decay on the tuned
  reference. It is *Step 1's extra arm*, above.
