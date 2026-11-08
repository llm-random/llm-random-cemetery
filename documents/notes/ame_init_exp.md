# AME init experiment: Xavier embedding, head and banks, pkmf's Linear init (plan and log)

**Status:** PLAN, 2026-11-08 (Maciej). Not started; it starts on Maciej's /goal.
Background: [pkmf_init_comparison.md](pkmf_init_comparison.md) and
[ame_implementation.md](ame_implementation.md) (weight initialization).

**Question:** do these four init changes, made the default for every model, lower final CE
for AME and dense at h12/4B and h16/8B?

## What changes

Branch `ame_initialization` from `ame_main`, in a new worktree `~/code/nano-ame-init`. It
is not merged into `ame_main`. The new defaults apply to every model, since every config
builds from [configs/_model/h_balanced/base.yaml](../../configs/_model/h_balanced/base.yaml).

| Weights | Now (`ame_main`) | New default | σ at h12 / h16 |
|---|---|---|---|
| Token embedding `[V, d]` | N(0, 1) cut at ±2 (std 0.88); weight decay 0.1 | Xavier normal, σ = √(2/(V + d)), V = 50,304; **no weight decay** | 0.00626 / 0.00624 |
| Head `[V, d]` | truncated normal, σ = 1/√d, cut at ±2σ | Xavier normal, σ = √(2/(V + d)); weight decay stays 0.1 | 0.00626 / 0.00624 |
| AME banks `q_bank`, `k_bank` `[V, d]` | N(0, 1) (`bank_init_std: 1.0`); no weight decay | Xavier normal, σ = √(2/(V + d)); still no weight decay | 0.00626 / 0.00624 |
| Attention and FF `Linear` weights | truncated normal, σ = 1/√fan_in, cut at ±2σ | pkmf's σ = √(0.1/fan_in), cut kept at ±2σ | 0.0114 / 0.0099 at fan_in = d; 0.0057 / 0.0049 for FF out |

Everything else stays as it is: norm gains, AME scalars, ALiBi, sink, KV shift, schedule,
and weight decay 0.1 on all other weights.

**Where:**
- [base.yaml](../../configs/_model/h_balanced/base.yaml):
  - embedding `init_fn` → `torch.nn.init.xavier_normal_`;
  - head `partial_init_fn` → a Xavier helper in [model.py](../../src/core/model.py);
  - FF `scale` and attention `init_scale` → √0.1, so `llm_random_weight_init` gives
    σ = √(0.1/fan_in), cut at ±2σ.
- [model.py](../../src/core/model.py): `TransformerEmbedding` names its `weight` in
  `no_weight_decay_parameters`; `parameter_groups` already reads that attribute.
- [ame.py](../../src/core/ame.py): the banks default to Xavier. An explicit
  `bank_init_std` number still draws N(0, σ²), so the stage 1-2 configs keep their meaning.
  Update the typed schema.
- Xavier takes its fans from the weight's shape, so `xavier_normal_` gives √(2/(V + d)) for
  all three `[V, d]` tables.
- **Memory-parallel banks (Tier 2):** check that the fp32 masters hold the Xavier draw.

## Local CPU tests

- The full pytest suite passes. Update any test that pins the old init. Baseline on
  `ame_main`: 393 passed, 18 skipped.
- A new init test, on tiny dense and AME models:
  - the embedding, head and banks have std ≈ √(2/(V + d)), untruncated;
  - each `Linear` has std ≈ 0.88·√(0.1/fan_in), with max |w| ≤ 2σ;
  - the embedding and banks sit in the weight-decay-0 group, and the head and `Linear`
    weights at 0.1;
  - the memory-parallel masters equal the module's draw.
- A CPU smoke of a few steps, on a tiny AME model (KV shift, ALiBi) and a tiny dense one:
  - the loss stays finite;
  - the first loss is ≈ ln V = 10.83, because the logits start at std 0.17-0.20.
- Compose the four run configs and print their init values.

## Runs on LEM

- **Configs:** `configs/ame_init/<run>.yaml`. Each copies its reference's config and
  changes only the names; the init comes from the branch defaults.
- **W&B:** project `nano-ame`, run name `ame_init_<run>`.
- **Seed:** 123, the references' data order.

| Run | Recipe: config copied | Tokens, steps | LR n | Limit | Reference, old init: W&B, final_ce, s/step |
|---|---|---|---|---|---|
| `ame_h12` | S2-PS, AME with KV shift and ALiBi: [bank_gates/base.yaml](../../configs/bank_gates/base.yaml) | 4B, 3,815 | 8 | 7 h | `doi36p8k`, 3.1250, 3.746 |
| `ame_h16` | S2b-PS h16: [bank_gates/h16_base.yaml](../../configs/bank_gates/h16_base.yaml) | 8B, 7,630 | 8 | 24 h | `hjw7znbz`, 2.8594, 6.810 |
| `dense_h12` | S1-D, GQA with RoPE: [stage_1/dense_h12_lr_grid.yaml](../../configs/stage_1/dense_h12_lr_grid.yaml), the grid replaced by `learning_rate: 8` | 4B, 3,815 | 8 | 5 h | `x0xlqroj`, 2.9824, 3.272 |
| `dense_h16` | S1b-D: [stage_1b/dense_h16_lr_grid.yaml](../../configs/stage_1b/dense_h16_lr_grid.yaml), `learning_rate: 9` | 8B, 7,630 | 9 | 24 h | `eh2sjsh7`, 2.7441, 5.964 |

**Submitting:**
- Commit everything on `ame_initialization` first: run_exp snapshots the tree, untracked
  files included.
- Submit from `~/code/nano-ame-init`, one `run_exp.py` at a time:

```
run_exp.py --config-path configs/ame_init --config-name <run> +infrastructure.slurm.exclude=r16-11 < /dev/null
```

- **Expected durations,** as the references took: about 4.3 h, 15 h, 3.5 h and 12.6 h.

**Checks:**
- **About 2 minutes after each start:** the job is running, the first loss is ≈ 10.83, and
  s/step is within 10% of the reference's.
- **At step 500 and mid-run:**
  - no NaN;
  - loss spikes and grad norm. The Xavier embedding sits in the small-embedding range
    where Takase et al. saw spikes;
  - train loss against the reference's at the same steps.
- **Never cancel a job without asking.**

## How to analyze

- **Each run against its reference:**
  - final_ce;
  - the paired train-loss gap (new − reference at the same step), for each part of
    training and for the last 500 steps, as in
    [ame_improve_runs_report.md](ame_improve_runs_report.md).
- **The paired gap decides,** since final_ce is bf16-quantized (±0.008 per run).
  - AME ties inside ±0.022.
  - Dense has no replicate floor, so report its gap without calling small ones.
- **The AME − dense gap under the new init,** against the old: 0.143 at h12
  (3.1250 − 2.9824) and 0.115 at h16 (2.8594 − 2.7441).
- **Health:** spikes, the grad-norm max after step 100, s/step, peak memory.

## Log

### Implementation

2026-11-08, branch `ame_initialization` from `ame_main` `33b2f059`, worktree
`~/code/nano-ame-init` (`data` symlinked from the main checkout; gitignored). The
code, tests, run configs and script are commit `79108114`.

- [base.yaml](../../configs/_model/h_balanced/base.yaml):
  - embedding `init_fn` is `torch.nn.init.xavier_normal_`;
  - head `partial_init_fn` is the new `src.core.model.xavier_normal_weight_init`,
    which returns `xavier_normal_` (torch reads both fans off the weight's shape);
  - attention `init_scale` and both FF `scale`s are `${eval:'0.1 ** 0.5'}`.
- [model.py](../../src/core/model.py): `TransformerEmbedding.no_weight_decay_parameters
  = ("weight",)`.
- [ame.py](../../src/core/ame.py):
  - `bank_init_std` defaults to `None`, which draws `xavier_normal_` on `[V, H·d]`;
  - a number still draws N(0, σ²), and a value ≤ 0 is refused;
  - header note 3 is rewritten. It said rows must start at σ ≈ 1. With QK-norm, σ
    only sets how fast a row turns; without QK-norm, pass `bank_init_std: 1.0`.
- **Typed schema:** no change needed. `common.attention.ame` is untyped (`Any`) in
  [definitions.py](../../src/definitions.py), and `build_attention` checks the option
  names; `bank_init_std` is already one, and `null` passes.
- **Memory-parallel masters:** no change needed. `ShardedBank.load_from` copies the
  drawn bank into the fp32 master, so the master holds the Xavier draw (tested).
- **Run configs:** [configs/ame_init/](../../configs/ame_init/). The AME ones copy
  `bank_gates/base.yaml` and `h16_base.yaml`. The dense ones copy the grid files with
  `learning_rate` fixed at 8 and 9, and stay without `/base_schema` like their sources;
  `run_exp.py` schema-checks them anyway. Each reference's logged W&B config matches
  its copy in seed (123), n, steps, warmup, micro-batch, attention and banks.
- **Analysis script:** [scripts/paired_gap.py](../../scripts/paired_gap.py), on the
  branch. Checked on a known pair: `y5ogvt3k` against `doi36p8k` gives Δ final_ce
  −0.0312 and a last-500 gap of −0.0342, as
  [ame_improve_runs_report.md](ame_improve_runs_report.md) has.
- **Corrections to the plan:**
  - The plan says an explicit `bank_init_std` keeps the stage 1-2 configs' meaning.
    It doesn't: only `bank_gates/*` set it. On this branch the stage 1-2 AME configs
    get Xavier banks too, which is the "default for every model" asked for;
    `ame_main` is untouched.
  - The code-level `init_scale=1` defaults in `build_attention` and `AMEAttention`
    stay. Every model config passes `init_scale` from `base.yaml`; only layers built
    directly in unit tests see the code default.
- **Pairing holds:** the new draws use the global torch RNG differently, but the
  data order comes only from the dataset's own `random.Random(seed)` and HF's seeded
  shuffle. So each step sees the same batch as in the reference.
- **Not resumable:** old dense checkpoints can't resume their optimizer state on this
  branch, since one group became two. No run here resumes.

### Local tests

- **Full suite** (`-m "not gpu"`, the repo's default; this machine has no CUDA):
  _running_. A first run without `-m` also ran the 13 `gpu` tests. They call
  `.cuda()`, so they fail on any machine without CUDA ("No CUDA GPUs are
  available"), on any branch. That run's other failures were caused by the change
  and are fixed in `79108114`:
  - **Two `test_ame` grouping tests pinned the old rule:** the embedding decays, and
    a dense model gets one optimizer group. They now check the new rule: the
    embedding alone is undecayed in a dense model, and the head still decays. The
    one-group guarantee is now tested on a bare stack of `Linear`s.
  - **`test_memory_parallel`:** the loss trajectories at `mem_parallel_size` 1 and 2
    split by 5e-11 relative at step 2, against a 1e-12 tolerance. The toy's AME now
    gets Xavier banks (σ = 0.079 at V = 256, d = 64), so Adam turns a row by about
    LR/σ = 0.13 rad per step instead of 0.01, and float64 rounding grows faster.
    Sharding is still exact: the 1e-14 gradient check passes under Xavier. The toy
    now pins `bank_init_std: 1.0`, the regime the tolerance was set in, and the file
    passes (28 tests).
- **New test,** [tests/test_init_defaults.py](../../tests/test_init_defaults.py): 12 pass,
  on tiny h4 models (V = 4,096) composed from `base.yaml`, dense and AME. It checks:
  - the embedding, head and banks have std √(2/(V + d)) to 1%, uncut;
  - every attention and FF `Linear` has std 0.88·√(0.1/fan_in), with max |w| ≤ 2σ;
  - the embedding and banks are in the weight-decay-0 group, the head and `Linear`s
    at 0.1;
  - an explicit `bank_init_std: 1.0` still draws N(0, 1), and 0 or a negative is
    refused;
  - the Tier 2 fp32 masters equal the module's draw, and the bf16 copy is its cast;
  - the composed h12 config carries the new targets and scales.
- **CPU smoke through `main.py`** (`bank_gates/smoke_local`, h4, V = 50,304, 4 steps,
  `bank_init_std=null`):

  | Model | Loss, step 0 | Loss, step 3 | final_ce | End of run |
  |---|---:|---:|---:|---|
  | AME, KV shift + ALiBi, Tier 2 banks | 10.821 | 10.053 | 9.879 | checkpoint, bank state and `token_counts.pt` written |
  | Dense GQA + RoPE | 10.827 | 10.060 | 9.789 | checkpoint written |

  ln V = 10.826. At h4 the logits start at std ≈ 0.10.
- **Logits at init, real sizes** (CPU forward, random ids): std 0.173 at h12 and 0.200
  at h16. Cross-entropy against random targets is 10.841 and 10.850.
- **The four configs, composed as `run_exp.py` does** (missing keys, grid,
  `check_point`) and built on meta. All of them get Xavier embedding and head, and
  scale 0.316228 on the attention and FF `Linear`s.

  | Run | Parameters | Weight decay 0.1 | Weight decay 0 | n, steps, warmup | Micro-batch |
  |---|---:|---:|---:|---|---|
  | `ame_h12` | 1,103,599,056 (as stage 2) | 137,743,104 | 965,855,952 (embedding + banks + scalars) | 8, 3,815, 38 | 32 × 16 |
  | `ame_h16` | 1,986,332,928 | 286,426,112 | 1,699,906,816 | 8, 7,630, 76 | 16 × 32 |
  | `dense_h12` | 183,454,464 | 144,820,992 | 38,633,472 (= V·d, the embedding) | 8, 3,815, 38 | 32 × 16 |
  | `dense_h16` | 354,714,624 | 303,203,328 | 51,511,296 (= V·d) | 9, 7,630, 76 | 16 × 32 |

### Runs

| Run | Job | W&B | Node | Submitted | Finished | State |
|---|---|---|---|---|---|---|
| `ame_h12` | | | | | | |
| `ame_h16` | | | | | | |
| `dense_h12` | | | | | | |
| `dense_h16` | | | | | | |

### Checks

### Results

| Run | final_ce | Reference | Δ final_ce | Paired train-loss gap, last 500 steps | Verdict |
|---|---:|---:|---:|---:|---|
| `ame_h12` | | 3.1250 | | | |
| `ame_h16` | | 2.8594 | | | |
| `dense_h12` | | 2.9824 | | | |
| `dense_h16` | | 2.7441 | | | |

AME − dense gap: h12 old 0.143, new —; h16 old 0.115, new —.

### Decisions and issues
