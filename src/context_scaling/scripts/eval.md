# Context-scaling eval

Two ways to run the same thing:

- **Main (automated)**: one local command does everything — push code, ssh, build jobs, generate sbatch, submit, tail log.
- **Legacy (manual)**: original 4-script flow if you want to control each step on the cluster yourself.

---

## Main (automated)

```
python src/context_scaling/scripts/run_ctx_eval.py
```

All knobs live in `configs/ctx_eval.yaml` — `eval.tags`, `eval.out_dir`, `eval.dataset_dir`, `eval.seq_len`, `eval.model_step`, `eval.batch_size`, `infrastructure.server`, `infrastructure.slurm.*`. Edit the yaml (or pass Hydra CLI overrides) and re-run.

### What happens

1. `run_ctx_eval.py` (local) calls `version_code` from `run_exp.py`: commits your working tree onto a temp branch `ctxeval_<name>_<ts>`, pushes it to the cemetery remote, restores your local state.
2. Ssh's to `cfg.infrastructure.server`, clones the branch into `cfg.infrastructure.cemetery_experiments_dir/<branch>`, opens a tmux session.
3. In tmux: exports your local `WANDB_API_KEY` + `HF_TOKEN`, sources `cfg.infrastructure.script` (pixi activation etc.), exports `PYTHONPATH`.
4. Runs `python src/context_scaling/scripts/ctx_eval.py`. With `SLURM_ARRAY_TASK_ID` unset, this triggers **setup mode**:
   - Fetches wandb runs matching `eval.tags` / `eval.negative_tags` (project from `wandb_utils.WANDB_PROJECT`).
   - Writes `{out_dir}/jobs.json` and per-run `yaml_cache/<run_id>.yaml`.
   - Generates `ctx_eval.job` — an sbatch built from `cfg.infrastructure.slurm` + `cfg.infrastructure.script`, with `--array=0-(len(jobs)-1)`.
5. `sbatch ctx_eval.job` submits the array. Each task re-invokes `ctx_eval.py`, this time with `SLURM_ARRAY_TASK_ID` set → **run mode**:
   - Loads the checkpoint via FSDP + `torch.distributed.checkpoint`.
   - Runs the per-token loss loop on `eval.dataset_dir`.
   - Saves CSV at `{out_dir}/{run_id}_step_{step}.csv`.
   - Resumes the corresponding wandb run and writes the batch-mean loss as a list under `summary["eval/per_position_loss/step{N}_seq{N}"]`.
   - Appends a row to `{out_dir}/index.jsonl` with the flat config.
6. Driver captures the SLURM job ID from the tmux pane and pipes `tail -f slurm-<id>_0.out` into the same pane.

To attach to that pane: `ssh <cluster> -t tmux attach -t <branch>` (or launch with `LOGLEVEL=DEBUG` to attach immediately).

### Files

- `configs/ctx_eval.yaml` — knobs
- `src/context_scaling/scripts/run_ctx_eval.py` — local driver
- `src/context_scaling/scripts/ctx_eval.py` — Hydra entrypoint (setup + run modes)
- `src/context_scaling/scripts/wandb_utils.py` — `WANDB_PROJECT`, `get_wandb_table`, `upload_mean_loss_to_wandb`

---

## Legacy (manual)

`src/context_scaling/scripts/setup_eval.sh` \
which handles \
`src/context_scaling/scripts/setup_eval.py` \
and \
`src/context_scaling/scripts/eval_models.sbatch` \
which handles \
`src/context_scaling/scripts/eval_models.py`

### How to use them

1. setup `src/context_scaling/scripts/setup_eval.sh`
    1. create unique set of neptune tags for grid you want to eval (WARNNG: all runs need to have same number of steps)
    2. update `--tags` and `--out_dir` in `src/context_scaling/scripts/setup_eval.sh`
    3. optionally pass `--model_step` to pin a specific checkpoint step (otherwise uses latest step_* per run)
    4. this script creates a jobs_json, a list[{"jobID","ckpt_path","yaml_config_path","seq_len","model_step"}] for each run. If you rsynced model checkpoints update ckpt_path in the json.
2. setup `src/context_scaling/scripts/eval_models.sbatch`
    1. make sure that `--jobs_json` points to the json created by setup script (model_step is now in jobs.json, `--model_step` on sbatch overrides it)
3. run eval
    1. commit push changes to github
    2. ssh to cluster
    3. git pull
    4. run `bash -l src/context_scaling/scripts/setup_eval.sh` (it modifies number of jobs in slurm array in `eval_models.sbatch`)
    5. run `sbatch src/context_scaling/scripts/eval_models.sbatch`

Note: the legacy flow does **not** upload per-position loss to wandb — that's a feature of the automated `ctx_eval.py` only. The legacy flow just writes the per-token loss CSV.

---

## Loading results in notebooks

Both flows append to `{out_dir}/index.jsonl` with the full flat config. Use the index to filter on any config field:

```python
from src.context_scaling.eval_index import load_eval_index, load_eval_csvs

idx = load_eval_index("path/to/eval_dir")

# filter on any config field with normal pandas
subset = idx[idx["common.kv_heads"] == 1]
subset = subset[subset["common.sequence_length"] == 2048]

# load CSVs with readable labels
labels, dfs = load_eval_csvs(subset, "path/to/eval_dir",
                             label_cols=["common.kv_heads", "common.dmodel"])
```
