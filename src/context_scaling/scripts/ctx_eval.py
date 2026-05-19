"""Hydra entrypoint for context-scaling eval.

Auto-dispatches on SLURM_ARRAY_TASK_ID:
  unset → setup (head node): fetch wandb runs, write jobs.json, generate ctx_eval.job
  set   → per-task eval (compute node): load ckpt, eval, save CSV, upload mean to wandb

The generated sbatch (ctx_eval.job) is built from cfg.infrastructure.slurm +
cfg.infrastructure.script, mirroring how run_exp.py builds exp.job.
"""
import json
import os
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf

from grid_generator.sbatch_builder import create_slurm_parameters
from setup_eval import resolve_model_step
from wandb_utils import (
    WANDB_PROJECT,
    get_wandb_table,
    save_yaml_config_from_row,
    upload_mean_loss_to_wandb,
)

SBATCH_OUT = "ctx_eval.job"


def _build_sbatch(cfg, n_jobs: int) -> str:
    lines = ["#!/bin/bash -l", ""]
    lines.append(f"#SBATCH --array=0-{n_jobs - 1}")
    lines.append("#SBATCH --requeue")
    lines.extend(create_slurm_parameters(cfg.infrastructure.slurm))

    script = cfg.infrastructure.get("script")
    if script:
        # skip placeholder secret exports; the job inherits real values via --export=ALL
        script_lines = [
            ln for ln in OmegaConf.to_container(script, resolve=True)
            if "PLACEHOLDER" not in ln
        ]
        lines.extend(["", "#---------- SCRIPT ----------"])
        lines.extend(script_lines)
        lines.extend(["#-------- SCRIPT END --------", ""])

    # single-GPU eval: no torchrun rendezvous, just one srun per array task
    lines.extend(
        [
            'export PYTHONPATH="$(pwd):${PYTHONPATH:-}"',
            "export TORCH_COMPILE_DISABLE=1",
            "export MASTER_ADDR=127.0.0.1",
            "export MASTER_PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))",
            "export RANK=0",
            "export WORLD_SIZE=1",
            "export LOCAL_RANK=0",
            "",
            "srun --export=ALL --ntasks=1 --nodes=1 --gpus=1 "
            "python src/context_scaling/scripts/ctx_eval.py",
        ]
    )
    return "\n".join(lines) + "\n"


def _setup(cfg):
    eval_cfg = cfg.eval
    out_dir = Path(eval_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    negative_tags = list(eval_cfg.negative_tags) if eval_cfg.negative_tags else None
    df = get_wandb_table(
        tags=list(eval_cfg.tags),
        project=WANDB_PROJECT,
        negative_tags=negative_tags,
    )
    df.to_csv(out_dir / "main.csv", index=False)

    yaml_dir = out_dir / "yaml_cache"
    records = []
    for _, row in df.iterrows():
        run_id = str(row["sys/id"])
        ckpt_path = str(row.get("summary/full_save_checkpoints_path", ""))

        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            save_yaml_config_from_row(row, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        seq_len = run_cfg["common"]["sequence_length"]
        if eval_cfg.seq_len is not None and eval_cfg.seq_len < seq_len:
            seq_len = int(eval_cfg.seq_len)

        model_step = resolve_model_step(
            ckpt_path,
            int(eval_cfg.model_step) if eval_cfg.model_step is not None else None,
        )

        records.append(
            {
                "jobID": run_id,
                "ckpt_path": ckpt_path,
                "yaml_config_path": str(yaml_path),
                "seq_len": seq_len,
                "model_step": model_step,
                "wandb_project": WANDB_PROJECT,
            }
        )

    jobs_path = out_dir / "jobs.json"
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(records)} jobs → {jobs_path}")

    Path(SBATCH_OUT).write_text(_build_sbatch(cfg, len(records)))
    print(f"wrote sbatch → {SBATCH_OUT} (array=0-{len(records) - 1})")


def _run_task(cfg):
    # heavy imports deferred so head-node setup stays fast
    from functools import partial

    import pandas as pd
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from datasets import load_from_disk
    from hydra.utils import instantiate
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from eval_models import (
        ModelOnly,
        append_to_index,
        batch_per_token_losses,
        collate_no_pad,
        load_cfg_from_yaml,
        setup_distributed,
    )

    eval_cfg = cfg.eval
    out_dir = Path(eval_cfg.out_dir)
    device = setup_distributed()

    with open(out_dir / "jobs.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id >= len(jobs):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} out of range (0..{len(jobs) - 1})"
        )

    job = jobs[task_id]
    run_id = job["jobID"]
    ckpt_dir = job["ckpt_path"]
    yaml_path = Path(job["yaml_config_path"])
    seq_len = job["seq_len"]
    model_step = job["model_step"]
    wandb_project = job.get("wandb_project", WANDB_PROJECT)

    print(f"task_id={task_id} run_id={run_id} step={model_step} seq_len={seq_len}")
    run_cfg = load_cfg_from_yaml(yaml_path)
    out_csv = out_dir / f"{run_id}_step_{model_step}.csv"

    try:
        model = instantiate(run_cfg.model, _convert_="all").to(device)
        model.eval()
        fsdp_model = FSDP(model)
        state = {"app": ModelOnly(fsdp_model)}
        ckpt_full = os.path.join(ckpt_dir, f"step_{model_step}")
        dcp.load(state, checkpoint_id=ckpt_full)
        fsdp_model.eval()

        ds = load_from_disk(eval_cfg.dataset_dir)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        collate_fn = partial(collate_no_pad, tokenizer=tokenizer, seq_len=seq_len)
        loader = DataLoader(
            ds, batch_size=eval_cfg.batch_size, shuffle=False, collate_fn=collate_fn
        )

        all_losses = []
        for batch in tqdm(loader):
            if batch is None:
                continue
            with torch.no_grad():
                losses, _ = batch_per_token_losses(
                    fsdp_model, batch["input_ids"], device
                )
            all_losses.append(losses)

        stacked = torch.cat([t.detach().cpu() for t in all_losses], dim=0)
        print(f"saving CSV to {out_csv}")
        pd.DataFrame(stacked.numpy()).to_csv(out_csv, index=False)

        upload_mean_loss_to_wandb(
            run_id=run_id,
            project=wandb_project,
            mean_losses=stacked.mean(dim=0),
            model_step=model_step,
            eval_seq_len=seq_len,
        )
        append_to_index(
            str(out_dir), out_csv.name, run_id, model_step, run_cfg, seq_len
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../../../configs", config_name="ctx_eval")
def main(cfg):
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        _run_task(cfg)
    else:
        _setup(cfg)


if __name__ == "__main__":
    main()
