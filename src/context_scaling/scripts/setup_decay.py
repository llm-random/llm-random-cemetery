import os
import copy
import warnings
from pathlib import Path
import argparse
import json
import yaml

from setup_eval import (
    get_wandb_table,
    save_yaml_config_from_row,
    update_slurm_array_line,
)


def find_checkpoint_steps(ckpt_path: str) -> list[int]:
    """List all step_* checkpoint directories, sorted ascending."""
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        warnings.warn(f"Checkpoint directory not found: {ckpt_dir}")
        return []

    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_"):
            try:
                steps.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    steps.sort()
    return steps


def build_decay_config(
    base_config: dict,
    ckpt_load_path: str,
    decay_steps: int,
    save_base_path: str,
) -> dict:
    """Modify a training config for a pure-decay run from a checkpoint."""
    cfg = copy.deepcopy(base_config)

    # Override training duration to just the decay phase
    cfg["trainer"]["n_steps"] = decay_steps

    # WSD scheduler: pure decay (no warmup, no constant phase)
    cfg["trainer"]["scheduler"] = {
        "_partial_": True,
        "_target_": "src.core.schedulers.WSDScheduler",
        "n_steps": decay_steps,
        "warmup_steps": 0,
        "decay_steps": decay_steps,
    }

    # Load full checkpoint (model + optimizer), but don't load training state
    cfg["trainer"]["checkpoint"] = {
        "load": {
            "type": "nano",
            "path": ckpt_load_path,
            "model_checkpoint_filename": "__model_checkpoint_filename.pt",
            "training_state_filename": None,
            "only_weights": False,
            "rewind_data": True,
        },
        "save": {
            "type": "nano",
            "interval": -1,
            "path": save_base_path,
            "model_checkpoint_filename": "__model_checkpoint_filename.pt",
            "training_state_filename": "__training_state_filename.pt",
        },
    }

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate decay-phase training jobs from intermediate checkpoints."
    )
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--negative_tags", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default="decay_grid")
    parser.add_argument(
        "--decay_fraction",
        type=float,
        default=0.1,
        help="Fraction of original training steps to use for decay (default: 0.1).",
    )
    parser.add_argument(
        "--save_ckpt_base",
        type=str,
        default=None,
        help="Base path for saving decay run checkpoints. If not set, decay runs won't save checkpoints.",
    )
    parser.add_argument(
        "--sbatch_path",
        type=str,
        default=None,
        help="Path to an sbatch script. Its '#SBATCH --array=...' line will be updated in-place.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Specific checkpoint steps to decay from. If not set, uses all checkpoints except the last.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    config_dir = out_dir / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_dir = out_dir / "yaml_cache"

    df = get_wandb_table(tags=args.tags, negative_tags=args.negative_tags)
    if df.empty:
        print("No runs found. Exiting.")
        return

    global_idx = 0
    records = []

    for _, row in df.iterrows():
        run_id = str(row["sys/id"])
        run_name = str(row.get("sys/name", run_id))
        ckpt_path = str(row.get("summary/full_save_checkpoints_path", ""))

        if not ckpt_path:
            warnings.warn(f"Run {run_id} ({run_name}): no checkpoint path, skipping.")
            continue

        # Reconstruct original config from wandb
        yaml_path = yaml_dir / f"{run_id}.yaml"
        if not yaml_path.exists() or yaml_path.stat().st_size == 0:
            save_yaml_config_from_row(row, yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        original_n_steps = base_config["trainer"]["n_steps"]
        decay_steps = int(original_n_steps * args.decay_fraction)

        # Determine which checkpoint steps to decay from
        if args.steps is not None:
            steps_to_decay = sorted(args.steps)
        else:
            all_steps = find_checkpoint_steps(ckpt_path)
            if len(all_steps) < 2:
                warnings.warn(
                    f"Run {run_id} ({run_name}): need at least 2 checkpoints, found {len(all_steps)}. Skipping."
                )
                continue
            steps_to_decay = all_steps[:-1]

        print(
            f"Run {run_id} ({run_name}): {len(steps_to_decay)} decay jobs "
            f"(steps {steps_to_decay}, decay_steps={decay_steps})"
        )

        for step in steps_to_decay:
            step_ckpt_path = f"{ckpt_path}/step_{step}"

            save_path = None
            if args.save_ckpt_base:
                save_path = f"{args.save_ckpt_base}/{run_id}/from_step_{step}"

            decay_cfg = build_decay_config(
                base_config=base_config,
                ckpt_load_path=step_ckpt_path,
                decay_steps=decay_steps,
                save_base_path=save_path,
            )

            config_path = config_dir / f"config_{global_idx}.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(decay_cfg, f, sort_keys=True)

            records.append(
                {
                    "array_idx": global_idx,
                    "run_id": run_id,
                    "run_name": run_name,
                    "source_step": step,
                    "decay_steps": decay_steps,
                    "ckpt_load_path": step_ckpt_path,
                    "config_path": str(config_path),
                }
            )
            global_idx += 1

    # Write jobs metadata
    jobs_path = out_dir / "jobs.json"
    with open(jobs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n{len(records)} total decay jobs")
    print(f"  configs: {config_dir}")
    print(f"  jobs metadata: {jobs_path}")

    # Update sbatch array size
    if args.sbatch_path is not None and records:
        update_slurm_array_line(Path(args.sbatch_path), num_jobs=len(records))


if __name__ == "__main__":
    main()

python src/context_scaling/scripts/run_decay.py \
    --tags test_decay \
    --out_dir test_decay_grid \
    --steps 25 50 75 \
    --decay_fraction 0.1 \
    --save_ckpt_base /storage_nvme_4/nano/models/decay_test \
    --job_name test_decay_submit \
    --submit