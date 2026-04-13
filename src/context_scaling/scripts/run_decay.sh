#!/bin/bash -l
set -euo pipefail

# Smoke test for run_decay.py against base runs produced by configs/test_decay.yaml
# (tiny model, 1001 steps, checkpoints every 200 steps).
#
# Prerequisites:
#   1. Run a base training job with: pixi run python run_exp.py --config configs/test_decay.yaml
#   2. The base run must be logged to wandb with the "test_decay" tag.
#
# This script generates decay configs locally (dry run, no --submit).

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay_local \
    --negative_tags decay \
    --out_dir test_decay_grid \
    --steps 200 600 \
    --decay_fraction 0.1 \
    --train_data_seed 999 \
    --eval_config configs/_eval/test_tasks.yaml \
    --job_name test_decay
