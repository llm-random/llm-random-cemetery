#!/bin/bash -l
set -euo pipefail

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags easter_grid torch \
    --negative_tags decay \
    --out_dir easter_grid_decay \
    --steps 32000 \
    --decay_fraction 0.1 \
    --train_data_seed 458 \
    --eval_config configs/_eval/test_tasks.yaml \
    --job_name easter_decay \
    --max_concurrent_jobs 40 \
    --slurm_time "12:00:00" \
    --submit
