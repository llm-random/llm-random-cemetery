#!/bin/bash -l
set -euo pipefail

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay \
    --negative_tags decay \
    --out_dir test_decay_grid \
    --steps 100 200 300 400 500 600 700 800 900 1000 \
    --decay_fraction 0.1 \
    --save_ckpt_base /storage_nvme_4/nano/models/decay_test \
    --job_name test_decay_submit \
    --submit
