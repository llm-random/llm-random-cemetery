#!/bin/bash -l
set -euo pipefail

pixi run python src/context_scaling/scripts/run_decay.py \
    --tags test_decay \
    --out_dir test_decay_grid \
    --steps 25 50 75 \
    --decay_fraction 0.1 \
    --save_ckpt_base /storage_nvme_4/nano/models/decay_test \
    --job_name test_decay_submit \
    --submit
