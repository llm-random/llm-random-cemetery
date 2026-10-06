#!/bin/bash
# S5: short LR sweep (100 steps, warmup 40) of plain recovery training from the r50 smoke checkpoint, 2 GPUs.
OUT=$HOME/slicegpt_dev/results; mkdir -p $OUT
for n in 15 14 13 12 11; do
  echo "===== lr 2^-$n ====="
  torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:$((29500 + RANDOM % 1000)) \
    main.py --config-path=smoke_configs --config-name=s5_lr$n.yaml > $OUT/s5_lr$n.log 2>&1
  echo "lr$n exit $?"
done
