#!/bin/bash
# S4 / V5: 30-step recovery-training smoke on 2 GPUs (FSDP2) from the S2 sliced checkpoint, plain and distillation.
OUT=$HOME/slicegpt_dev/results; mkdir -p $OUT
for cfg in s4_rft s4_rft_distil; do
  echo "===== $cfg ====="
  torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:$((29500 + RANDOM % 1000)) \
    main.py --config-path=smoke_configs --config-name=$cfg.yaml > $OUT/$cfg.log 2>&1
  echo "$cfg exit $?"
done
