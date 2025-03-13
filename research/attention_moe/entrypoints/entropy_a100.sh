#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy A100 Entrypoint"
conda activate /storage_ssd_1/llm-random/differential-conda
echo "Will run the following command:"
echo "$@"
echo "==============================="
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
$@