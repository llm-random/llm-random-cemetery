#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy A100 Entrypoint"
conda activate llm-random-310
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@