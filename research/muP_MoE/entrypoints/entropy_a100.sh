#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy A100 Entrypoint"
conda activate llm-random_main
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@