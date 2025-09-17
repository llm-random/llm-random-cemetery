#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
echo "Running Entropy H100 Entrypoint"
conda activate llm_random_main
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@