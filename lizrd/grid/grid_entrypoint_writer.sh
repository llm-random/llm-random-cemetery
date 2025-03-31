#!/bin/bash -l

module load ML-bundle/24.06a
conda activate flop_count_env
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@