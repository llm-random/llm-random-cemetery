#!/bin/bash -l

export OMP_NUM_THREADS=8
module load ML-bundle/24.06a
export OMP_NUM_THREADS=8
source /net/storage/pr3/plgrid/plggllmeffi/datasets/make_singularity_image/venv/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@