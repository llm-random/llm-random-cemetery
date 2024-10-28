#!/bin/bash

which python
which python3
conda deactivate
which python
which python3
module load ML-bundle/24.06a
which python
which python3
source /net/storage/pr3/plgrid/plggllmeffi/datasets/make_singularity_image/venv/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@