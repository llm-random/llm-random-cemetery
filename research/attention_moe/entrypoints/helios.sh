#!/bin/bash -l

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggllmeffi/momqa/venv/bin/activate
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@