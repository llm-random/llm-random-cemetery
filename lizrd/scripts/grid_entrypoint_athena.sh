#!/bin/bash

# module load CUDA/12.0.0
ldconfig /usr/lib64-nvidia
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@