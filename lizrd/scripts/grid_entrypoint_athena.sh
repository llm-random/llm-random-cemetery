#!/bin/bash

ldconfig /.singularity.d/libs
module CUDA/11.7.0
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@