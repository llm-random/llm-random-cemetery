#!/bin/bash

# module CUDA/12.3.0
ldconfig /usr/lib64-nvidia
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@