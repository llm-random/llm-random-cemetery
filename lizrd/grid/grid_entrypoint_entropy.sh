#!/bin/bash

module CUDA/11.7.0
source /home/ludziej_a100/ms/venv/bin/python
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@