#!/bin/bash

module CUDA/12.3.0
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@