#!/bin/bash

module CUDA/12.7.0
echo "Will run the following command:"
echo "$@"
echo "==============================="
$@