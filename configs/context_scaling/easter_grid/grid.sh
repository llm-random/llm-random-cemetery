#!/bin/bash

DIR="context_scaling/easter_grid"

# k24
python run_exp.py --config-path configs --config-name $DIR/k24

python run_exp.py --config-path configs --config-name $DIR/k24 \
  common.kv_heads=1 common.dff=6144

# k28
python run_exp.py --config-path configs --config-name $DIR/k28

python run_exp.py --config-path configs --config-name $DIR/k28 \
  common.kv_heads=1 common.dff=7168
