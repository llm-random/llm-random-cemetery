#!/bin/bash
python3 -m lizrd.grid --config_path=configs/private/diff_attn/2025.02.13_float32_lowrank.yaml --git_branch=debug_diff_2025-02-13_14-32-56 --skip_copy_code --custom_backends_module=research.attention_moe.backends