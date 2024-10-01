import itertools
import sys
from typing import List, Optional, Tuple

import yaml

from lizrd.grid.prepare_configs import get_yaml_md5

OUTPUT_FILE = "configs/experiments/grad_norm/std_norm_grid/post_add_c_lr_grid_reduced_bn_long.yaml"
BASELINE_INPUT = "configs/experiments/grad_norm/medium_reduced_bs.yaml"


GRAD_MODIF_PLACEMENT_COMBINATIONS: List[Tuple[List[str], Optional[str]]] = [
    (["post_attn", "post_ff"], "post_attn_and_ff"),
    (["post_attn_norm", "post_ff_norm"], "post_norm"),
    (["post_attn_add", "post_ff_add"], "post_add"),
]

STD_NORM_MODIF_PARAMS_1: List[Tuple[List[str], Optional[str]]] = [
    (["layer_type=v1", "c=0.2", "eps=1e-6"], "layer_type_v1"),
    (["layer_type=v2", "c=0.2", "eps=1e-6"], "layer_type_v2"),
]

BASELINE_GRAD_MODIF_PLACEMENT = (
    ["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"],
    "all",
)

LR_MULTIPLIERS: List[Tuple[float, str]] = [
    (1 / 100, "div100"),
    (1 / 10, "div10"),
    (1, "mul1"),
    (10, "mul10"),
    (100, "mul100"),
]

C_GRID: List[Tuple[float, str]] = [
    (0, "c_0"),
    (1e-2, "c_1e-2"),
    (1e-3, "c_1e-3"),
    (1e-4, "c_1e-4"),
    (1e-5, "c_1e-5"),
]

NAME_PREFIX = "post_add_c_lr_grid"


def main():
    parent_md5_hash = get_yaml_md5(BASELINE_INPUT)
    configs = []

    for i, ((lr_multiplier, lr_tag), (c, c_tag)) in enumerate(itertools.product(LR_MULTIPLIERS, C_GRID)):
        config_name = f"exp_{i}_lr_{lr_tag}_{c_tag}"

        config = {
            "parent": BASELINE_INPUT,
            "time": "0-12:00:00",
            "md5_parent_hash": parent_md5_hash,
            "params": {
                "grad_modif_placement": ["post_attn_add", "post_ff_add"],
                "grad_modif_params": ["layer_type=v1", f"c={c}", "eps=0"],
                "tags": [f"lr_{lr_tag}", "std_norm", "post_add_c_lr_grid_long", "grad_norm", c_tag, "post_add"],
                "name": f"{NAME_PREFIX}_{config_name}",
                "grad_modif_type": "std_norm",
                "learning_rate": 1e-4 * lr_multiplier,
                "n_steps": 16_000,
            },
        }

        configs.append(config)
    
    for i, (lr_mul, lr_tag) in enumerate(LR_MULTIPLIERS, start=len(C_GRID)*len(LR_MULTIPLIERS)):
        config_name = f"exp_{i}_lr_{lr_tag}_baseline"

        config = {
            "parent": BASELINE_INPUT,
            "time": "0-12:00:00",
            "md5_parent_hash": parent_md5_hash,
            "params": {
                "grad_modif_placement": [],
                "grad_modif_params": ["layer_type=v1", "c=99", "eps=0"],
                "tags": [f"lr_{lr_tag}", "std_norm", "post_add_c_lr_grid_long", "grad_norm", "true_baseline"],
                "name": f"{NAME_PREFIX}_{config_name}",
                "grad_modif_type": "std_norm",
                "learning_rate": 1e-4 * lr_mul,
                "n_steps": 16_000,
            },
        }

        configs.append(config)



    with open(OUTPUT_FILE, "w") as f:
        print(f"Writing to {OUTPUT_FILE}")
        yaml.dump_all(configs, f)


if __name__ == "__main__":
    sys.exit(main())
