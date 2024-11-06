import itertools
import sys
from typing import List, Optional, Tuple

import yaml

from lizrd.grid.prepare_configs import get_yaml_md5

OUTPUT_FILE = "configs/experiments/grad_norm/std_norm_grid/c_lr_placement_dims_grid_scale_norm_k_auto_short.yaml"
BASELINE_INPUT = "configs/experiments/grad_norm/medium_reduced_bs.yaml"


GRAD_MODIF_PLACEMENT_COMBINATIONS: List[Tuple[List[str], Optional[str]]] = [
    (["post_attn", "post_ff"], "post_attn_and_ff"),
    (["post_attn_norm", "post_ff_norm"], "post_norm"),
    (["post_attn_add", "post_ff_add"], "post_add"),
    (["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"], "all"),
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
    (1 / 10, "div10"),
    (1, "mul1"),
    (10, "mul10"),
    (100, "mul100"),
]

C_GRID: List[Tuple[float, str]] = [
    (0, "c_0"),
    (1, "c_1"),
    (1e-1, "c_1e-1"),
    (1e-2, "c_1e-2"),
    (1e-3, "c_1e-3"),
    (1e-4, "c_1e-4"),
    (1e-5, "c_1e-5"),
]

NORM_DIMS: List[Tuple[str, str]] = [
    ('(2,)', "dims_2"),
    ('(1,2)', "dims_1_2"),
    ('(0,1,2)', "dims_0_1_2"),
]

NAME_PREFIX = "scale_norm_k_auto_c_lr_dims_grid_placement_short"

def main():
    parent_md5_hash = get_yaml_md5(BASELINE_INPUT)
    configs = []
    time = "0-04:00:00"

    for i, ((lr_multiplier, lr_tag), (c, c_tag), (placement, placement_tag), (norm_dims, nd_tag)) in enumerate(itertools.product(LR_MULTIPLIERS, C_GRID, GRAD_MODIF_PLACEMENT_COMBINATIONS, NORM_DIMS)):
        config_name = f"exp_{i}_lr_{lr_tag}_{c_tag}"

        config = {
            "parent": BASELINE_INPUT,
            "time": time,
            "md5_parent_hash": parent_md5_hash,
            "params": {
                "grad_modif_placement": placement,
                "grad_modif_params": ["k=auto", f"c={c}", "eps=0", f"norm_dims={norm_dims}"],
                "tags": [f"lr_{lr_tag}", "scale_norm", "scale_norm_c_lr_grid_placement_short", "k_auto", "grad_norm", c_tag, placement_tag, nd_tag],
                "name": f"{NAME_PREFIX}_{config_name}",
                "grad_modif_type": "scale_norm",
                "learning_rate": 1e-4 * lr_multiplier,
                "n_steps": 2000,
            },
        }

        configs.append(config)
    
    with open(OUTPUT_FILE, "w") as f:
        print(f"Writing to {OUTPUT_FILE}")
        yaml.dump_all(configs, f)


if __name__ == "__main__":
    sys.exit(main())
