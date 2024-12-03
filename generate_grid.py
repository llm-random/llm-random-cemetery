import sys
from typing import List, Optional, Tuple, Union

import itertools
import yaml

from lizrd.grid.prepare_configs import get_yaml_md5

OUTPUT_FILE = "configs/experiments/grad_norm/std_norm_grid/c_lr_placement_grid_std_v4_short.yaml"
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

STD_V1_LONG_CONFIGS: List[Tuple[List[str], float, List[str]]] = [ # placements, c, tags
    (["post_attn_add", "post_ff_add"], 1e-4, ["post_add", "c_1e-4"]),
    (["post_attn", "post_ff"], 1e-4, ["post_attn_and_ff", "c_1e-4"]),
    (["post_attn_norm", "post_ff_norm"], 1e-3, ["post_norm", "c_1e-3"]),
    (["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"], 1e-5, ["all", "c_1e-5"]),
]

STD_V2_LONG_CONFIGS: List[Tuple[List[str], float, List[str]]] = [ # placements, c, tags
    (["post_attn", "post_ff"], 1e-5, ["post_attn_and_ff", "c_1e-5"]),
    (["post_attn", "post_ff"], 1e-4, ["post_attn_and_ff", "c_1e-4"]),
    (["post_attn_add", "post_ff_add"], 1e-4, ["post_add", "c_1e-4"]),
    (["post_attn_add", "post_ff_add"], 1e-5, ["post_add", "c_1e-5"]),
]

SCALE_NORM_LONG_CONFIGS: List[Tuple[List[str], float, Union[float, str], str, List[str]]] = [ # placements, c, k, norm_dims, tags
    (["post_attn_add", "post_ff_add"], 1e-4, 1, "(0,1,2)", ["post_add", "c_1e-4", "k_1", "dims_0_1_2"]),
    (["post_attn_add", "post_ff_add"], 1e-4, 1, "(1,2)", ["post_add", "c_1e-4", "k_1", "dims_1_2"]),
    (["post_attn_add", "post_ff_add"], 1, "auto", "(0,1,2)", ["post_add", "c_1", "k_auto", "dims_0_1_2"]),
    (["post_attn_add", "post_ff_add"], 1, "auto", "(1,2)", ["post_add", "c_1", "k_auto", "dims_1_2"]),
    (["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"], 1e-4, 1, "(0,1,2)", ["all", "c_1e-4", "k_1", "dims_0_1_2"]),
    (["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"], 1e-5, 1, "(0,1,2)", ["all", "c_1e-5", "k_1", "dims_0_1_2"]),
    (["post_attn_norm", "post_ff_norm"], 1e-4, 1, "(1,2)", ["post_norm", "c_1e-4", "k_1", "dims_1_2"]),
    (["post_attn_norm", "post_ff_norm"], 1e-4, 1, "(0,1,2)", ["post_norm", "c_1e-4", "k_1", "dims_0_1_2"]),
    (["post_attn", "post_ff"], 1e-4, 1, "(1,2)", ["post_attn_and_ff", "c_1e-4", "k_1", "dims_1_2"]),
    (["post_attn", "post_ff"], 1e-3, 1, "(1,2)", ["post_attn_and_ff", "c_1e-3", "k_1", "dims_1_2"]),
]


NAME_PREFIX = "std_v4_c_lr_grid_placement_short"
LR = 1e-4
N_STEPS = 2_000
COMMON_TAGS = ["reduced_bs", "grad_norm", "std_v4", "short", "std_norm"]

def main():
    parent_md5_hash = get_yaml_md5(BASELINE_INPUT)
    configs = []
    time = "0-4:00:00"

    for (placements, placement_tag), (c, c_tag), (lr_multiplier, lr_tag) in itertools.product(
        GRAD_MODIF_PLACEMENT_COMBINATIONS,
        C_GRID,
        LR_MULTIPLIERS
    ):

        tag_str = f"{placement_tag}_{c_tag}_{lr_tag}"
        config_name = f"std_v4_{tag_str}_exp_{len(configs) + 1}"

        config = {
            "parent": BASELINE_INPUT,
            "time": time,
            "md5_parent_hash": parent_md5_hash,
            "params": {
                "grad_modif_placement": placements,
                "grad_modif_params": ["layer_type=v4", f"c={c}", "eps=0"],
                "tags": [*COMMON_TAGS, placement_tag, c_tag, lr_tag],
                "name": f"{NAME_PREFIX}_{config_name}",
                "grad_modif_type": "std_norm",
                "learning_rate": LR * lr_multiplier,
                "n_steps": N_STEPS,
            },
        }
        configs.append(config)

    with open(OUTPUT_FILE, "w") as f:
        print(f"Writing to {OUTPUT_FILE}")
        yaml.dump_all(configs, f)

if __name__ == "__main__":
    sys.exit(main())
