import itertools
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import yaml

from lizrd.grid.prepare_configs import get_yaml_md5

GRID_OUTPUT = "configs/experiments/grad_norm/activation_norm_grid/reduced_bs"
BASELINE_INPUT = "configs/experiments/grad_norm/medium_reduced_bs.yaml"


GRAD_MODIF_PLACEMENT_COMBINATIONS: List[Tuple[List[str], str]] = [
    (["post_attn", "post_attn_norm", "post_attn_add", "post_ff", "post_ff_norm", "post_ff_add"], "all"),
    (["post_attn", "post_ff"], "post_attn_and_ff"),
    (["post_attn_norm", "post_ff_norm"], "post_norm"),
    (["post_attn_add", "post_ff_add"], "post_add"),
]

STD_NORM_MODIF_PARAMS: List[Tuple[List[str], str]] = [
    (["norm_dims=\(0,1,2\)", "eps=1e-8"], "norm_dims_1_2_3"),
    (["norm_dims=\(1,2\)", "eps=1e-8"], "norm_dims_1_2"),
    (["norm_dims=\(2,\)", "eps=1e-8"], "norm_dims_2"),
]

NAME_PREFIX = "grad_act_norm"


def main():
    with open(BASELINE_INPUT, "r") as f:
        baseline = yaml.safe_load(f)

    parent_md5_hash = get_yaml_md5(BASELINE_INPUT)

    for i, ((grad_placement, tag1), (layer_type, tag2)) in enumerate(
        itertools.product(GRAD_MODIF_PLACEMENT_COMBINATIONS, STD_NORM_MODIF_PARAMS)
    ):
        config_name = f"exp_{i}_{tag1}_{tag2}"

        config = deepcopy(baseline)
        config["parent"] = BASELINE_INPUT
        config["md5_parent_hash"] = parent_md5_hash
        config["params"]["grad_modif_placement"] = grad_placement
        config["params"]["grad_modif_params"].extend(layer_type)
        config["params"]["tags"].append(tag1)
        config["params"]["tags"].append(tag2)
        config["params"]["tags"].append("activation_norm")
        config["params"]["name"] = f"{NAME_PREFIX}_{config_name}"
        config["params"]["grad_modif_type"] = "activation_norm"

        output_file = (Path(GRID_OUTPUT) / config_name).with_suffix(".yaml")
        with open(output_file, "w") as f:
            print(f"Writing to {output_file}")
            yaml.dump(config, f)


if __name__ == "__main__":
    sys.exit(main())
