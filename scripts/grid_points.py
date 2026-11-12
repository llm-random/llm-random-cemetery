"""Compose a run_exp config as run_exp.py does and print every grid point.

For documents/notes/ame_bank_sigma_grid.md. Nothing is submitted: the config is
composed with Hydra, grid-expanded with run_exp's own create_grid_config,
schema-checked with check_point, and the sbatch header run_exp would write is
printed. With REF_RUN, each point's resolved config is also diffed against that
W&B run's logged config (runtime keys skipped).

usage (from the tree's root):
  python scripts/grid_points.py CONFIG_DIR CONFIG_NAME [REF_RUN [POINT_INDEX]]
"""

import os
import sys
import tempfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
import resolver  # noqa: E402,F401  (registers the custom resolvers)
from grid_generator.generate_configs import create_grid_config  # noqa: E402
from grid_generator.sbatch_builder import generate_sbatch_script  # noqa: E402
from src.definitions import check_point  # noqa: E402

RUNTIME = ("infrastructure.", "_wandb", "checkpoint_config", "job_id", "slurm")


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def main():
    config_dir, config_name = os.path.abspath(sys.argv[1]), sys.argv[2]
    ref_run = sys.argv[3] if len(sys.argv) > 3 else None
    only = int(sys.argv[4]) if len(sys.argv) > 4 else None
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    grid = create_grid_config(cfg)
    points = []
    for i, (point, overrides) in enumerate(grid):
        checked = check_point(point)
        resolved = OmegaConf.to_container(checked, resolve=True)
        ame = resolved["common"]["attention"].get("ame", {})
        print(f"point {i}: overrides {overrides} -> bank_init_std {ame.get('bank_init_std')!r}, "
              f"bank_gate {ame.get('bank_gate')!r}, lr {resolved['trainer']['learning_rate']}, "
              f"name {resolved['infrastructure']['metric_logger']['name']}, "
              f"tags {resolved['infrastructure']['metric_logger']['tags']}")
        points.append(resolved)
    # generate_sbatch_script writes exp.job into the cwd: do it in a temp dir.
    inf = cfg.infrastructure
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            generate_sbatch_script(inf.slurm, "generated_configs", len(grid),
                                   inf.get("max_concurrent_jobs", None), None)
            lines = open("exp.job").read().splitlines()
        finally:
            os.chdir(here)
    print("sbatch header:", [x for x in lines if x.startswith("#SBATCH")])
    if ref_run:
        import wandb
        ref = wandb.Api(timeout=120).run(f"ideas_cv/nano-ame/{ref_run}")
        logged = flat({k: v for k, v in ref.config.items()})
        for i, resolved in enumerate(points):
            if only is not None and i != only:
                continue
            mine = flat(resolved)
            keys = sorted(set(mine) | set(logged))
            diff = [(k, logged.get(k, "<absent>"), mine.get(k, "<absent>")) for k in keys
                    if not k.startswith(RUNTIME) and logged.get(k, "<absent>") != mine.get(k, "<absent>")]
            print(f"== point {i} against {ref_run}'s logged config: {len(diff)} keys differ")
            for k, a, b in diff:
                print(f"   {k}: {a!r} -> {b!r}")


if __name__ == "__main__":
    main()
