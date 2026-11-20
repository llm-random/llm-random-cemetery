"""GPU smoke of the rebuttal code paths, built from the REAL queued job configs.

Each smoke config is a real job's composed config_0.yaml with only the model size, the
assets, the data, the step count and the logger swapped (tiny model, random source and
importances, the repo's 644 KB sample dataset, 12 steps, stdout logger). Trainer class,
clipping, optimizers, dataset class with holdout: null, distillation type and the mask
flag are untouched, so the smoke exercises exactly the code the jobs will run.

    python scripts/rebuttal/smoke_configs.py --root /home/mstefaniak/rb_smoke
writes rebuttal_smoke/<run>/config_0.yaml and rebuttal_smoke/assets/{tiny_source,tiny_imp}.pt
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import resolver  # noqa: E402,F401
from main import dump_grid_configs  # noqa: E402

RUNS = {  # smoke name: real job it is cut from
    "s1_pcd": "rb_e1_pcd90_1bt_s1_ent",                     # E1 / replica PCD path
    "s2_pcd_masked": "rb_e3_pcd90_2bt_s0_masked_ent",       # E3 PCD-masked (teacher-safe copies)
    "s3_pc_masked": "rb_e3_pc90_1bt_s0_masked_ent",         # E3 PC-masked (in place)
    "s4_pc_clip1": "rb_e6_pc90_1bt_s0_clip1_ent",           # E6 PC / replica PC path
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="smoke root on the cluster")
    args = ap.parse_args()
    out = os.path.join(REPO, "rebuttal_smoke")
    os.makedirs(os.path.join(out, "assets"), exist_ok=True)

    with initialize_config_dir(config_dir=os.path.join(REPO, "configs", "_model", "llama"),
                               version_base=None):
        tiny = compose(config_name="projected_mem_eff_tiny")
    tc = tiny.common
    # assets: a random tiny source model in nano format + shape-correct importances
    torch.manual_seed(0)
    src = instantiate(tiny.model.source_model, _convert_="all")
    torch.save(src.state_dict(), os.path.join(out, "assets", "tiny_source.pt"))
    torch.save({"dmodel_importances": torch.rand(tc.base_dmodel),
                "dff_importances": torch.rand(tc.n_blocks, tc.base_dff)},
               os.path.join(out, "assets", "tiny_imp.pt"))

    for run, job in RUNS.items():
        cfg = OmegaConf.load(os.path.join(REPO, "rebuttal_jobs", job, "config_0.yaml"))
        for k in ("_run_", "overrides"):
            cfg.pop(k, None)
        cfg.model = tiny.model
        for k, v in tiny.common.items():
            cfg.common[k] = v
        cfg.common.sequence_length = 64
        cfg.common.batch_size = 16
        cfg.trainer.n_steps = 12
        cfg.trainer.gradient_accumulation_steps = 2
        cfg.trainer.scheduler.warmup_steps = 2
        cfg.trainer.train_dataloader.dataset.path = f"{args.root}/data"
        cfg.trainer.checkpoint.save.path = f"{args.root}/ckpt/{run}"
        cfg.trainer.checkpoint.save.type = "nano"  # the HF export needs Llama-1B dims
        cfg.projected_compression.source_model_path = f"{args.root}/assets/tiny_source.pt"
        cfg.projected_compression.path_to_importances = f"{args.root}/assets/tiny_imp.pt"
        cfg.cluster_switch.train_path_fineweb = f"{args.root}/data"
        cfg.cluster_switch.eval_path_fineweb = f"{args.root}/data"
        cfg.infrastructure.metric_logger.type = "stdout"
        cfg.infrastructure.metric_logger.name = f"smoke_{run}"
        d = os.path.join(out, run)
        dump_grid_configs([(OmegaConf.to_container(cfg, resolve=False), [])], d)
        # sanity: resolves and keeps the real job's training settings
        r = OmegaConf.to_container(OmegaConf.load(os.path.join(d, "config_0.yaml")), resolve=True)
        print(run, "<-", job, "| trainer", r["trainer"]["_target_"].split(".")[-1],
              "| clip", r["trainer"]["gradient_clipping"], r["trainer"].get("only_target_model_gradient_clipping"),
              "| mask", r["projected_compression"]["mask_discarded_source"],
              "| distil", r["projected_compression"]["source_model_for_distillation"],
              "| holdout", r["trainer"]["train_dataloader"]["dataset"]["holdout"],
              "| dims", r["common"]["dmodel"], r["common"]["dff"])


if __name__ == "__main__":
    main()
