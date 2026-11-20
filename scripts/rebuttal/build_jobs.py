"""Build the AISTATS 2027 rebuttal training jobs (plan: pc_paper/reviews/rebuttal_plan.md).

For every job and cluster this writes configs/rebuttal/<name>.yaml, composes it with
Hydra exactly as run_exp.py would, and writes rebuttal_jobs/<name>/{config_0.yaml,exp.job}.
Submit from the clone root with `sbatch rebuttal_jobs/<name>/exp.job`.

Each job copies the logged wandb config of a Table 1 run (the ground truth, not the repo
YAMLs) and changes only what the experiment changes: the data-order seed (E1), the
clipping threshold (E6), the discarded-block mask (E3), or nothing (seed-0 replicas).
`--verify` diffs every composed config against the logged config of its reference run.

Run with WANDB_API_KEY and HF_TOKEN unset, so the exp.job export lines come out
commented; the secrets reach the job through sbatch's environment at submit time.

    env -u WANDB_API_KEY -u HF_TOKEN python scripts/rebuttal/build_jobs.py [--verify DIR]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

import yaml  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import resolver  # noqa: E402,F401  (registers eval / env-placeholder resolvers)
from grid_generator.generate_configs import create_grid_config  # noqa: E402
from grid_generator.sbatch_builder import generate_sbatch_script  # noqa: E402
from main import dump_grid_configs  # noqa: E402

RATIOS = {70: (960, 3840), 90: (448, 1728)}
STEPS = {"1bt": 2048, "2bt": 4096}

# Llama 3 8B (E5 peak memory only), from the logged configs of the 8B 90% Table 1 runs.
SIZE_8B = {
    "ratios": {90: (1056, 3168)},
    "lr": {("pc", 90): 13, ("pcd", 90): 13, ("hp", 90): 9, ("hpd", 90): 9},
    "seq": 2048, "gacc": 32, "hf": "meta-llama/Llama-3.1-8B",
    "pc_model": "projected_mem_eff_8B", "hp_model": "llama_3_8",
    "distil_trainer": "llama3_8B_distillation",
    "assets": {"lem": (
        "/lustre/pd01/plgrid/plgllmefficont2/projected_compression/hf_to_nano_converted/llama8b_in_nano.pt",
        "/lustre/pd01/plgrid/plgllmefficont2/projected_compression/importances/8b/minitron_dimensions_importances.pt",
    )},
    "reference": {("pc", 90): "v45fo4fk", ("hp", 90): "5dhq5nlz",
                  ("pcd", 90): "m4aghdz0", ("hpd", 90): "y2vjj8y1"},
}

# Learning rates n (LR = 2^-n), from the logged configs of the Table 1 runs.
LR = {
    ("pc", 70): 13, ("pc", 90): 13,
    ("pcd", 70): 13, ("pcd", 90): 13,
    ("hp", 70): 12, ("hp", 90): 11,
    ("hpd", 70): 11, ("hpd", 90): 12,
}

CLUSTERS = {
    "lem": {
        "source": "/lustre/pd01/plgrid/plgllmefficont2/projected_compression/hf_to_nano_converted/llama1b_in_nano.pt",
        "importances": "/lustre/pd01/plgrid/plgllmefficont2/projected_compression/importances/1b/minitron_dimensions_importances.pt",
        "ckpt_root": "/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/checkpoints/rebuttal",
        "gres": "gpu:hopper:4",
        "extra_slurm": {"exclude": "r16-11"},
    },
    "entropy": {
        "source": "/storage_nvme_4/nano/j321m/pc/llama1b_in_nano.pt",
        "importances": "/storage_nvme_4/nano/j321m/pc/importances/1b/minitron_dimensions_importances.pt",
        "ckpt_root": "/storage_nvme_4/nano/mstefaniak/checkpoints/rebuttal",
        "gres": "gpu:h100:4",
        "extra_slurm": {},
    },
}

# Reference (seed-0, Table 1) run of each (method, ratio, tokens) cell.
REFERENCE_RUN = {
    ("pc", 90, "1bt"): "a83pvc44", ("pc", 70, "1bt"): "tu71ks3t",
    ("hp", 90, "1bt"): "mqlk0lvt", ("hp", 70, "1bt"): "3hs71dfy",
    ("pcd", 90, "1bt"): "xk83ntk1", ("pcd", 70, "1bt"): "45yj0mdp",
    ("hpd", 90, "1bt"): "hvd1i37o", ("hpd", 70, "1bt"): "xogsbrqa",
    ("pcd", 90, "2bt"): "k9ndr8i1", ("pcd", 70, "2bt"): "pk6lz5t9",
    ("hpd", 90, "2bt"): "pajazura", ("hpd", 70, "2bt"): "4yuyjuf9",
}

# Hours of the reference run on lem (documents/notes/table1_runs.md), for time limits.
HOURS = {
    ("pc", 90, "1bt"): 0.81, ("pc", 70, "1bt"): 1.14,
    ("hp", 90, "1bt"): 0.71, ("hp", 70, "1bt"): 1.00,
    ("pcd", 90, "1bt"): 1.69, ("pcd", 70, "1bt"): 2.00,
    ("hpd", 90, "1bt"): 1.51, ("hpd", 70, "1bt"): 1.79,
    ("pcd", 90, "2bt"): 3.33, ("pcd", 70, "2bt"): 3.96,
    ("hpd", 90, "2bt"): 3.00, ("hpd", 70, "2bt"): 3.54,
}

SHARDED = [
    "src.projected_compression.mem_eff.CompressibleBlock",
    "${model.source_model.embedding._target_}",
    "${model.source_model.encoder.block_fn._target_}",
    "${model.source_model.head._target_}",
]


def job(action, method, ratio, tokens, seed=123, clip=None, mask=False, save=None, tag=None,
        n_steps=None, size="1b", env=None, clusters=("lem", "entropy")):
    """One training job. seed 123 is the Table 1 data order ("seed 0" in the plan).
    n_steps overrides the token budget (smoke runs); tokens then names the reference cell.
    env: extra `export K=V` lines for exp.job (E5's NANO_PEAK_MEMORY_EVERY)."""
    seed_label = "s0" if seed == 123 else f"s{seed}"
    size_label = "" if size == "1b" else f"{size}_"
    name = f"rb_{action}_{size_label}{method}{ratio}_{tokens if n_steps is None else f'{n_steps}st'}_{seed_label}" + (f"_{tag}" if tag else "")
    return dict(action=action, method=method, ratio=ratio, tokens=tokens, seed=seed,
                clip=clip, mask=mask, save=save, name=name, n_steps=n_steps, size=size,
                env=env or {}, clusters=clusters)


# ---- The job lists (plan §6.1). Order = lem priority order; entropy gets it reversed. ----
E1 = [job("e1", m, 90, "1bt", seed=s, save="final") for m in ("pcd", "hpd") for s in (1, 2)]
E6 = [job("e6", "pc", 90, "1bt", clip=1.0, tag="clip1"),
      job("e6", "hp", 90, "1bt", clip=0.25, tag="clip025")]
E3 = [job("e3", "pcd", 70, "2bt", mask=True, tag="masked", save="final"),
      job("e3", "pcd", 90, "2bt", mask=True, tag="masked", save="final"),
      job("e3", "pc", 70, "1bt", mask=True, tag="masked", save="final"),
      job("e3", "pc", 90, "1bt", mask=True, tag="masked", save="final")]
# 50-step smokes of both mask modes (teacher-safe copies for PCD, in place for PC),
# including the end-of-run save; run alongside E3, cancel E3 if they fail.
E3_SMOKE = [job("e3smoke", "pcd", 90, "2bt", mask=True, tag="masked", save="final", n_steps=50),
            job("e3smoke", "pc", 90, "1bt", mask=True, tag="masked", save="final", n_steps=50)]
# Seed-0 replicas of every comparator, entropy only (same-cluster comparators).
REPLICAS = [job("rep", m, r, t, save=("final" if (m, r, t) == ("pcd", 90, "1bt") else None))
            for (m, r, t) in [("pcd", 70, "2bt"), ("hpd", 70, "2bt"), ("pcd", 90, "2bt"),
                              ("hpd", 90, "2bt"), ("pcd", 90, "1bt"), ("hpd", 90, "1bt"),
                              ("pc", 70, "1bt"), ("hp", 70, "1bt"), ("pc", 90, "1bt"),
                              ("hp", 90, "1bt")]]

# E5 peak memory: Table 1 configs, 30 steps, no save. The 8B assets are verified only on
# lem (entropy's h100a had no free CPU for a check on 2026-11-20), so 8B runs on lem only.
E5_ENV = {"NANO_PEAK_MEMORY_EVERY": "5"}
E5 = ([job("e5", m, 90, "1bt", n_steps=30, size="8b", env=E5_ENV, clusters=("lem",))
       for m in ("pcd", "hpd", "pc", "hp")]
      + [job("e5", m, 90, "1bt", n_steps=30, env=E5_ENV) for m in ("pcd", "hpd", "pc", "hp")])

GROUPS = {"e1": E1, "e6": E6, "e3": E3, "e3smoke": E3_SMOKE, "rep": REPLICAS, "e5": E5}


def time_limit(j):
    if j["n_steps"] is not None:
        return "0-01:30:00" if j["size"] == "8b" else "0-00:40:00"
    h = HOURS[(j["method"], j["ratio"], j["tokens"])]
    hours = int(h * 2.0 + 1.5)  # 2x the lem run plus startup margin
    return f"0-{hours:02d}:00:00"


def yaml_for(j, cluster):
    c = dict(CLUSTERS[cluster])
    m, ratio, tokens = j["method"], j["ratio"], j["tokens"]
    big = j["size"] == "8b"
    if big:
        dmodel, dff = SIZE_8B["ratios"][ratio]
        c["source"], c["importances"] = SIZE_8B["assets"][cluster]
    else:
        dmodel, dff = RATIOS[ratio]
    distil = m in ("pcd", "hpd")
    pc = m in ("pc", "pcd")
    run_name = f"{j['name']}_{'lem' if cluster == 'lem' else 'ent'}"
    ckpt_path = f"{c['ckpt_root']}/{run_name}" if j["save"] else None

    defaults = [{"../_cluster@_here_": cluster}]
    if pc:
        defaults.append({"../_model/llama@_here_": SIZE_8B["pc_model"] if big else "projected_mem_eff_1B"})
    else:
        defaults.append({"../_model@_here_": SIZE_8B["hp_model"] if big else "llama_3_1"})
    distil_trainer = SIZE_8B["distil_trainer"] if big else "llama3_1B_distillation"
    defaults += [
        {"../_trainer@_here_": distil_trainer if distil else "llama"},
        {"../_dataset@_here_": "fineweb"},
        {"../_checkpoints@_here_": "none"},
        {"../_misc@_here_": "default"},
        "_self_",
    ]

    hf = SIZE_8B["hf"] if big else "meta-llama/Llama-3.2-1B"
    common = {"sequence_length": SIZE_8B["seq"] if big else 1024, "batch_size": 512}
    trainer = {
        "gradient_accumulation_steps": SIZE_8B["gacc"] if big else 8,
        "n_steps": STEPS[tokens] if j["n_steps"] is None else j["n_steps"],
        "learning_rate": (SIZE_8B["lr"] if big else LR)[(m, ratio)],
        "scheduler": {"warmup_steps": 40},
        # The Table 1 runs predate the fineweb holdout (e894422f); without this the
        # training stream, and so the seed-123 data order, would differ from theirs.
        "train_dataloader": {"dataset": {"seed": j["seed"], "holdout": None}},
    }
    cfg = {"defaults": defaults, "common": common, "trainer": trainer}

    if pc:
        common.update({"dmodel": dmodel, "dff": dff})
        trainer.update({
            "_target_": ("src.projected_compression.trainer_distillation.PCDistillationTrainer"
                         if distil else "src.projected_compression.trainer.PCTrainer"),
            "distributed": {"fsdp2": None},
            "eval_dataloader": None,
            "eval_interval": 0,
            "gradient_clipping": j["clip"] if j["clip"] is not None else 0.25,
            "only_compress_model_gradient_clipping": False,
            "only_target_model_gradient_clipping": 1.0,
            "original_llama_path": hf,
            "checkpoint": {"save": {
                "interval": 0,
                "path": ckpt_path,
                "type": "nano_and_hf",
            }},
        })
        cfg["projected_compression"] = {
            "modules_to_shard": SHARDED,
            "source_model_path": c["source"],
            "path_to_importances": c["importances"],
            "init_norms_with_ones": False,
            "separate_block_optimizers": True,
            "adjust_grad_norm": False,
            "cpu_offload_projections": False,
            "source_model_for_distillation": distil,
            "mask_discarded_source": bool(j["mask"]),
        }
        cfg["model"] = {"cast_bfloat16": False}
        if distil:
            cfg["distillation"] = {"load": {"type": "pc_memeff_base"}}
    else:
        common.update({"target_dmodel": dmodel, "target_dff": dff})
        trainer["checkpoint"] = {
            "load": {"type": "huggingface", "path": hf},
            "save": {"type": "huggingface", "path": ckpt_path},
        }
        if j["clip"] is not None:
            trainer["gradient_clipping"] = j["clip"]
        if distil or big:  # the 8B HP Table 1 run had no periodic eval either
            trainer.update({"eval_interval": 0, "eval_dataloader": None})
        cfg["apply_functions"] = [{
            "_target_": "src.projected_compression.pruning.prune",
            "_partial_": True,
            "dimensions_importances_path": c["importances"],
            "target_dmodel": "${common.target_dmodel}",
            "target_dff": "${common.target_dff}",
        }]

    slurm = {"job-name": run_name, "gres": c["gres"], "time": time_limit(j)}
    slurm.update(c["extra_slurm"])
    cfg["infrastructure"] = {
        "metric_logger": {
            "name": run_name,
            "new_wandb_job": True,
            "wandb_entity": "ideas_cv",
            "project_name": "pc_v2",
            "tags": ["rebuttal", j["action"], m, f"{ratio}p", tokens, f"seed{j['seed']}",
                     cluster] + (["masked"] if j["mask"] else [])
                    + ([f"clip{j['clip']}"] if j["clip"] is not None else []),
        },
        "slurm": slurm,
    }
    return run_name, cfg


def build(j, cluster, out_root):
    run_name, cfg = yaml_for(j, cluster)
    if j["size"] == "8b" and cluster not in SIZE_8B["assets"]:
        raise ValueError(f"{run_name}: no verified 8B assets on {cluster}")
    cfg_dir = os.path.join(REPO, "configs", "rebuttal")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f"{run_name}.yaml"), "w") as f:
        f.write(f"# Generated by scripts/rebuttal/build_jobs.py -- edit the script, not this.\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        composed = compose(config_name=run_name)
    missing = OmegaConf.missing_keys(composed)
    if missing:
        raise RuntimeError(f"{run_name}: missing keys {missing}")
    grid = create_grid_config(composed)
    assert len(grid) == 1, f"{run_name}: expected one config, got {len(grid)}"

    job_dir = os.path.join(out_root, run_name)
    shutil.rmtree(job_dir, ignore_errors=True)
    dump_grid_configs(grid, job_dir)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            generate_sbatch_script(composed.infrastructure.slurm,
                                   f"rebuttal_jobs/{run_name}", 1, None,
                                   composed.infrastructure.get("script", None))
        finally:
            os.chdir(cwd)
        shutil.move(os.path.join(tmp, "exp.job"), os.path.join(job_dir, "exp.job"))
    if j["env"]:
        exp_job = os.path.join(job_dir, "exp.job")
        text = open(exp_job).read()
        marker = "#---------- SCRIPT ----------\n"
        assert marker in text, f"{run_name}: no script section in exp.job"
        exports = "".join(f"export {k}={v}\n" for k, v in j["env"].items())
        open(exp_job, "w").write(text.replace(marker, marker + exports, 1))
    return run_name, OmegaConf.to_container(composed, resolve=True)


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    else:
        out[p] = d
    return out


IGNORE = ("infrastructure", "run_env", "checkpoint_config", "cluster_switch", "overrides",
          "_run_")


def verify(j, resolved, ref_dir):
    if j["size"] == "8b":
        ref = SIZE_8B["reference"][(j["method"], j["ratio"])]
    else:
        ref = REFERENCE_RUN[(j["method"], j["ratio"], j["tokens"])]
    logged = json.load(open(os.path.join(ref_dir, f"{ref}.json")))["config"]
    a, b = flat(resolved), flat(logged)
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k.startswith(IGNORE):
            continue
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, float) and isinstance(vb, (int, float)) and va == vb:
            continue
        if str(va) != str(vb):
            diffs.append((k, va, vb))
    return ref, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="e1,e6,rep,e3")
    ap.add_argument("--verify", default=None, help="dir with <wandb id>.json logged configs")
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_jobs"))
    args = ap.parse_args()
    for var in ("WANDB_API_KEY", "HF_TOKEN"):
        if var in os.environ:
            sys.exit(f"Unset {var} first: exp.job must not carry it (see the docstring).")

    for g in args.groups.split(","):
        for j in GROUPS[g]:
            clusters = ["entropy"] if g == "rep" else list(j["clusters"])
            for cluster in clusters:
                run_name, resolved = build(j, cluster, args.out)
                line = f"{run_name}: ok"
                if args.verify:
                    ref, diffs = verify(j, resolved, args.verify)
                    line += f"  vs {ref}: {len(diffs)} diffs"
                    for k, va, vb in diffs:
                        line += f"\n    {k}: new={str(va)[:90]!s} | logged={str(vb)[:90]!s}"
                print(line)


if __name__ == "__main__":
    main()
