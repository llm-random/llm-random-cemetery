"""Write the 1-GPU follow-up jobs of the rebuttal (lm-eval and E4 on finished checkpoints).

    python scripts/rebuttal/post_jobs.py   # -> rebuttal_jobs/post_<what>_<cluster>/exp.job

The environment section is copied from a generated training exp.job of the same
cluster, so the job runs in the same env with its secrets commented out (they come from
sbatch's environment). Submit from the clone root with a dependency on the training jobs
(`sbatch --dependency=afterany:<ids> rebuttal_jobs/post_.../exp.job`); every step
tolerates a missing checkpoint, since a twin cancelled on one cluster never writes one.
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CL = {
    "lem": dict(sfx="lem", ckpt="/lustre/pd03/plgrid/plgllmefficont3/nano/plgmstefaniak/checkpoints/rebuttal",
                slurm=["--account=plgllmefficont3", "--partition=plgrid-lem-gpu-h100", "--gres=gpu:hopper:1",
                       "--cpus-per-gpu=16", "--mem-per-gpu=220G", "--exclude=r16-11"],
                env_from="rb_e1_pcd90_1bt_s1_lem"),
    "entropy": dict(sfx="ent", ckpt="/storage_nvme_4/nano/mstefaniak/checkpoints/rebuttal",
                    slurm=["--partition=h100", "--gres=gpu:h100:1", "--cpus-per-gpu=14", "--mem-per-gpu=90G"],
                    env_from="rb_e1_pcd90_1bt_s1_ent"),
}


def env_section(job):
    text = open(os.path.join(REPO, "rebuttal_jobs", job, "exp.job")).read()
    start = text.index("#---------- SCRIPT ----------")
    end = text.index("#-------- SCRIPT END --------") + len("#-------- SCRIPT END --------")
    return text[start:end]


def write(name, cluster, time, body):
    c = CL[cluster]
    lines = ["#!/bin/bash -l", f"#SBATCH --job-name={name}", "#SBATCH --nodes=1",
             f"#SBATCH --time={time}", f"#SBATCH --output=rebuttal_jobs/{name}/slurm-%j.out"]
    lines += [f"#SBATCH {s}" for s in c["slurm"]]
    lines += ["", env_section(c["env_from"]), "",
              "export MASTER_ADDR=localhost",
              "export HF_DATASETS_TRUST_REMOTE_CODE=1  # social_iqa (not set by entropy.yaml)",
              "export MASTER_PORT=$((40000 + SLURM_JOB_ID % 10000))",
              f"CK={c['ckpt']}", f"OUT={c['ckpt']}/{name}", 'mkdir -p "$OUT"',
              "set -x", body]
    d = os.path.join(REPO, "rebuttal_jobs", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "exp.job"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(d, "exp.job"))


def main():
    for cluster, c in CL.items():
        s = c["sfx"]
        models = [f"e1_pcd90_1bt_s{k}_{s}=$CK/rb_e1_pcd90_1bt_s{k}_{s}/*/0/step_2047/hf" for k in (1, 2)]
        models += [f"e1_hpd90_1bt_s{k}_{s}=$CK/rb_e1_hpd90_1bt_s{k}_{s}/*/0" for k in (1, 2)]
        if cluster == "entropy":  # the seed-0 replica saves a checkpoint too (E4 seed 0)
            models.append(f"rep_pcd90_1bt_s0_{s}=$CK/rb_rep_pcd90_1bt_s0_{s}/*/0/step_2047/hf")
        e4_jobs = [f"rb_e1_pcd90_1bt_s{k}_{s}" for k in (1, 2)]
        if cluster == "entropy":
            e4_jobs.append(f"rb_rep_pcd90_1bt_s0_{s}")
        body = ["# E4 first: it is quick, and lm-eval runs longest.",
                "for J in " + " ".join(e4_jobs) + "; do",
                '  STEP=$(ls -d $CK/$J/*/0/step_2047 2>/dev/null | tail -1)',
                '  if [ -n "$STEP" ] && [ -d "$STEP/model" ]; then',
                "    torchrun --nnodes=1 --nproc-per-node=1 --master-port=$MASTER_PORT "
                "scripts/rebuttal/e4_knockout.py --job-config rebuttal_jobs/$J/config_0.yaml "
                '--ckpt "$STEP" --out "$OUT/e4_$J.json" || echo "E4 FAILED $J"',
                '  else echo "E4 SKIP $J: no checkpoint"; fi',
                "done",
                "python scripts/rebuttal/lmeval_ckpts.py --out-dir \"$OUT/lmeval\" " + " ".join(models)
                + ' || echo "LMEVAL had missing or failed models"']
        write(f"post_e1_{cluster}", cluster, "0-04:00:00", "\n".join(body))

        # E4 sanity on the E3 PCD-masked checkpoints: as trained == knocked out, exactly.
        j3 = [f"rb_e3_pcd90_2bt_s0_masked_{s}", f"rb_e3_pcd70_2bt_s0_masked_{s}"]
        body3 = ["for J in " + " ".join(j3) + "; do",
                 '  STEP=$(ls -d $CK/$J/*/0/step_4095 2>/dev/null | tail -1)',
                 '  if [ -n "$STEP" ] && [ -d "$STEP/model" ]; then',
                 "    torchrun --nnodes=1 --nproc-per-node=1 --master-port=$MASTER_PORT "
                 "scripts/rebuttal/e4_knockout.py --job-config rebuttal_jobs/$J/config_0.yaml "
                 '--ckpt "$STEP" --out "$OUT/e4_$J.json" || echo "E4 FAILED $J"',
                 '  else echo "E4 SKIP $J: no checkpoint"; fi',
                 "done"]
        write(f"post_e3_{cluster}", cluster, "0-01:30:00", "\n".join(body3))


if __name__ == "__main__":
    main()
