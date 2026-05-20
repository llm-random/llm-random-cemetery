"""Local submit driver for context-scaling eval. Run setup_ctx_eval.py first.

Hydra entrypoint. Verifies {out_dir}/jobs.json and {out_dir}/ctx_eval.sbatch
exist (both written by setup_ctx_eval.py), pushes the working tree on a temp
branch (force-adding the generated artifacts), ssh's to the cluster, clones
the branch, opens a tmux session, exports secrets, sbatch's the generated
sbatch, tails the log.
"""
import logging
import os
import sys
from pathlib import Path
from shlex import quote

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from run_exp import ConnectWithPassphrase, version_code, wait_for_job_id  # noqa: E402
import resolver  # noqa: E402

logger = logging.getLogger(__name__)

SBATCH_NAME = "ctx_eval.sbatch"


def tmux_send(connection, session: str, line: str) -> None:
    connection.run(f"tmux send -t {session}.0 {quote(line)} ENTER")


@hydra.main(version_base=None, config_path="../../../configs", config_name="ctx_eval")
def submit_eval(cfg: OmegaConf):
    if cfg.infrastructure.server == "local":
        raise RuntimeError(
            "run_ctx_eval.py is for remote submission; run ctx_eval.py directly for local."
        )

    out_dir = Path(cfg.eval.out_dir)
    for required in ("jobs.json", SBATCH_NAME):
        if not (out_dir / required).exists():
            raise FileNotFoundError(
                f"{out_dir / required} not found — run setup_ctx_eval.py first."
            )

    branch = version_code(
        remote_url=cfg.infrastructure.git.remote_url,
        force_add_paths=[str(cfg.eval.out_dir)],
        job_name=f"ctxeval_{cfg.eval.name}",
    )

    with ConnectWithPassphrase(
        host=cfg.infrastructure.server, inline_ssh_env=True
    ) as connection:
        cemetery_dir = cfg.infrastructure.cemetery_experiments_dir
        connection.run(f"mkdir -p {cemetery_dir}")
        experiment_dir = f"{cemetery_dir}/{branch}"

        if connection.run(f"test -d {experiment_dir}", warn=True).failed:
            connection.run(
                f"git clone --depth 1 -b {branch} "
                f"{cfg.infrastructure.git.remote_url} {experiment_dir}"
            )
        else:
            print(f"{experiment_dir} already exists; skipping clone.")

        connection.run(f"tmux new -d -s {branch}")
        tmux_send(connection, branch, f"cd {experiment_dir}")

        # secrets propagate to the job via sbatch's default --export=ALL
        for var in resolver.ENV_VARS_TO_FORWARD:
            if var in os.environ:
                tmux_send(connection, branch, f"export {var}={quote(os.environ[var])}")
            else:
                logger.warning("%s not in local env; eval job may fail.", var)

        tmux_send(connection, branch, f"sbatch {out_dir / SBATCH_NAME}")

        job_id = wait_for_job_id(connection, branch, tries=120)
        print(f"Job ID: {job_id}")
        tmux_send(connection, branch, f"tail -f --retry slurm-{job_id}_0.out")

        if os.environ.get("LOGLEVEL", "WARNING").upper() == "DEBUG":
            connection.run(f"tmux attach-session -t {branch}", pty=True)


if __name__ == "__main__":
    submit_eval()
