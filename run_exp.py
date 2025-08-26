#!/usr/bin/env python
import re
import yaml
import datetime

import os
import time
from git import Repo
from contextlib import contextmanager
import copy
import getpass
from typing import Generator, Optional
from fabric import Connection
import hydra
from omegaconf import OmegaConf
import paramiko.ssh_exception
from hydra import compose, initialize
from omegaconf import OmegaConf
from grid_generator.generate_configs import create_grid_config
from grid_generator.sbatch_builder import generate_sbatch_script
from resolver import get_cluster_name

from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console()
# logger = logging.getLogger(__name__)
# print(os.getenv("LOG_LEVEL","WARNING").upper())
# logging.basicConfig(
#     level=logging.getLevelNamesMapping()[os.getenv("LOG_LEVEL","WARNING").upper()],
#     format=f"%(message)s",
#     handlers=[RichHandler(console=console, rich_tracebacks=True)],
# )

# logger.debug("XD")
# logger.info("Logging is set up.")
# logger.warning("Warning message")
# logger.error("XD")

# logger.info("🚀 Processing started", extra={"user_message": True})
# logger.info("✅ Processing complete!", extra={"user_message": True})

# exit(0)

_SSH_HOSTS_TO_PASSPHRASES = {}

def dump_grid_configs(configs_grid, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    class CustomDumper(yaml.SafeDumper):
        def write_line_break(self, data=None):
            super().write_line_break(data)
            if len(self.indents) == 1:  # Check if we're at the root level
                super().write_line_break()

    for idx, (cfg_dict, overrides_list) in enumerate(configs_grid):
        cfg_dict["overrides"] = overrides_list

        out_path = os.path.join(output_folder, f"config_{idx}.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, Dumper=CustomDumper, sort_keys=True)

def commit_pending_changes(repo: Repo):
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit(m="Versioning code", no_verify=True)

def reset_to_original_repo_state(
    repo: Repo,
    original_branch: str,
    original_branch_commit_hash: str,
    versioning_branch: str,
):
    repo.git.checkout(original_branch, "-f")
    if versioning_branch in repo.branches:
        repo.git.branch("-D", versioning_branch)
    repo.head.reset(original_branch_commit_hash, index=True)

def git_ssh_to_https_tree(ssh_url: str) -> str:
    without_git = ssh_url.removeprefix("git@").removesuffix(".git")
    host, path = without_git.split(":", 1)
    return f"https://{host}/{path}/tree"

def version_code(
    remote_url: str,
    experiment_config_path: Optional[str] = None,
    exp_job_path: Optional[str] = None,
    job_name: Optional[str] = None,
) -> str:
    repo = Repo(".", search_parent_directories=True)

    experiment_branch_name = (
        f"{job_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    original_branch = repo.active_branch.name
    original_branch_commit_hash = repo.head.object.hexsha

    repo.git.add(experiment_config_path, force=True)
    repo.git.add(exp_job_path, force=True)
    repo.git.add(all=True)

    try:
        commit_pending_changes(repo)
        repo.git.checkout(b=experiment_branch_name)

        text = Text("Pushing experiment code to branch '")
        text.append(experiment_branch_name, style="bold magenta")  # branch name in magenta
        text.append("' at '")
        text.append(remote_url, style="bold cyan")  # URL in cyan
        text.append("'...")
        spinner = Spinner("dots", text=text, style="bold green on black")
        with Live(spinner, refresh_per_second=10, console=console):
            repo.git.push(remote_url, experiment_branch_name)
        console.print(f"Experiment pushed to:  \'[bold green]{git_ssh_to_https_tree(remote_url)}/{experiment_branch_name}[/bold green]\'")
    finally:
        reset_to_original_repo_state(
            repo, original_branch, original_branch_commit_hash, experiment_branch_name
        )

    return experiment_branch_name


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        yield connection
    except paramiko.ssh_exception.PasswordRequiredException as e:
        if connection.host not in _SSH_HOSTS_TO_PASSPHRASES:
            passphrase = getpass.getpass(
                f"SSH key encrypted, provide the passphrase ({connection.host}): "
            )
            _SSH_HOSTS_TO_PASSPHRASES[connection.host] = passphrase
        else:
            passphrase = _SSH_HOSTS_TO_PASSPHRASES[connection.host]
        kwargs["connect_kwargs"] = copy.deepcopy(
            kwargs.get("connect_kwargs", {})
        )  # avoid modifying the original connect_kwargs
        kwargs["connect_kwargs"]["passphrase"] = passphrase
        connection = Connection(*args, **kwargs)
        yield connection
    finally:
        connection.close()


def get_experiment_components(
    hydra_config: OmegaConf,
) -> str:
    # this is a workaround as hydra does not provide a way to get the config path
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_name = hydra_config.job.config_name
    config_path = [
        path["path"]
        for path in hydra_config.runtime.config_sources
        if path["schema"] == "file"
    ][0]
    return config_path, config_name


@hydra.main(version_base=None, config_path=".", config_name="experiment")
def submit_experiment(
    cfg: OmegaConf,
):
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    configs_grid = create_grid_config(cfg)
    dump_grid_configs(configs_grid, cfg.infrastructure.generated_configs_path)

    modules_to_add = cfg.infrastructure.get("modules_to_add")
    generate_sbatch_script(
        cfg.infrastructure.slurm, cfg.infrastructure.generated_configs_path, len(configs_grid), cfg.infrastructure.venv_path, modules_to_add
    )

    experiment_branch_name = version_code(
        remote_url=cfg.infrastructure.git.remote_url,
        experiment_config_path=cfg.infrastructure.generated_configs_path,
        exp_job_path="exp.job",
        job_name=cfg.infrastructure.metric_logger.name,
    )

    with ConnectWithPassphrase(host=cfg.infrastructure.server, inline_ssh_env=True) as connection:
        cemetery_dir = cfg.infrastructure.cemetery_experiments_dir
        connection.run(f"mkdir -p {cemetery_dir}")

        experiment_dir = f"{cemetery_dir}/{experiment_branch_name}"
        if connection.run(f"test -d {experiment_dir}", warn=True).failed:
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {cfg.infrastructure.git.remote_url} {experiment_dir}"
            )
        else:
            print(
                f"Experiment {experiment_branch_name} already exists. Skipping."
            )
        # if "NEPTUNE_API_TOKEN" in os.environ:
        #     connection.config["run"]["env"]["NEPTUNE_API_TOKEN"] = os.environ[
        #         "NEPTUNE_API_TOKEN"
        #     ]

        try:
            connection.run(f"tmux new -d -s {experiment_branch_name}")
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "cd {experiment_dir}" C-m'
            )
            #TODO create venv
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "source {cfg.infrastructure.venv_path}" C-m'
            )
            connection.run(
                f'tmux send -t {experiment_branch_name}.0 "sbatch exp.job" C-m'
            )

            output = connection.run(
                f"tmux capture-pane -pt {experiment_branch_name}.0", hide=True
            ).stdout

            # Parse job ID from sbatch output
            # match = re.search(r"Submitted batch job (\d+)", result.stdout)
            # if not match:
            #     raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
            # job_id = match.group(1)
            # print(f"Submitted job {job_id}")

            while True:
                squeue_result = connection.run(f"squeue -j {job_id} -h -o '%T'", hide=True)
                state = squeue_result.stdout.strip()
                if state == "R":  # Running
                    print(f"Job {job_id} is running")
                    break
                elif state == "PD":  # Pending
                    print(f"Job {job_id} is pending...")
                elif state == "":
                    raise RuntimeError(f"Job {job_id} disappeared from queue")
                else:
                    print(f"Job {job_id} state: {state}")
                time.sleep(1)

            # # logger.info("=" * 38 + "TMUX" + "=" * 38)
            # time.sleep(3)
            # output = connection.run(
            #     f"tmux capture-pane -t {experiment_branch_name}.0 -p", hide=True
            # ).stdout
            # logger.info(output)
        except Exception as e:
            print("Exception while running an experiment: ", e)


if __name__ == "__main__":
    submit_experiment()
