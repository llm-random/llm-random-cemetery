import argparse
import os
from lizrd.grid.infrastructure import get_machine_backend
from lizrd.support.code_versioning import version_code

from contextlib import contextmanager
import copy
import getpass
from typing import Generator
from fabric import Connection
import paramiko.ssh_exception

from lizrd.support.code_versioning import version_code

CEMETERY_REPO_URL = "git@github.com:llm-random/llm-random-cemetery.git"  # TODO(crewtool) move to constants

_SSH_HOSTS_TO_PASSPHRASES = {}


@contextmanager
def ConnectWithPassphrase(*args, **kwargs) -> Generator[Connection, None, None]:
    """Connect to a remote host using a passphrase if the key is encrypted. The passphrase is preserved for subsequent connections to the same host."""
    try:
        connection = Connection(*args, **kwargs)
        connection.run('echo "Connection successful."')
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


def submit_experiment(
    hostname, experiment_branch_name, experiment_config_path, clone_only
):
    print("ELOOO")
    if experiment_branch_name is None:
        experiment_branch_name = version_code(experiment_config_path)

    print("XDDDDDDD")
    with ConnectWithPassphrase(hostname) as connection:
        result = connection.run("uname -n", hide=True)
        node = result.stdout.strip()
        cluster = get_machine_backend(node)

        cemetery_dir = cluster.get_cemetery_directory()
        connection.run(f"mkdir -p {cemetery_dir}")
        experiment_directory = f"{cemetery_dir}/{experiment_branch_name}"

        if "NEPTUNE_API_TOKEN" in os.environ:
            connection.config["run"]["env"] = {
                "NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"]
            }
        if "WANDB_API_KEY" in os.environ:
            connection.config["run"]["env"] = {
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
            }

        if connection.run(f"test -d {experiment_directory}", warn=True).failed:
            connection.run(
                f"git clone --depth 1 -b {experiment_branch_name} {CEMETERY_REPO_URL} {experiment_directory}"
            )
            print(
                f"Cloned <<'{experiment_branch_name}'>> to ##*{experiment_directory}*##"
            )
        else:
            print(
                f"Experiment <<'{experiment_branch_name}'>> already exists on {node} at ##*{experiment_directory}*##. Skipping."
            )

        connection.run(f"chmod +x {experiment_directory}/run_experiment.sh")
        if not clone_only:
            connection.run(f"cd {experiment_directory} && ./run_experiment.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Hostname as in ~/.ssh/config",
        required=True,
    )
    parser.add_argument(
        "--clone_only",
        action="store_true",
        help="Only clone the experiment, do not run it.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment",
        type=str,
        help="[Optional] Name of the existing branch on cemetery to run experiment.",
    )
    group.add_argument(
        "--config",
        type=str,
        help="[Optional] Path to experiment config file.",
    )

    args = parser.parse_args()
    submit_experiment(args.host, args.experiment, args.config, args.clone_only)
