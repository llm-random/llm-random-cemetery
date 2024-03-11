import argparse
import datetime
from lizrd.support.code_versioning import version_code
import subprocess

from contextlib import contextmanager
import copy
import getpass
from typing import Generator
from fabric import Connection
import paramiko.ssh_exception

# from fabric. import env
from lizrd.support.code_versioning import version_code
import subprocess

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


def clone_branch_remotely(connection, branch, repo_url, experiment_directory):
    connection.run(f"git clone --depth 1 -b {branch} {repo_url} {experiment_directory}")


def submit_experiment(hostname, experiment_branch_name, experiment_config_path):
    assert experiment_branch_name is not None or experiment_config_path is not None, "You need to provide either experiment_branch_name or experiment_config_path."

    if experiment_branch_name is None:
        #TODO(crewtool) use name from job_name from config file
        experiment_branch_name = f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        version_code(experiment_branch_name, experiment_config_path)

    repo_url = "git@github.com:llm-random/llm-random-cemetery.git" #TODO(crewtool) move to constants
    experiment_directory = f"~/experiments_cemetery/{experiment_branch_name}" #TODO make sexperiments_cemetery exists

    with ConnectWithPassphrase(hostname) as connection:
        connection.run(
            f"git clone --depth 1 -b {experiment_branch_name} {repo_url} {experiment_directory}"
        )
        connection.run(f"chmod +x {experiment_directory}/run_experiment.sh")
        connection.run(f"cd {experiment_directory} && ./run_experiment.sh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        help="Hostname as in ~/.ssh/config",
        required=True,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--existing_experiment_branch", type=str, help="[Optional] Name of the existing branch on cemetery to run experiment."
    )
    group.add_argument(
        "--config",
        type=str,
        help="[Optional] Path to experiment config file.",
    )
    args = parser.parse_args()
    hostname = args.host

    submit_experiment(args.host, args.existing_experiment_branch, args.config)
