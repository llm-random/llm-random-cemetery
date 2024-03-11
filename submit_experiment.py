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

# env.forward_agent = True
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


def submit_experiment(experiment_branch_name, experiment_config_path="hej_test3.yaml"):
    version_code(experiment_branch_name, experiment_config_path)
    # pull_branch_remotely(connection, experiment_branch_name, )
    # connection = "XD"
    branch = experiment_branch_name
    repo_url = "git@github.com:llm-random/llm-random-cemetery.git"
    experiment_directory = f"~/experiments_cemetery/{branch}"

    with ConnectWithPassphrase("entropia") as connection:
        connection.run(
            f"git clone --depth 1 -b {branch} {repo_url} {experiment_directory}"
        )
        connection.run("pwd")
        connection.run(f"cd {experiment_directory} && pwd")
        connection.run(f"cd {experiment_directory} && ./run_experiment.sh")

    # with connection.cd(f"{experiment_directory}/llm-random"):
    #     connection.run('tmux new-session -d -s costam bash')
    # run_grid_remotely()


if __name__ == "__main__":
    submit_experiment("test_send8")
