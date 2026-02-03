import random
from omegaconf import OmegaConf
import platform
import os
import re


def get_cluster_name(hostname=None, username=None) -> str:
    if "LLMRANDOM_CLUSTER" in os.environ:
        return os.environ.get("LLMRANDOM_CLUSTER")

    if hostname is None:
        hostname = platform.uname().node

    conf = OmegaConf.load("configs/clusters.yaml")

    for cluster in conf:
        if "hosts" in cluster:
            for host_pattern in cluster.hosts:
                if re.match(host_pattern, hostname):
                    return cluster.name

    return "default"

def get_common_env_variables_export():
    variables = ["WANDB_API_KEY", "HF_TOKEN"]
    res = []
    for var in variables:
        if var in os.environ:
            res.append(f'export {var}="{os.environ[var]}"')
        else:
            print(f"Warning: {var} not found in environment variables. This might lead to issues.")
            res.append(f'# export {var}=... <- potentially missing variable')
    return '\n'.join(res)



OmegaConf.register_new_resolver("__llmrandom_cluster_config", get_cluster_name)

OmegaConf.register_new_resolver("random_seed", lambda: random.randint(0, 100000))

OmegaConf.register_new_resolver("eval", eval)

OmegaConf.register_new_resolver("export_env_variables", get_common_env_variables_export)