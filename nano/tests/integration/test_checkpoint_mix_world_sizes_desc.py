"""
Test comparing the results of a job from scratch and from a checkpoint with different world sizes (descending)
"""

import logging
import shutil
import hydra
import os
import pathlib

import numpy as np

from model import (
    get_metric_logger,
    run,
)

logger = logging.getLogger(__name__)


def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="test_checkpoint_loading",
)
def main(job_config):
    if os.environ["RANK"] == "0":
        cleanup(job_config.checkpoint_config.path)

    try:
        hydra_config = hydra.utils.HydraConfig.get()

        run(job_config, hydra_config)
        record_metric_logger = get_metric_logger()
        first_data = record_metric_logger.data

        record_metric_logger.clear()

        if int(os.environ["RANK"]) > 0:
            return

        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

        run(job_config, hydra_config)

        target = {key: value[-5:] for key, value in first_data.items()}

        check_keys = [
            "step",
            "steps/train/loss",
            "steps/train/lr",
            "steps/train/grad_norm",
            "steps/train/processed_tokens",
            "tokens/train/loss",
            "tokens/lr",
            "tokens/train/grad_norm",
        ]

        for key in target:
            if key in check_keys:
                target_values = list(zip(*target[key]))[0]
                from_checkpoint_values = list(zip(*record_metric_logger.data[key]))[0]

                assert np.allclose(
                    target_values, from_checkpoint_values, rtol=1e-5, atol=1e-8
                ), f"Mismatch for key '{key}'"

        logger.info("Done!")
    finally:
        if os.environ["RANK"] == "0":
            cleanup(job_config.checkpoint_config.path)

if __name__ == "__main__":
    main()
