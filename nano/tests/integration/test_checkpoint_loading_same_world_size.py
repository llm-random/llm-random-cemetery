"""
Test comparing the results of a job from scratch and from a checkpoint
"""

import logging
import shutil
import hydra
import os

import numpy as np

from model import get_metric_logger, run, setup_enviroment
import model

logger = logging.getLogger(__name__)


def run_on_condition(func, predicate):
    """
    Mock a function so that it only runs if `predicate` is True.
    For example, if `predicate` is `lambda count: count == 1`,
    the original function will only run the first time it's called.
    """

    def side_effect(*args, **kwargs):
        side_effect.call_count += 1
        if predicate(side_effect.call_count):
            return func(*args, **kwargs)
        return None

    side_effect.call_count = 0
    return side_effect


def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="test_checkpoint_loading",
)
def main(job_config):
    # We do not want to init and destroy process_group multiple times, as it
    # leads to race condition (first process group is not destroyed yet, while another started second)
    model.distributed_setup = run_on_condition(
        model.distributed_setup, lambda x: x == 1
    )
    model.cleanup = run_on_condition(model.cleanup, lambda x: x > 1)
    setup_enviroment()

    if os.environ["RANK"] == "0":
        cleanup(job_config.checkpoint_config.path)

    try:
        hydra_config = hydra.utils.HydraConfig.get()
        run(job_config, hydra_config)
        record_metric_logger = get_metric_logger()
        first_data = record_metric_logger.data

        record_metric_logger.clear()
        logger.info(f"First run done! -> rank {os.environ['RANK']}")

        run(job_config, hydra_config)

        target = {key: value[-5:] for key, value in first_data.items()}

        skip_keys = ["batch"]
        for key in target:
            if key not in skip_keys:
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
