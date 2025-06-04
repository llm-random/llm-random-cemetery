# """
# Test comparing the results of a job from scratch and from a checkpoint with different world sizes (ascending)
# """

# import logging
# import shutil
# import hydra
# import os
# import pathlib

# import numpy as np

# from model import (
#     get_metric_logger,
#     run,
# )
# import model

# logger = logging.getLogger(__name__)


# def run_on_condition(func, predicate):
#     """
#     Mock a function so that it only runs if `predicate` is True.
#     For example, if `predicate` is `lambda count: count == 1`,
#     the original function will only run the first time it's called.
#     """

#     def side_effect(*args, **kwargs):
#         side_effect.call_count += 1
#         if predicate(side_effect.call_count):
#             return func(*args, **kwargs)
#         return None

#     side_effect.call_count = 0
#     return side_effect


# def cleanup(path):
#     if os.path.exists(path):
#         shutil.rmtree(path)


# @hydra.main(
#     version_base=None,
#     config_path=".",
#     config_name=pathlib.Path(__file__).stem,
# )
# def main(job_config):
#     # We do not want to init and destroy process_group multiple times, as it
#     # leads to race condition (first process group is not destroyed yet, while another started second)
#     # model.distributed_setup = run_on_condition(
#     #     model.distributed_setup, lambda x: x == 1
#     # )
#     # model.cleanup = run_on_condition(model.cleanup, lambda x: x > 1)

#     if os.environ["RANK"] == "0":
#         cleanup(job_config.checkpoint_config.path)

#     try:
#         hydra_config = hydra.utils.HydraConfig.get()

#         if os.environ["RANK"] == "0":
#             run(job_config, hydra_config)
#             record_metric_logger = get_metric_logger()
#             first_data = record_metric_logger.data

#             record_metric_logger.clear()
#             logger.info("First run done!")

#         ##here
#         os.environ["WORLD_SIZE"] = "1"
#         os.environ["LOCAL_RANK"] = "0"
#         # os.environ["RANK"] = "0"

#         if int(os.environ["RANK"]) > 0:
#             return
#         # else:
#         run(job_config, hydra_config)

#         target = {key: value[-5:] for key, value in first_data.items()}

#         # assert target == record_metric_logger.data
#         if os.environ["RANK"] == "0":
#             skip_keys = ["batch"]
#             for key in target:
#                 if key not in skip_keys:

#                     assert np.allclose(
#                         target[key], record_metric_logger.data[key], rtol=1e-5, atol=1e-8
#                     ), f"Mismatch for key '{key}'"

#         logger.info("Done!")
#     finally:
#         if os.environ["RANK"] == "0":
#             cleanup(job_config.checkpoint_config.path)


# if __name__ == "__main__":
#     main()


"""
Test comparing the results of a job from scratch and from a checkpoint with different world sizes (ascending)
"""

import logging
import shutil
import hydra
import os

import numpy as np

from model import (
    get_metric_logger,
    run,
)

logger = logging.getLogger(__name__)


def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def first_elements(list_of_tuples):
    return list(zip(*list_of_tuples))[0]


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="test_checkpoint_loading",
)
def main(job_config):
    hydra_config = hydra.utils.HydraConfig.get()

    if os.environ["RANK"] == "0":
        cleanup(job_config.checkpoint_config.path)

        try:
            saved_world_size = os.environ["WORLD_SIZE"]
            saved_local_rank = os.environ["LOCAL_RANK"]
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"

            run(job_config, hydra_config)
            record_metric_logger = get_metric_logger()
            first_data = record_metric_logger.data

            record_metric_logger.clear()

            os.environ["WORLD_SIZE"] = saved_world_size
            os.environ["LOCAL_RANK"] = saved_local_rank

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
                    target_values = first_elements(target[key])
                    from_checkpoint_values = first_elements(
                        record_metric_logger.data[key]
                    )

                    assert np.allclose(
                        target_values, from_checkpoint_values, rtol=1e-5, atol=1e-8
                    ), f"Mismatch for key '{key}'"

            logger.info(f"Done from RANK:{os.environ['RANK']}!")
        finally:
            cleanup(job_config.checkpoint_config.path)
    else:
        run(job_config, hydra_config)
        logger.info(f"Done from RANK:{os.environ['RANK']}!")


if __name__ == "__main__":
    main()
