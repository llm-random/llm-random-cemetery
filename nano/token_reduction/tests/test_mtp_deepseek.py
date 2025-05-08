import unittest
from unittest.mock import patch
import torch

from hydra import initialize, compose
from hydra.utils import instantiate


from token_reduction.model import (
    LLM_DeepSeekMTP,
    TrainerDeepSeekMTP,
    get_deepseek_embedding,
)

from model import (
    RecorderLogger,
    load_training_state,
    run,
)

class TestSimpleRun(unittest.TestCase):
    @patch("model.get_metric_logger", return_value=RecorderLogger())
    def test_simple_mtp_deepseek(self, get_metric_logger):
        target_losses_dropping = [
            (11.903800010681152, 0),
            (11.852548599243164, 1),
            (11.754332542419434, 2),
            (11.814672470092773, 3),
            (11.735774993896484, 4),
            (11.87263298034668, 5),
            (11.693036079406738, 6),
            (11.756917953491211, 7),
            (11.757512092590332, 8),
            (11.67790412902832, 9),
        ]
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test_mtp_deepseek", overrides=[])
            training_state = load_training_state(cfg.checkpoint_config)
            metric_logger = get_metric_logger(
                metric_logger_config=instantiate(cfg.metric_logger, _convert_="all"),
                neptune_run_id=training_state["run_id"],
            )
            run(cfg)

            self.assertListEqual(
                target_losses_dropping, metric_logger.data["steps/train/loss"]
            )


if __name__ == "__main__":
    unittest.main()