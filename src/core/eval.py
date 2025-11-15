from lm_eval import evaluator
from attr import define
from typing import Optional
import json
import os

from src.core.metric_loggers import MetricLogger
from src.core.checkpointing import get_full_checkpoint_path


@define(slots=False)
class Evaluator:
    checkpoint_path: str
    tokenizer: str
    tasks: list[str]
    limit: Optional[int]
    device: str
    metric_logger: MetricLogger

    def eval(self):
        full_ckpt_path = get_full_checkpoint_path(self.checkpoint_path)
        eval_model_args = (
            f"pretrained={full_ckpt_path}," f"tokenizer={self.tokenizer}"
        )

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=eval_model_args,
            tasks=list(self.tasks),
            limit=self.limit,
            device=self.device,
            log_samples=False,
        )

        with open(os.path.join(full_ckpt_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.log_eval(results)

    def log_eval(self, eval_results: dict):
        """Log evaluation results to Neptune."""
        for task_name, metrics in eval_results["results"].items():
            for metric_name, value in metrics.items():
                clean_metric_name = metric_name.replace(",none", "")
                self.metric_logger.run[f"eval/{task_name}/{clean_metric_name}"] = value

        self.metric_logger.run["eval/limit"] = eval_results["config"]["limit"]
