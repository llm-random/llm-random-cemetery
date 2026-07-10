import os
import time
from attr import define, field
import torch
import torch.nn.functional as F
from typing import List, Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.projected_compression.compression import finalize_projection_weights
from src.core.conversion_to_hf import save_to_llama_3_hf
import torch.distributed.checkpoint as dcp
import logging

from src.core.checkpointing import (
    TrainingState,
    get_full_checkpoint_path,
    save_training_state,
    step_checkpoint_path,
)
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger
from src.core.utils import cast_state_dict_to_tensors, create_batch_fingerprint

logger = logging.getLogger(__name__)


@define(slots=False)
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation_steps: int
    training_state: dict
    n_steps: int
    train_dataloader: IterableDataset
    eval_dataloader: IterableDataset
    metric_logger: MetricLogger
    eval_interval: int
    n_eval_steps: int
    gradient_clipping: Optional[float]
    checkpoint: Optional[dict]
    learning_rate: float
    weight_decay: float
    distributed: Optional[dict]
    per_dataset_dataloaders: Optional[List] = field(default=None, kw_only=True)  # list of (name, eval_dl)
    # Grad-norm-spike guard: skip the optimizer update (but keep the gradients
    # out of future accumulation) when a step's pre-clip grad_norm is
    # anomalously high relative to a running EMA of recent grad norms. This
    # protects Adam's optimizer state from being knocked into a bad basin by a
    # directionally-bad update, which is a known cause of the multi-hundred-step
    # "ringing" loss spikes seen after aggressive pruning/distillation.
    # grad_norm (pre-clip) is a much better-separated spike signal than loss: a
    # handful of outlier-token gradients can blow up the parameter L2 norm while
    # barely moving the batch-averaged loss (loss is diluted over ~500K
    # tokens/batch; gradients from a few bad tokens are not). kw_only so
    # subclasses (e.g. TrainerDistillation) can keep adding required fields
    # after it.
    grad_norm_spike_ema_decay: float = field(default=0.99, kw_only=True)
    grad_norm_spike_threshold_multiplier: float = field(default=10.0, kw_only=True)

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self.loss_interval_100 = 0.0
        self.grad_norm_ema = None

        if self.eval_dataloader is not None and hasattr(
            self.eval_dataloader, "__iter__"
        ):
            self.eval_iterator = iter(self.eval_dataloader)
        self.step = self.start_step - 1

        if self.per_dataset_dataloaders:
            self.per_dataset_eval_iterators = [
                (name, iter(dl))
                for name, dl in self.per_dataset_dataloaders
            ]
        else:
            self.per_dataset_eval_iterators = []

        if self.start_step > 0:
            n_skip_eval_batches = (
                (self.start_step - 1) // self.eval_interval * self.n_eval_steps
            )
            logger.debug(f"Skipping {n_skip_eval_batches} eval batches")
            for _ in range(n_skip_eval_batches):
                next(self.eval_iterator)
            for _, it in self.per_dataset_eval_iterators:
                for _ in range(n_skip_eval_batches):
                    next(it)

        self.loss_averaged_100 = AveMetric(100, "100/train/loss")
        self.time_diff_averaged_100 = AveDiffMetric(100, "100/time", time.time())

    @property
    def _should_evaluate(self) -> bool:
        return (
            self.eval_interval > 0
            and self.step % self.eval_interval == 0
            and self.step != 0
        )

    @property
    def _should_log_eval_input(self) -> bool:
        return self.step % (self.eval_interval * 100) == 0

    @property
    def _should_save_checkpoint(self) -> bool:
        return (
            self.checkpoint.save.interval > 0
            and (self.step) % self.checkpoint.save.interval == 0
            and self.step != 0
            and self.checkpoint.save.path is not None
        )

    @property
    def _should_save_final_checkpoint(self) -> bool:
        return (
            not self._should_save_checkpoint  # checkpoint was already saved
            and self.step >= self.n_steps - 1
            and self.checkpoint.save.path is not None
        )

    def train(self):
        # step only advances on a real applied update (not a skipped one), so the
        # LR schedule, checkpoint/eval cadence, and total training budget are all
        # defined in terms of n_steps *real* updates - a skip doesn't shrink that
        # count, it just costs an extra batch pulled from the dataloader.
        train_iterator = iter(self.train_dataloader)
        step = self.start_step

        while step < self.n_steps:
            self.step = step
            self.metric_logger.set_step(step)
            self.metric_logger.set_tokens(self.processed_tokens)
            self.model.train()

            try:
                batch = next(train_iterator)
            except StopIteration:
                logger.warning(
                    f"train_dataloader exhausted at step {step} (target n_steps="
                    f"{self.n_steps}); stopping early."
                )
                break

            loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()
            loss_value = loss.item()
            grad_norm_value = grad_norm.item() if grad_norm is not None else None

            if grad_norm_value is not None and self._is_grad_norm_spike(grad_norm_value):
                self.log_skipped_update(loss_value, grad_norm)
                self.optimizer.zero_grad()
                self.metric_logger.flush()
                continue  # retry this step index with the next batch

            self.log_metrics(loss, grad_norm)
            if grad_norm_value is not None:
                self._update_grad_norm_ema(grad_norm_value)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

            self.metric_logger.flush()

            step += 1

        if self._should_save_final_checkpoint:
            if self.checkpoint.save.type == "nano":
                self.save_checkpoint()
            elif self.checkpoint.save.type == "huggingface":
                # self.model.unshard() # alternative that might not work for a very large > 1gpu memory models
                model_state_dict = self.model.state_dict()
                full_state = cast_state_dict_to_tensors(model_state_dict)

                if os.environ["RANK"] == "0":
                    dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = (
                        self.model.encoder.get_model_dimensions()
                    )

                    save_to_llama_3_hf(  # dev fixed values
                        full_state,
                        save_dir=get_full_checkpoint_path(self.checkpoint.save.path),
                        dmodel=dmodel,
                        dff=dff,
                        n_att_heads=n_att_heads,
                        n_kvatt_heads=n_kvatt_heads,
                        head_dim=head_dim,
                        nlayers=nlayers,
                    )
            elif self.checkpoint.save.type == "pc_finalize":
                self.save_pc_finalized_checkpoint()

    def _preprocess_input(self, batch):  # TODO test it
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()

        return input_ids, target_ids

    def calculate_loss(self, batch):
        def _hack_for_python_garbage_collection(input_ids, target_ids):
            """we want to have no reference to model output while backpropagating to allow torch to free memory,
            so we wrap loss calculation in a function"""
            predicted_ids = self.model(input_ids)

            # Tensors should be on the same device for loss calculation #TODO check
            target_ids = target_ids.to(predicted_ids.device)

            mask_loss = F.cross_entropy(
                predicted_ids.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            loss = mask_loss.mean() / self.gradient_accumulation_steps
            return loss

        losses = []
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input(batch_chunk)
            input_ids = input_ids.to(self.device)
            if self.model.training:
                self._update_processed_tokens(input_ids)

            loss = _hack_for_python_garbage_collection(input_ids, target_ids)
            if self.model.training:
                loss.backward()
            losses.append(loss.item())

        # gloo backend supports only sum reduce operation, therfore we first divide by world size and then sum
        avg_loss = torch.tensor(losses, device=loss.device).sum()
        if dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

        return avg_loss / float(os.environ["WORLD_SIZE"])

    def _log_per_dataset_losses(self, iterators: List, log_prefix: str):
        """Compute and log per-dataset loss for the given iterators."""
        with torch.no_grad():
            for name, iterator in iterators:
                losses = []
                for _ in range(self.n_eval_steps):
                    batch = next(iterator).to(self.device)
                    loss = self.calculate_loss(batch)
                    # Use CE loss when available (distillation trainer stores it as _last_ce_loss)
                    losses.append(getattr(self, "_last_ce_loss", loss).item())
                    self.metric_logger.flush_accumulated_metrics()
                self.metric_logger.log(
                    f"{log_prefix}/{name}/loss",
                    torch.tensor(losses).mean().item(),
                )

    def eval(self):
        self.model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)  # disables heavy logging
        losses = []
        eval_fingerprint = []
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                loss = self.calculate_loss(batch)
                losses.append(loss.item())
                self.metric_logger.flush_accumulated_metrics()
            avg_loss = torch.tensor(losses).mean()
            self.metric_logger.log("eval/loss", avg_loss.item())

        self._log_per_dataset_losses(self.per_dataset_eval_iterators, "eval")

        if self._should_log_eval_input:
            self.metric_logger.log("eval/batch", str(eval_fingerprint))

        self.step = saved_step  # Restore step
        self.metric_logger.set_step(saved_step)

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.model, FSDP):
                return self.model.clip_grad_norm_(self.gradient_clipping)
            else:
                return torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def _is_grad_norm_spike(self, grad_norm_value: float) -> bool:
        """True if the pre-clip grad_norm is more than
        grad_norm_spike_threshold_multiplier times the running EMA of past
        (non-skipped) grad norms. Returns False until the EMA has been seeded
        by at least one real step. grad_norm is a much better-separated spike
        signal than loss (see field comment above), so it's the sole trigger
        for skipping an update."""
        return (
            self.grad_norm_ema is not None
            and grad_norm_value
            > self.grad_norm_spike_threshold_multiplier * self.grad_norm_ema
        )

    def _update_grad_norm_ema(self, grad_norm_value: float):
        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm_value
        else:
            self.grad_norm_ema = (
                self.grad_norm_spike_ema_decay * self.grad_norm_ema
                + (1 - self.grad_norm_spike_ema_decay) * grad_norm_value
            )

    def log_skipped_update(self, loss_value, grad_norm):
        """Called instead of log_metrics when a step's update is skipped due to
        an anomalous grad_norm. Deliberately does NOT log to train/loss (or
        train/total_loss for distillation) or train/grad_norm so the spike
        doesn't pollute the main curves; the raw values are recorded under
        separate train/skipped_* metrics instead, alongside a stdout warning
        for visibility."""
        grad_norm_value = grad_norm.item() if grad_norm is not None else None
        logger.warning(
            f"Skipping optimizer update at step {self.step} (grad_norm spike): "
            f"grad_norm={grad_norm_value} (EMA={self.grad_norm_ema}), "
            f"loss={loss_value:.4f}. "
            f"Gradients discarded; not logged to train/loss or train/grad_norm."
        )
        self.metric_logger.set_tokens(self.processed_tokens)
        self.metric_logger.log("train/update_skipped", 1)
        self.metric_logger.log("train/skipped_loss", loss_value)
        if grad_norm_value is not None:
            self.metric_logger.log("train/skipped_grad_norm", grad_norm_value)
        if self.grad_norm_ema is not None:
            self.metric_logger.log("train/grad_norm_ema", self.grad_norm_ema)

        self.metric_logger.flush_accumulated_metrics()

    def log_metrics(self, loss, grad_norm):
        self.metric_logger.set_tokens(self.processed_tokens)
        self.metric_logger.log("train/loss", loss.item())
        self.metric_logger.log("train/lr", self.scheduler.get_last_lr()[0])
        self.metric_logger.log("train/grad_norm", grad_norm.item())
        self.metric_logger.log("train/update_skipped", 0)

        self.loss_averaged_100.log(self.metric_logger, loss.item())
        self.time_diff_averaged_100.log(self.metric_logger, time.time())

        self.metric_logger.flush_accumulated_metrics()

    def save_checkpoint(self):
        if (
            isinstance(self.model, FSDP)
            or self.model.__module__
            == "torch.distributed.fsdp._fully_shard._fully_shard"
        ):
            # Sharded save
            checkpoint_folder = step_checkpoint_path(
                self.checkpoint.save.path, self.step
            )
            state_dict = {
                "app": TrainingState(self.model, self.optimizer, self.scheduler)
            }
            dcp.save(state_dict, checkpoint_id=checkpoint_folder)
            logger.info(f"Saved sharded model checkpoint in {checkpoint_folder}")
        else:
            # Non-sharded save
            if os.environ["RANK"] == "0":
                checkpoint_folder = step_checkpoint_path(
                    self.checkpoint.save.path, self.step
                )
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_path = f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
                state_to_save = {
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(state_to_save, checkpoint_path)
                logger.info(
                    f"Saved non-sharded model checkpoint in '{checkpoint_path}'"
                )

        if os.environ["RANK"] == "0":
            save_training_state(
                save_config=self.checkpoint.save,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )

    def save_pc_finalized_checkpoint(self):
        with torch.no_grad():
            finalize_projection_weights(self.model)
            model_state_dict = cast_state_dict_to_tensors(self.model.state_dict())

        if os.environ["RANK"] == "0":
            checkpoint_folder = step_checkpoint_path(
                self.checkpoint.save.path, self.step
            )
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_path = (
                f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
            )
            state_to_save = {
                "model": model_state_dict,
                "optim": None,
                "scheduler": None,
            }
            torch.save(state_to_save, checkpoint_path)
            logger.info(
                f"Saved non-sharded Finalized PC model checkpoint in '{checkpoint_path}'"
            )
