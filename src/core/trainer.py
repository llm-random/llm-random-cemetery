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
    # Master on/off switch for the whole guard. Currently defaulted to False
    # ("off for now"): the mechanism turned out to be masking a duration-
    # dependent optimal-LR issue rather than fixing a real bug, and its
    # forced-through pushes were themselves adding a slow upward drift to the
    # loss. Kept in the code (rather than removed) since it's still a
    # reasonable safety net for genuine one-off outlier batches - flip this
    # back on per-experiment via config once that's wanted again.
    grad_norm_spike_guard_enabled: bool = field(default=False, kw_only=True)
    grad_norm_spike_ema_decay: float = field(default=0.99, kw_only=True)
    grad_norm_spike_threshold_multiplier: float = field(default=10.0, kw_only=True)
    # Safety valve: if the EMA baseline is stale (e.g. seeded during warmup when
    # grad norms are naturally tiny) and the true grad_norm regime then shifts
    # upward for good, every subsequent step would exceed the threshold forever
    # and the EMA would never be allowed to update - permanently stalling
    # training (100% of updates skipped). After this many *consecutive* skips,
    # force the update through (and let the EMA jump to the new regime) instead
    # of skipping again.
    grad_norm_spike_max_consecutive_skips: int = field(default=20, kw_only=True)
    # When the safety valve above fires, we're deliberately applying an update
    # we've flagged as anomalous. Its magnitude is already bounded by
    # gradient_clipping (clip_gradient() clips in-place before we ever look at
    # the spike condition), but its *direction* is untrusted - it could still
    # be dominated by a few outlier parameters and knock Adam's per-parameter
    # state into a bad basin. So we clip forced-through updates to a much
    # smaller fraction of the normal clip, trading a smaller step for safety;
    # normal (non-flagged) updates are unaffected and use the full
    # gradient_clipping norm as usual.
    grad_norm_spike_escape_clip_fraction: float = field(default=0.1, kw_only=True)
    only_compress_model_gradient_clipping: bool

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.model.parameters()).device
        self.loss_interval_100 = 0.0
        self.grad_norm_ema = None
        self.consecutive_grad_norm_skips = 0

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
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.metric_logger.set_tokens(self.processed_tokens)
            self.model.train()
            loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()

            self.metric_logger.flush()

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

    def _apply_escape_clip(self):
        """Extra-tight clip applied only to forced-through grad_norm-spike
        updates (see grad_norm_spike_max_consecutive_skips /
        grad_norm_spike_escape_clip_fraction). Gradients have already been
        clipped once to gradient_clipping by clip_gradient(); this clips them
        again to a much smaller norm so a forced-through update - whose
        direction we don't trust, only its magnitude is bounded - can only
        nudge the model a little rather than take a full-strength step that
        could knock Adam's per-parameter state into a bad basin."""
        if self.gradient_clipping is None:
            return
        escape_norm = self.grad_norm_spike_escape_clip_fraction * self.gradient_clipping
        if isinstance(self.model, FSDP):
            self.model.clip_grad_norm_(escape_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), escape_norm)

    def _sync_decision_across_ranks(self, decision: bool) -> bool:
        """Force a boolean control-flow decision to be identical on every
        rank before branching on it. train() uses grad_norm-spike booleans to
        decide whether to call an *additional* collective (_apply_escape_clip)
        on top of the always-called clip_gradient(). If that decision were
        ever to disagree across ranks - e.g. a one-ULP float difference right
        at the threshold boundary, or a NaN/Inf appearing asymmetrically -
        some ranks would call a different number of collectives than others,
        which deadlocks FSDP's all-gather/all-reduce ops permanently (this is
        the textbook cause of an NCCL watchdog collective-timeout hang, and
        the way you find out is a training run silently wasting hours before
        timing out). Uses MAX so any disagreement resolves to the safer,
        more-conservative outcome (skip/clip) rather than silently letting a
        subset of ranks proceed differently."""
        if not dist.is_initialized():
            return decision
        flag = torch.tensor(1 if decision else 0, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

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

    def _update_grad_norm_ema(self, grad_norm_value: float, winsorize: bool = False):
        """Update the running grad_norm EMA. When winsorize=True (used on the
        skipped-update path), the value fed into the EMA is capped at
        threshold_multiplier * current EMA. This lets the EMA keep drifting
        upward when grad_norm undergoes a genuine, sustained regime shift
        (instead of staying frozen and causing every future step to be
        skipped, see grad_norm_spike_max_consecutive_skips), while still
        preventing a single one-off outlier from blowing the EMA up in one
        shot."""
        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm_value
            return

        effective_value = grad_norm_value
        if winsorize:
            cap = self.grad_norm_spike_threshold_multiplier * self.grad_norm_ema
            effective_value = min(grad_norm_value, cap)

        self.grad_norm_ema = (
            self.grad_norm_spike_ema_decay * self.grad_norm_ema
            + (1 - self.grad_norm_spike_ema_decay) * effective_value
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
