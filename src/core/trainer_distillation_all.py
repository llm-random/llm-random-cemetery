import os
import time
from attr import define
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.projected_compression.compression import finalize_projection_weights
from src.core.conversion_to_hf import save_to_llama_3_hf
import torch.distributed.checkpoint as dcp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from src.core.checkpointing import TrainingState, get_full_checkpoint_path, save_training_state, step_checkpoint_path
from src.core.metric_loggers import AveDiffMetric, AveMetric, MetricLogger
from src.core.utils import cast_state_dict_to_tensors, create_batch_fingerprint

logger = logging.getLogger(__name__)

@define(slots=False)
class TrainerDistillation:
    """
    Trainer for online knowledge distillation from a teacher model to a student model.
    
    Args:
        student_model: The student model to be trained
        teacher_model: The teacher model (frozen) used for distillation
        optimizer: Optimizer for the student model
        scheduler: Learning rate scheduler
        gradient_accumulation_steps: Number of gradient accumulation steps
        training_state: Dictionary containing training state (processed_tokens, next_step)
        n_steps: Total number of training steps
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        metric_logger: Logger for metrics
        eval_interval: Interval for evaluation
        n_eval_steps: Number of evaluation steps
        gradient_clipping: Gradient clipping threshold
        checkpoint: Checkpoint configuration
        learning_rate: Learning rate
        exp_learning_rate: Exponential learning rate
        weight_decay: Weight decay
        distributed: Distributed training configuration
        distillation_alpha: Weight for distillation loss (0.0 to 1.0)
        distillation_temperature: Temperature for softening probability distributions
        distillation_type: Type of distillation ('logits', 'hidden', 'attention', 'combined')
    """
    student_model: torch.nn.Module
    teacher_model: torch.nn.Module
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
    exp_learning_rate: float
    weight_decay: float
    distributed: Optional[dict]
    distillation_alpha: float = 0.5  # Weight for distillation loss
    distillation_temperature: float = 2.0  # Temperature for KL divergence
    distillation_type: str = "logits"  # Type of distillation

    def __attrs_post_init__(self):
        self.processed_tokens = self.training_state["processed_tokens"]
        self.start_step = self.training_state["next_step"]
        self.device = next(self.student_model.parameters()).device
        self.loss_interval_100 = 0.0
        self.eval_iterator = iter(self.eval_dataloader)
        self.step = self.start_step - 1

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if self.start_step > 0:
            n_skip_eval_batches = (
                (self.start_step - 1) // self.eval_interval * self.n_eval_steps
            )
            logger.debug(f"Skipping {n_skip_eval_batches} eval batches")
            for _ in range(n_skip_eval_batches):
                next(self.eval_iterator)

        self.loss_averaged_100 = AveMetric(100, "steps/100/train/loss")
        self.ce_loss_averaged_100 = AveMetric(100, "steps/100/train/ce_loss")
        self.distill_loss_averaged_100 = AveMetric(100, "steps/100/train/distill_loss")
        self.time_diff_averaged_100 = AveDiffMetric(100, "steps/100/time", time.time())

    @property
    def model(self):
        """Alias for student_model to maintain compatibility"""
        return self.student_model

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
            not self._should_save_checkpoint
            and self.step >= self.n_steps - 1
            and self.checkpoint.save.path is not None
        )

    def train(self):
        for step, batch in zip(
            range(self.start_step, self.n_steps), self.train_dataloader
        ):
            self.step = step
            self.metric_logger.set_step(step)
            self.student_model.train()
            
            loss, ce_loss, distill_loss = self.calculate_loss(batch)

            grad_norm = self.clip_gradient()

            self.log_metrics(loss, ce_loss, distill_loss, grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            if self._should_save_checkpoint:
                self.save_checkpoint()

            if self._should_evaluate:
                self.eval()        
        
        if self._should_save_final_checkpoint:
            if self.checkpoint.save.type == "nano":
                self.save_checkpoint()
            elif self.checkpoint.save.type == "huggingface":
                model_state_dict = self.student_model.state_dict()
                full_state = cast_state_dict_to_tensors(model_state_dict)
   
                if os.environ["RANK"] == "0":
                    dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = self.student_model.encoder.get_model_dimensions()

                    save_to_llama_3_hf(
                        full_state, save_dir = get_full_checkpoint_path(self.checkpoint.save.path), 
                        dmodel = dmodel, 
                        dff = dff, 
                        n_att_heads = n_att_heads, 
                        n_kvatt_heads = n_kvatt_heads, 
                        head_dim = head_dim,
                        nlayers = nlayers, 
                    ) 
            elif self.checkpoint.save.type == "pc_finalize":
                self.save_pc_finalized_checkpoint()

    def _preprocess_input(self, batch):
        input_ids = batch[:, :-1].contiguous()
        target_ids = batch[:, 1:].contiguous()
        return input_ids, target_ids

    def compute_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute distillation loss using KL divergence with temperature scaling.
        
        Args:
            student_logits: Logits from student model [batch, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch, seq_len, vocab_size]
        
        Returns:
            distillation_loss: KL divergence loss
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.distillation_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.distillation_temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            student_log_probs.flatten(0, -2),
            teacher_probs.flatten(0, -2),
            reduction="batchmean"
        )
        
        # Scale by temperature^2 to normalize
        return kl_loss * (self.distillation_temperature ** 2)

    def calculate_loss(self, batch):
        def _compute_losses(input_ids, target_ids):
            """Compute both CE loss and distillation loss"""
            # Student forward pass
            student_logits = self.student_model(input_ids)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids)
            
            # Move target_ids to same device as student_logits
            target_ids = target_ids.to(student_logits.device)
            
            # Cross-entropy loss (standard supervised loss)
            ce_loss = F.cross_entropy(
                student_logits.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="mean"
            )
            
            # Distillation loss (KL divergence between student and teacher)
            distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)
            
            # Combined loss
            total_loss = (
                (1.0 - self.distillation_alpha) * ce_loss +
                self.distillation_alpha * distill_loss
            )
            
            total_loss = total_loss / self.gradient_accumulation_steps
            ce_loss = ce_loss / self.gradient_accumulation_steps
            distill_loss = distill_loss / self.gradient_accumulation_steps
            
            return total_loss, ce_loss, distill_loss

        total_losses = []
        ce_losses = []
        distill_losses = []
        
        for batch_chunk in batch.chunk(self.gradient_accumulation_steps):
            input_ids, target_ids = self._preprocess_input(batch_chunk)
            input_ids = input_ids.to(self.device)
            
            if self.student_model.training:
                self._update_processed_tokens(input_ids)

            total_loss, ce_loss, distill_loss = _compute_losses(input_ids, target_ids)
            
            if self.student_model.training:
                total_loss.backward()
            
            total_losses.append(total_loss.item())
            ce_losses.append(ce_loss.item())
            distill_losses.append(distill_loss.item())

        # Average and synchronize across devices
        avg_total_loss = torch.tensor(total_losses, device=self.device).sum()
        avg_ce_loss = torch.tensor(ce_losses, device=self.device).sum()
        avg_distill_loss = torch.tensor(distill_losses, device=self.device).sum()
        
        if dist.is_initialized():
            dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_ce_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(avg_distill_loss, op=dist.ReduceOp.SUM)
        
        world_size = float(os.environ["WORLD_SIZE"])
        return (
            avg_total_loss / world_size,
            avg_ce_loss / world_size,
            avg_distill_loss / world_size
        )

    def eval(self):
        self.student_model.eval()
        saved_step = self.step
        self.metric_logger.set_step(None)
        
        losses = []
        ce_losses = []
        distill_losses = []
        eval_fingerprint = []
        
        with torch.no_grad():
            for _ in range(self.n_eval_steps):
                batch = next(self.eval_iterator)
                batch_fingerprint = create_batch_fingerprint(batch)
                eval_fingerprint.extend(batch_fingerprint)
                batch = batch.to(self.device)
                
                loss, ce_loss, distill_loss = self.calculate_loss(batch)
                losses.append(loss.item())
                ce_losses.append(ce_loss.item())
                distill_losses.append(distill_loss.item())
                
                self.metric_logger.flush_accumulated_metrics(self.step)
            
            avg_loss = torch.tensor(losses).mean()
            avg_ce_loss = torch.tensor(ce_losses).mean()
            avg_distill_loss = torch.tensor(distill_losses).mean()
            
            self.metric_logger.log("steps/eval/loss", self.step, avg_loss.item())
            self.metric_logger.log("steps/eval/ce_loss", self.step, avg_ce_loss.item())
            self.metric_logger.log("steps/eval/distill_loss", self.step, avg_distill_loss.item())
            self.metric_logger.log("tokens/eval/loss", self.processed_tokens, avg_loss.item())
            self.metric_logger.log("tokens/eval/ce_loss", self.processed_tokens, avg_ce_loss.item())
            self.metric_logger.log("tokens/eval/distill_loss", self.processed_tokens, avg_distill_loss.item())

        if self._should_log_eval_input:
            self.metric_logger.log(f"steps/eval/batch", self.step, str(eval_fingerprint))
        
        self.step = saved_step

    def clip_gradient(self):
        if self.gradient_clipping is not None:
            if isinstance(self.student_model, FSDP):
                return self.student_model.clip_grad_norm_(self.gradient_clipping)
            else:
                return torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), self.gradient_clipping
                )

    def _update_processed_tokens(self, batch):
        self.processed_tokens += batch.numel() * int(os.environ["WORLD_SIZE"])

    def log_metrics(self, loss, ce_loss, distill_loss, grad_norm):
        self.metric_logger.log("step", self.step, self.step)
        self.metric_logger.log("steps/train/loss", self.step, loss.item())
        self.metric_logger.log("steps/train/ce_loss", self.step, ce_loss.item())
        self.metric_logger.log("steps/train/distill_loss", self.step, distill_loss.item())
        self.metric_logger.log("steps/train/lr", self.step, self.scheduler.get_last_lr()[0])
        self.metric_logger.log("steps/train/grad_norm", self.step, grad_norm.item())
        self.metric_logger.log("steps/train/processed_tokens", self.step, self.processed_tokens)

        self.metric_logger.log("tokens/train/loss", self.processed_tokens, loss.item())
        self.metric_logger.log("tokens/train/ce_loss", self.processed_tokens, ce_loss.item())
        self.metric_logger.log("tokens/train/distill_loss", self.processed_tokens, distill_loss.item())
        self.metric_logger.log("tokens/lr", self.processed_tokens, self.scheduler.get_last_lr()[0])
        self.metric_logger.log("tokens/train/grad_norm", self.processed_tokens, grad_norm.item())

        self.loss_averaged_100.log(self.metric_logger, self.step, loss.item())
        self.ce_loss_averaged_100.log(self.metric_logger, self.step, ce_loss.item())
        self.distill_loss_averaged_100.log(self.metric_logger, self.step, distill_loss.item())
        self.time_diff_averaged_100.log(self.metric_logger, self.step, time.time())

        self.metric_logger.flush_accumulated_metrics(self.step)

    def save_checkpoint(self):
        # Save student model only
        if isinstance(self.student_model, FSDP) or self.student_model.__module__ == "torch.distributed.fsdp._fully_shard._fully_shard":
            checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)
            state_dict = {
                "app": TrainingState(self.student_model, self.optimizer, self.scheduler)
            }
            dcp.save(state_dict, checkpoint_id=checkpoint_folder)
            logger.info(f"Saved sharded student model checkpoint in {checkpoint_folder}")
        else:
            if os.environ["RANK"] == "0":
                checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)
                os.makedirs(checkpoint_folder, exist_ok=True)
                checkpoint_path = f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
                state_to_save = {
                    "model": (
                        self.student_model.module.state_dict()
                        if type(self.student_model) is DDP
                        else self.student_model.state_dict()
                    ),
                    "optim": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(state_to_save, checkpoint_path)
                logger.info(f"Saved non-sharded student model checkpoint in '{checkpoint_path}'")
            
        if os.environ["RANK"] == "0":
            save_training_state(
                save_config=self.checkpoint.save,
                step=self.step,
                processed_tokens=self.processed_tokens,
                metric_logger=self.metric_logger,
            )

    def save_pc_finalized_checkpoint(self):
        with torch.no_grad():
            finalize_projection_weights(self.student_model)
            model_state_dict = cast_state_dict_to_tensors(self.student_model.state_dict())
            
        if os.environ["RANK"] == "0":
            checkpoint_folder = step_checkpoint_path(self.checkpoint.save.path, self.step)
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_path = f"{checkpoint_folder}/{self.checkpoint.save.model_checkpoint_filename}"
            state_to_save = {
                "model": model_state_dict,
                "optim": None,
                "scheduler": None,
            }
            torch.save(state_to_save, checkpoint_path)
            logger.info(f"Saved non-sharded Finalized PC student model checkpoint in '{checkpoint_path}'")