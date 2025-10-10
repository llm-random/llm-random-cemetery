from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
    _LRScheduler,
)


class CosineScheduler(SequentialLR):
    """
    Cosine annealing learning rate scheduler with optional warmup.
    Decays learning rate from initial value to final_lr_fraction * base_lr following a cosine curve.

    Uses SequentialLR to compose LinearLR (warmup) and CosineAnnealingLR schedulers.
    """

    def __init__(
        self, optimizer, n_steps, final_lr_fraction=0, warmup_steps=0, last_epoch=-1
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of steps for the schedule
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Number of warmup steps with linear increase (default: 0)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction
        self.warmup_steps = warmup_steps

        # Get base learning rate for eta_min calculation
        optimizer_lr = optimizer.param_groups[0]["lr"]

        schedulers = []
        milestones = []

        if warmup_steps > 0:
            # Warmup phase: linear increase from 0.1 to 1.0
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            schedulers.append(warmup_scheduler)
            milestones.append(warmup_steps)

        # Cosine annealing phase
        cosine_steps = n_steps - warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=final_lr_fraction * optimizer_lr,
        )
        schedulers.append(cosine_scheduler)

        super().__init__(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        """Override step to ignore epoch parameter for compatibility with nested SequentialLR"""
        super().step()


class WSDScheduler(SequentialLR):
    """
    Warmup-Stable-Decay (WSD) scheduler with linear decay.
    - Warmup: Linear increase from 0 to peak_lr
    - Stable: Constant at peak_lr
    - Decay: Linear decrease to final_lr_fraction * base_lr

    Uses SequentialLR to compose LinearLR (warmup), ConstantLR (stable), and LinearLR (decay) schedulers.
    """

    def __init__(
        self,
        optimizer,
        n_steps,
        warmup_fraction=0.1,
        decay_fraction=0.1,
        final_lr_fraction=0,
        warmup_steps=None,
        decay_steps=None,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            n_steps: Total number of training steps
            warmup_fraction: Fraction of n_steps for warmup (default: 0.1)
            decay_fraction: Fraction of n_steps for decay (default: 0.1)
            final_lr_fraction: Final learning rate as a fraction of base_lr (default: 0)
            warmup_steps: Override warmup_fraction with explicit step count (default: None)
            decay_steps: Override decay_fraction with explicit step count (default: None)
            last_epoch: Current step count, -1 means start from beginning (default: -1)
        """
        # Allow explicit step counts to override fractions
        self.n_steps = n_steps
        self.warmup_steps = (
            warmup_steps if warmup_steps is not None else int(n_steps * warmup_fraction)
        )
        self.decay_steps = (
            decay_steps if decay_steps is not None else int(n_steps * decay_fraction)
        )
        self.stable_steps = n_steps - self.warmup_steps - self.decay_steps
        self.final_lr_fraction = final_lr_fraction

        schedulers = []
        milestones = []

        # Warmup phase: linear increase from 0.1 to 1.0
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            schedulers.append(warmup_scheduler)
            milestones.append(self.warmup_steps)

        # Stable phase: constant at peak_lr
        if self.stable_steps > 0:
            stable_scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=self.stable_steps,
            )
            schedulers.append(stable_scheduler)
            milestones.append(self.warmup_steps + self.stable_steps)

        # Decay phase: linear decrease to final_lr_fraction
        if self.decay_steps > 0:
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=final_lr_fraction,
                total_iters=self.decay_steps,
            )
            schedulers.append(decay_scheduler)

        super().__init__(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        """Override step to ignore epoch parameter for compatibility with nested SequentialLR"""
        super().step()


class RepeatedScheduler(_LRScheduler):
    """
    Repeats a base scheduler for multiple cycles.
    After each cycle completes, the scheduler resets to its initial state.

    First cycle can have warmup, subsequent cycles start at peak LR.

    Note: This uses a delegation pattern rather than SequentialLR because we need
    to reset the base scheduler at cycle boundaries, which SequentialLR doesn't support
    (SequentialLR schedulers continue from the optimizer's current LR, not the base LR).
    """

    def __init__(
        self,
        optimizer,
        base_scheduler_factory,
        num_cycles,
        n_steps,
        warmup_steps=0,
        last_epoch=-1,
        **base_scheduler_kwargs
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            base_scheduler_factory: Function/partial that creates a scheduler given (optimizer, n_steps, warmup_steps)
            num_cycles: Number of times to repeat the scheduler
            n_steps: Total number of training steps
            warmup_steps: Warmup steps for first cycle only (default: 0)
            last_epoch: Current step (default: -1)
            **base_scheduler_kwargs: Additional arguments to pass to base_scheduler_factory
        """
        self.num_cycles = num_cycles
        self.n_steps = n_steps
        self.warmup_steps = warmup_steps
        self.base_scheduler_kwargs = base_scheduler_kwargs

        # Calculate steps per cycle
        cycle_steps = (n_steps - warmup_steps) // num_cycles

        # Create base scheduler for first cycle with warmup
        self.base_scheduler = base_scheduler_factory(
            optimizer=optimizer,
            n_steps=cycle_steps + warmup_steps,
            warmup_steps=warmup_steps,
            **base_scheduler_kwargs
        )

        # Store attributes from base scheduler
        if hasattr(self.base_scheduler, "final_lr_fraction"):
            self.final_lr_fraction = self.base_scheduler.final_lr_fraction
        if hasattr(self.base_scheduler, "decay_steps"):
            self.decay_steps = self.base_scheduler.decay_steps
        if hasattr(self.base_scheduler, "stable_steps"):
            self.stable_steps = self.base_scheduler.stable_steps

        self.cycle_steps = cycle_steps
        self.current_cycle = 0
        self.step_in_cycle = 0
        self.base_scheduler_factory = base_scheduler_factory

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Delegate to base scheduler"""
        return self.base_scheduler.get_lr()

    def get_last_lr(self):
        """Delegate to base scheduler"""
        return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        """Step the scheduler, resetting at cycle boundaries"""
        self.base_scheduler.step()
        self.step_in_cycle += 1

        # Check if cycle complete
        if self.step_in_cycle >= self.cycle_steps + (
            self.warmup_steps if self.current_cycle == 0 else 0
        ):
            self.current_cycle += 1
            self.step_in_cycle = 0

            # Reset for next cycle (if not last)
            if self.current_cycle < self.num_cycles:
                # Reset base scheduler with no warmup for subsequent cycles
                self.base_scheduler = self.base_scheduler_factory(
                    optimizer=self.optimizer,
                    n_steps=self.cycle_steps,
                    warmup_steps=0,
                    **self.base_scheduler_kwargs
                )

        self.last_epoch = self.base_scheduler.last_epoch

    def state_dict(self):
        """Return state for checkpointing"""
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_cycle": self.current_cycle,
            "step_in_cycle": self.step_in_cycle,
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self.current_cycle = state_dict["current_cycle"]
        self.step_in_cycle = state_dict["step_in_cycle"]
