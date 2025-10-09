class RepeatedScheduler:
    """
    Wraps a learning rate scheduler and repeats it for a specified number of cycles.
    After each cycle completes, the scheduler is reset to its initial state.
    """

    def __init__(self, base_scheduler, num_cycles):
        """
        Args:
            base_scheduler: A PyTorch LR scheduler instance to repeat
            num_cycles: Number of times to repeat the scheduler cycle
        """
        self.base_scheduler = base_scheduler
        self.num_cycles = num_cycles
        self.optimizer = base_scheduler.optimizer

        # Store initial state
        self.initial_state = base_scheduler.state_dict()

        # Track current cycle and total steps
        self.current_cycle = 0
        self.total_steps = 0

        # Get the cycle length from the base scheduler
        # This assumes the scheduler has a total_iters or T_max attribute
        if hasattr(base_scheduler, 'total_iters'):
            self.cycle_length = base_scheduler.total_iters
        elif hasattr(base_scheduler, 'T_max'):
            self.cycle_length = base_scheduler.T_max
        else:
            # Try to infer from milestones for SequentialLR
            if hasattr(base_scheduler, 'milestones') and base_scheduler.milestones:
                self.cycle_length = max(base_scheduler.milestones) + 1
            else:
                raise ValueError("Unable to determine cycle length from scheduler")

    def step(self):
        """Take a step with the scheduler, resetting if cycle is complete"""
        self.base_scheduler.step()
        self.total_steps += 1

        # Check if we've completed a cycle
        if self.total_steps % self.cycle_length == 0:
            self.current_cycle += 1

            # Reset scheduler if we haven't exhausted repeats
            if self.current_cycle < self.num_cycles:
                self.base_scheduler.load_state_dict(self.initial_state)

    def get_last_lr(self):
        """Get the last computed learning rate"""
        return self.base_scheduler.get_last_lr()

    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_cycle': self.current_cycle,
            'total_steps': self.total_steps,
            'cycle_length': self.cycle_length,
            'num_cycles': self.num_cycles,
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_cycle = state_dict['current_cycle']
        self.total_steps = state_dict['total_steps']
        self.cycle_length = state_dict['cycle_length']
        self.num_cycles = state_dict['num_cycles']