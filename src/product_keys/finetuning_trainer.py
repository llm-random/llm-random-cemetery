from attr import define, field

from src.core.trainer import Trainer

@define(slots=False)
class FineTuningTrainer(Trainer):

    def __init__(self):
        super().__init__()





