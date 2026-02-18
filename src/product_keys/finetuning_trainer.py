from attr import define, field
import logging 

from src.core.trainer import Trainer

logger = logging.getLogger(__name__)
from transformers import AutoModelForSequenceClassification

# for now focus solely on sst2

def create_classifier_model(from_pretrained):
    

@define(slots=False)
class FinetuneTrainer(Trainer):
    freeze_backbone: bool = field(default=False)
    
    trainable_modules: list = field(factory=list) 

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.freeze_backbone:
            self._freeze_model_layers()
        
    def _freeze_model_layers(self):
        logger.info("Freezing backbone layers...")
        for name, param in self.model.named_parameters():
            should_train = any(mod in name for mod in self.trainable_modules)
            
            if not should_train:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def save_checkpoint(self):
        logger.info("Saving finetune checkpoint...")
        super().save_checkpoint()

