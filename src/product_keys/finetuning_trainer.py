from attr import define, field
import logging 
import torch

from src.core.trainer import Trainer

logger = logging.getLogger(__name__)
from src.product_keys.model_sequence_classifiaction import ModelSequenceClassification

# for now focus solely on sst2
HIDDEN_SIZE = 128256
SST2_LABELS: int = 2

def create_classifier_model(model: torch.nn.Module,
                            device: torch.device) -> torch.nn.Module:
    logger.info("Printing model shapes...")
    for name, layer in model.named_modules():
        logger.info(f"Layer name: {name}")

        if hasattr(layer, 'weight') and layer.weight is not None:
            logger.info(f"Layer: {name} | Size: {layer.weight.shape}")
    
    return ModelSequenceClassification(model, hidden_size=HIDDEN_SIZE, num_labels=SST2_LABELS).to(device)


@define(slots=False)
class FinetuningTrainer(Trainer):
    freeze_backbone: bool = field(default=False)
    trainable_modules: list = field(factory=list) 

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        if self.freeze_backbone:
            self._freeze_model_layers()

        self.model = create_classifier_model(self.model, self.device)
        
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

