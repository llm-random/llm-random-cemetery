import numpy as np

FREEZE_PARAMS_REGULES = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    "embedding_layer.layers.0.embedding.weight",
    "embedding_layer._checkpoint_wrapped_module.layers.1.layer.weight",

    # "#dev head.weight",
    # "",
]

def freeze_projected_params(model):
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
    return model 


# PROJECTIONS_1_1 = [
#     ""
# ]

# PROJECTIONS_1_4 = [
#     ""
# ]

# PROJECTIONS_1_3 = [
#     ""
# ]

# def initialize_projectors(model):
