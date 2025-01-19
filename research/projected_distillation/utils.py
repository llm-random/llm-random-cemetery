import numpy as np

FREEZE_PARAMS_REGULES = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    "embedding_layer.layers.0.embedding.weight", #TE
    "embedding_layer.layers.1.layer.weight", #PE

    ".pre_norm.", # Layer norm

    # "head.head.weight", #Head
]

def freeze_projected_params(model):
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
    return model 


PROJECTIONS_1_1 = [
    ""
]

PROJECTIONS_1_4 = [
    ""
]

PROJECTIONS_1_3 = [
    ""
]

Project_PARAMS_NAME_PARTS

def initialize_projections(model:torch.nn.Module, init_type=None):
    if not init_type:
        return
    print("------------------------------init projections------------------------") #dev
    for name, params in model.named_parameters():
        for e in CAST_PROJECTED_PARAMS_NAME_PARTS:
            if e[0] in name:
                name = name.replace(e[0], e[1])
                # print("replaced name", name) #dev
        prj_params = projected_weights.get(name)
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            params.data.copy_(prj_params)
            print(f"REPLACED: {name}")
    print("------------------------------init projections end------------------------") #dev