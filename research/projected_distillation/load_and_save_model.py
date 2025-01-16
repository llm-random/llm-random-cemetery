import torch

TRANSFER_PARAMS = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",
]

CAST_PROJECTED_PARAMS_NAME_PARTS = [
    (".output_projection.output_projection.", ".output_projection."),
    (".input_projection.input_projection.", ".input_projection.")
]


def load_projected_weights(model:torch.nn.Module, projected_weights):
    print("----------------replace with new values----------------") #dev
    for name, params in model.named_parameters():
        prj_params = projected_weights.get(name)
        for e in CAST_PROJECTED_PARAMS_NAME_PARTS:
            if e[0] in name:
                name = name.replace(e[0], e[1])
                print("replaced name", name) #dev
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            print(name) #dev
            params.data.copy_(prj_params)
    print("--------------end------------------") #dev
    # raise
"""
encoder.blocks.block_0.block.residual_attention.layer.pre_norm.weight requires_grad: True
encoder.blocks.block_0.block.residual_attention.layer.pre_norm.bias requires_grad: True
encoder.blocks.block_0.block.residual_attention.layer.attention.input_projection.input_projection_p11.weight requires_grad: True
encoder.blocks.block_0.block.residual_attention.layer.attention.input_projection.input_projection.weight requires_grad: False
encoder.blocks.block_0.block.residual_attention.layer.attention.input_projection.input_projection_p12.weight requires_grad: True
encoder.blocks.block_0.block.residual_attention.layer.attention.output_projection.output_projection_p21.weight requires_grad: True
encoder.blocks.block_0.block.residual_attention.layer.attention.output_projection.output_projection.weight requires_grad: False
encoder.blocks.block_0.block.residual_attention.layer.attention.output_projection.output_projection_p22.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.pre_norm.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.pre_norm.bias requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.bias requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.weight requires_grad: False
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.bias requires_grad: False
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.bias requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.bias requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu.weight requires_grad: False
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu.bias requires_grad: False
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight requires_grad: True
encoder.blocks.block_0.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.bias requires_grad: True
"""