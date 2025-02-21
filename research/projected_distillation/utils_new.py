import numpy as np
import torch

from lizrd.core.initialization import get_init_weight


FREEZE_PARAMS_REGULES = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    "embedding_layer.layers.0.embedding.weight", #TE
    "embedding_layer.layers.1.projected_layer.pe_layer.weight", #PE

    "head.head.weight", #Head
]

FF_PARAMS_BLACKLIST = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",
]

def freeze_projected_params(model, unprojected_ff):
    frozen_modules = []
    for name, param in model.named_parameters():
        if unprojected_ff and any([reg in name for reg in FF_PARAMS_BLACKLIST]):  # Check if the parameter belongs to layer1
            continue
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules


FREEZE_LN_REGULES = [
    ".pre_norm.", # Layer norm
]

def freeze_ln_params(model):
    frozen_modules = []
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_LN_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules 



PROJECTIONS_1_1 = [
    ".block.residual_attention.layer.attention.input_projection.input_projection_p11.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection_p21.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight",
    "head.head_p.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight", #FF in - 1ff configuration
]

PROJECTIONS_1_1_T = [
    "embedding_layer.layers.0.embedding_p.weight",
    "embedding_layer.layers.1.projected_layer.pe_layer_p.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight", #FF out - 1ff configuration
]

PROJECTIONS_1_3 = [

]

PROJECTIONS_1_3_T = [
    ".block.residual_attention.layer.attention.input_projection.input_projection_p12.weight",
]
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight, shape: torch.Size([512, 256]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight, shape: torch.Size([256, 512]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight, shape: torch.Size([512, 256]) requires_grad: True, cuda:0
# encoder.blocks.block_7.block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight, shape: torch.Size([256, 512]) requires_grad: True, cuda:0



def is_in_partial_list(elemen_name:str, partials_list:list[str]):
    for weight_name in partials_list:
        if weight_name in elemen_name:
            return True
    return False

def print_dict_hierarchy(d, indent=0): #dev debug
    """Recursively print dictionary keys hierarchically."""
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_hierarchy(value, indent + 2)

def svd_init_truncated_sv(weight:torch.Tensor, dm, projected_dm):
    u, s, v = torch.svd(weight)
    assert u.shape[0] == u.shape[1] == s.shape[0] == s.shape[1] == v.shape[0] == v.shape[1] == projected_dm
    
    projection_in = u[:, :dm] # f.e. projected_dm=512, than shape=[512, 256]
    projection_out = v[:dm, :] # f.e. projected_dm=512, than shape=[256, 512]
    projected_weight = s # f.e. projected_dm=512, than shape=[512, 512]
    
    return projection_in, projection_out, projected_weight


def initialize_projections(model:torch.nn.Module, dmodel:int, projected_dmodel:int, projection:torch.Tensor, diagonal=True):
    if projection is None:
        print("No projection initialization")
        return
    
    embedding_layer_tag = "embedding_layer."
    head_tag = "head."
    encode_block_tag = "encoder.blocks.block_"

    model_grouped = {
        embedding_layer_tag: {},
        head_tag: {},
        encode_block_tag: {

        }
    }
    
    for name, params in model.named_parameters():
        if embedding_layer_tag == name[:len(embedding_layer_tag)]:
            model_grouped[embedding_layer_tag][name[len(embedding_layer_tag):]] = params
            continue
        if head_tag == name[:len(head_tag)]:
            model_grouped[head_tag][name[len(head_tag):]] = params
            continue
        if encode_block_tag == name[:len(encode_block_tag)]:
            parsed_name = name[:len(encode_block_tag)].split('.')
            block_number = int(parsed_name[0])
            block_component_name = ".".join(parsed_name[1:])
            model_grouped[encode_block_tag][str(block_number)][block_component_name] = params
            continue
        raise Exception(f"Could not parse model into expected template, unexpected name: name")
        
    print_dict_hierarchy(model_grouped, 3) #dev

    raise Exception(f"Good ending") #dev

    EMBEDDING_P = []
    EMBEDDING_P_T = []
    for k, v in model_grouped[model_grouped]:
        ...
    
    DEEMBEDDING_P = ["head_p.weight"]
    DEEMBEDDING_P_T = []
    model_grouped[model_grouped][]



    projection_z = torch.zeros((projected_dmodel, dmodel), device=projection.device)

    print("------------------------------init_projections------------------------") #dev
    for name, params in model.named_parameters():

        if is_in_partial_list(name, PROJECTIONS_1_1):
            # projection
            print(f"projection: {name}, {params.shape}")
            params.data.copy_(projection)
            # params.data = projection #dev coupled 
        elif is_in_partial_list(name, PROJECTIONS_1_1_T):
            # projection_T
            print(f"projection_T: {name}, {params.shape}")
            params.data.copy_(projection.T)
            # params.data = projection.T #dev coupled 
            # params.data.copy_(torch.inverse(projection).T) #dev inverted_test
            # params.data.copy_(torch.inverse(projection)) #dev inverted_test
        elif is_in_partial_list(name, PROJECTIONS_1_3):
            # projection_3
            print(f"projection_3: {name}, {params.shape}")
            raise NotImplemented()
            params.data.copy_(projection_3)
        elif is_in_partial_list(name, PROJECTIONS_1_3_T):
            # projection_3_T
            print(f"projection_3_T: {name}, {params.shape}")
            projection_c = projection
            projection_3 = torch.concat((
                torch.concat((projection_c, projection_z, projection_z), dim=0),
                torch.concat((projection_z, projection_c, projection_z), dim=0),
                torch.concat((projection_z, projection_z, projection_c), dim=0),
            ), dim=1)
            params.data.copy_(projection_3.T)
            # params.data = projection_3.T #dev coupled 
            # params.data.copy_(torch.inverse(projection_3).T) #dev inverted_test
            # params.data.copy_(torch.inverse(projection_3)) #dev inverted_test
        else:
            print(f"Not projection: {name}, {params.shape}, {params.requires_grad}")
    print("------------------------------init projections end------------------------") #dev