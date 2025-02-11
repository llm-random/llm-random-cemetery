import torch

from lizrd.core.initialization import get_init_weight

TRANSFER_PARAMS = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.", #FF

    ".block.residual_attention.layer.attention.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.weight", #ATT

    "embedding_layer.layers.0.weight", #TE
    "embedding_layer.layers.1.layer.weight", #PE

    "head.weight", #Head
]

CAST_PROJECTED_PARAMS_NAME_PARTS = [
    (".output_projection.output_projection.", ".output_projection."), #ATT
    (".input_projection.input_projection.", ".input_projection."), #ATT
    ("embedding_layer.layers.0.embedding.weight", "embedding_layer.layers.0.weight"), #TE
    ("head.head.weight", "head.weight"), #Head
    # ("head.weight", "default_head"), #Head - prevents coping to default head
    ("embedding_layer.layers.1.projected_layer.pe_layer.weight", "embedding_layer.layers.1.layer.weight"), # PE
]

LAYER_NORM_COPY = [
    # ".block.residual_feedforward.layer.pre_norm."
]

UNPROJECTED_EMBEDDINGS_BLACKLIST = [
    "embedding_layer.layers.0.weight",
    "head.weight",
    "embedding_layer.layers.1.layer.weight",
    "asd",
    "asd",
]


def load_projected_weights(model:torch.nn.Module, projected_weights, projection:torch.Tensor, dm, projected_dmodel, init_scale, unprojected_embeddings): 
    print(list(projected_weights.keys())) #dev
    print("------------------------------replace with new values------------------------") #dev
    for name, params in model.named_parameters():
        for e in CAST_PROJECTED_PARAMS_NAME_PARTS:
            if e == "head.weight": # prevents copying to default not projected head 
                name = "default_head"
            if e[0] in name:
                name = name.replace(e[0], e[1])
                print("replaced name", name) #dev
        if unprojected_embeddings:
            for e in UNPROJECTED_EMBEDDINGS_BLACKLIST:
                if e in name:
                    name = name + "_unprojected_embeddings"
        prj_params = projected_weights.get(name)
        print(f"{name} - prj_params: ", prj_params.shape if prj_params is not None else prj_params) #dev
        if (prj_params is not None) and any([reg in name for reg in TRANSFER_PARAMS]):
            print(f"REPLACED: {name}, {prj_params.device}")
            params.data.copy_(prj_params)
        if (prj_params is not None) and any([reg in name for reg in LAYER_NORM_COPY]):
            print(f"REPLACED_PROJECTED: {name}, {prj_params.device}")
            if projection is None:
                local_p = get_init_weight(
                    shape=(projected_dmodel, dm),
                    fan_in=1,  # fan_in=1 is also default in pytorch
                    init_type="truncated_normal",
                    scale=init_scale,
                ).to(prj_params.device)
            else:
                local_p = projection.to(prj_params.device)
            # print(prj_params.shape) #dev
            # print(prj_params.device) #dev
            # print(local_p.shape) #dev
            # print(local_p.device) #dev
            np = prj_params@local_p
            print(np.shape) #dev
            params.data.copy_(np)
    print("------------------------------replace with new values end------------------------") #dev

