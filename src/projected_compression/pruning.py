import torch
import torch.nn as nn
import logging
from main import get_device

logger = logging.getLogger(__name__)


def prune(model: nn.Module, dimensions_importances_path, target_dmodel, target_dff):
    device = get_device()

    # map_location="cpu": the importances file was saved from a GPU, so a plain load puts it
    # on cuda:0 in EVERY rank, and ranks > 0 then copy the top-k indices cuda:0 -> own GPU.
    # Prune runs per rank before FSDP2 shards the model, so a bad copy gives each rank a
    # different pruned model and FSDP2 stitches their shards together (2026-11-20: entropy
    # a100a corrupts peer copies; CE 68 instead of 11.4 at step 0 on 2 GPUs). On CPU the
    # selection is identical on every rank and needs no GPU peer access.
    dimensions_importances = torch.load(dimensions_importances_path, map_location="cpu")
    dmodel_importances = dimensions_importances["dmodel_importances"]
    dff_importances = dimensions_importances["dff_importances"]

    dmodel_indices = torch.topk(
        dmodel_importances, dim=0, largest=True, k=target_dmodel
    ).indices.to(device)
    dff_indices = []
    for i in range(len(dff_importances)):
        dff_top_indices_current = torch.topk(
            dff_importances[i], dim=0, largest=True, k=target_dff
        ).indices.to(device)
        dff_indices.append(dff_top_indices_current)

    # Embedding
    embedding_weight = model.embedding.weight.data
    model.embedding.weight.data = embedding_weight[:, dmodel_indices]

    # Head
    head_weight = model.head.linear.weight.data
    model.head.linear.weight.data = head_weight[:, dmodel_indices]
    model.head.norm.weight.data = model.head.norm.weight[dmodel_indices]
    model.head.norm.normalized_shape = tuple([target_dmodel])

    for layer, dff_indices_per_layer in zip(model.encoder.blocks, dff_indices):
        layer.attention_layer.norm.weight.data = layer.attention_layer.norm.weight[
            dmodel_indices
        ]
        layer.attention_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.attention_layer.layer.q_proj.weight.data = (
            layer.attention_layer.layer.q_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.k_proj.weight.data = (
            layer.attention_layer.layer.k_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.v_proj.weight.data = (
            layer.attention_layer.layer.v_proj.weight[:, dmodel_indices]
        )
        layer.attention_layer.layer.o_proj.weight.data = (
            layer.attention_layer.layer.o_proj.weight[dmodel_indices, :]
        )

        layer.ff_layer.norm.weight.data = layer.ff_layer.norm.weight[dmodel_indices]
        layer.ff_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.ff_layer.layer.ff_pre_act.weight.data = (
            layer.ff_layer.layer.ff_pre_act.weight[:, dmodel_indices][
                dff_indices_per_layer, :
            ]
        )
        if hasattr(layer.ff_layer.layer, "gate"):
            layer.ff_layer.layer.gate.weight.data = layer.ff_layer.layer.gate.weight[
                :, dmodel_indices
            ][dff_indices_per_layer, :]
        layer.ff_layer.layer.ff_post_act.weight.data = (
            layer.ff_layer.layer.ff_post_act.weight[dmodel_indices, :][
                :, dff_indices_per_layer
            ]
        )

    logger.info("Model pruned.")
    return model
