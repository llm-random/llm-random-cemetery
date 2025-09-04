import torch
import torch.nn as nn
import logging
import os

from main import get_device
from src.core.checkpointing import get_full_checkpoint_path

device = get_device()
logger = logging.getLogger(__name__)


def _calculate_activations_dimension_importances(model: nn.Module, calibration_data, dmodel, dff, n_blocks, device="cuda"):
    """
    Calculate importance of each neuron (dmodel and dff) using forward hooks.
    """
    dmodel_importance = torch.zeros(dmodel, device=device)
    dff_importance = torch.zeros(n_blocks, dff, device=device)

    handles = []

    # --- Hook functions ---
    def hook_dmodel_pre_attn(layer, inp, out):
        nonlocal dmodel_importance
        # inp[0] has shape [batch, seq, dmodel]
        dmodel_importance += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    def hook_dmodel_pre_ff(layer, inp, out):
        nonlocal dmodel_importance
        dmodel_importance += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    def hook_ff_pre_act(layer, inp, out, block_idx=None):
        nonlocal dff_importance
        dff_importance[block_idx] += torch.sum(torch.abs(out.detach()), dim=[0, 1])

    # --- Register hooks ---
    for block_idx, block in enumerate(model.encoder.blocks):
        # normalized pre-attention
        handles.append(block.attention_layer.norm.register_forward_hook(hook_dmodel_pre_attn))

        # normalized pre-ff
        handles.append(block.ff_layer.norm.register_forward_hook(hook_dmodel_pre_ff))

        # ff_pre_act with block index captured
        handles.append(
            block.ff_layer.layer.ff_pre_act.register_forward_hook(
                lambda layer, inp, out, idx=block_idx: hook_ff_pre_act(layer, inp, out, idx)
            )
        )

    # --- Run calibration data ---
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            print(f"Beginning batch {i}")
            _ = model(batch.to(device))

    # cleanup
    for h in handles:
        h.remove()

    logger.debug("Importance dimensions calculated.")

    return dmodel_importance, dff_importance


def minitron_importances(model: nn.Module, dataloader, dmodel, dff, calibration_dataset_size, seq_len, total_batch_size, n_blocks, checkpoint_save_path):
    
    calibration_data = torch.zeros(calibration_dataset_size // total_batch_size, total_batch_size, seq_len, dtype=torch.long, device=device)
    for i, batch in enumerate(dataloader):
        if i * total_batch_size >= calibration_dataset_size:
            break
        calibration_data[i] = batch[:, :seq_len]

    dmodel_importances, dff_importances = _calculate_activations_dimension_importances(
        model, calibration_data, dmodel, dff, n_blocks
    )

    dict_to_save = {"dmodel_importances": dmodel_importances, "dff_importances": dff_importances}
    path = get_full_checkpoint_path(checkpoint_save_path) + "/dimensions_importances.pt"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict_to_save, path)

    logger.info(f"Saved {type} importances to {path}.")


    return dict_to_save


def prune(model: nn.Module, dimensions_importances_path, target_dmodel, target_dff):
    dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = model.get_model_dimensions()

    dimensions_importances = torch.load(dimensions_importances_path)
    dmodel_importances = dimensions_importances["dmodel_importances"]
    dff_importances = dimensions_importances["dff_importances"]

    dmodel_indices = torch.topk(dmodel_importances, dim=0, largest=True, k=target_dmodel).indices
    dff_indices = []
    for i in range(nlayers):
        dff_top_indices_current = torch.topk(dff_importances[i], dim=0, largest=True, k=target_dff).indices
        dff_indices.append(dff_top_indices_current)

    # Embedding
    embedding_weight = model.embedding.embedding.weight.data
    model.embedding.embedding.weight.data = embedding_weight[:, dmodel_indices]

    # Head
    head_weight = model.head.linear.weight.data
    model.head.linear.weight.data = head_weight[:, dmodel_indices]
    model.head.norm.weight.data = model.head.norm.weight[dmodel_indices]
    model.head.norm.normalized_shape = tuple([target_dmodel])

    for layer, dff_indices_per_layer in zip(model.encoder.blocks, dff_indices):
        layer.attention_layer.norm.weight.data = layer.attention_layer.norm.weight[dmodel_indices]
        layer.attention_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.attention_layer.layer.q_proj.weight.data = layer.attention_layer.layer.q_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.k_proj.weight.data = layer.attention_layer.layer.k_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.v_proj.weight.data = layer.attention_layer.layer.v_proj.weight[:, dmodel_indices]
        layer.attention_layer.layer.o_proj.weight.data = layer.attention_layer.layer.o_proj.weight[dmodel_indices, :]

        layer.ff_layer.norm.weight.data = layer.ff_layer.norm.weight[dmodel_indices]
        layer.ff_layer.norm.normalized_shape = tuple([target_dmodel])
        layer.ff_layer.layer.ff_pre_act.weight.data = layer.ff_layer.layer.ff_pre_act.weight[:, dmodel_indices][dff_indices_per_layer, :]
        layer.ff_layer.layer.gate.weight.data = layer.ff_layer.layer.gate.weight[:, dmodel_indices][dff_indices_per_layer, :]
        layer.ff_layer.layer.ff_post_act.weight.data = layer.ff_layer.layer.ff_post_act.weight[dmodel_indices, :][:, dff_indices_per_layer]

    return model

def delete_model(model):
    return None

# def prune(model, dimensions_importances_path, target_dmodel, target_dff):
#     dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers = model.get_model_dimensions()

#     dimensions_importances = torch.load("dimensions_importances_path")

#     dmodel_importances = dimensions_importances["dmodel_importances"]
#     dff_importances = dimensions_importances["dff_importances"]

#     dmodel_top_indices = torch.topk(dmodel_importances, dim=0, largest=True, k=target_dmodel).indices

#     dff_top_indices = []
#     for i in range(nlayers):
#         dff_top_indices_current = torch.topk(dff_importances[i], dim=0, largest=True, k=target_dff).indices
#         dff_top_indices.append(dff_top_indices_current)

#     model = _prune(model, dmodel_top_indices, dff_top_indices, target_dmodel)
#     return model


# def minitron_prune(model: nn.Module, dataloader, dmodel, target_dmodel, dff, target_dff, calibration_dataset_size, seq_len, total_batch_size, n_blocks, checkpoint_save_path):
#     calibration_data = torch.zeros(calibration_dataset_size // total_batch_size, total_batch_size, seq_len, dtype=torch.long, device=device)
#     for i, batch in enumerate(dataloader):
#         if i * total_batch_size >= calibration_dataset_size:
#             break
#         calibration_data[i] = batch[:, :seq_len]

#     dmodel_importance, dff_importance = calculate_dimension_importances(
#         model, calibration_data, dmodel, dff, n_blocks
#     )

#     logger.debug("Importance dimensions calculated.")

#     dmodel_top_indices = torch.topk(dmodel_importance, dim=0, largest=True, k=target_dmodel).indices

#     dff_top_indices = []
#     for i in range(n_blocks):
#         dff_top_indices_current = torch.topk(dff_importance[i], dim=0, largest=True, k=target_dff).indices
#         dff_top_indices.append(dff_top_indices_current)


#     dict_to_save = {"dmodel_top_indices": dmodel_top_indices, "dff_top_indices": dff_top_indices}
#     path = get_full_checkpoint_path(checkpoint_save_path) + "/top_indices.pt"

#     os.makedirs(os.path.dirname(path), exist_ok=True)

#     torch.save(dict_to_save, path)

#     model = prune(model, dmodel_top_indices, dff_top_indices, target_dmodel)

#     logger.info("Model pruned.")

#     return model

