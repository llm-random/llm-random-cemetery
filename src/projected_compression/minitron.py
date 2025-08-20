import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hidden dimension: 4096 → 3072
# MLP hidden dimension: 14336 → 9216

def calculate_dimension_importances(model: nn.Module, calibration_data, dmodel, dff):
    """
    Calculate the importance of each neuron in the model based on the calibration data.
    Returns a list of importance scores for dmodel and dff dimensions.
    """
    dmodel_importance = torch.zeros(dmodel, device=device)
    dff_importance = torch.zeros(dff, device=device)

    # Forward pass through the model with calibration data
    with torch.no_grad():
        for document in calibration_data:
            document.unsqueeze_(0)
            # x = model.embedding(calibration_data)
            x = model.embedding(document)
            for layer in model.encoder.blocks:
                # TODO sum

                y = layer.attention_layer.norm(x) # normalized_pre_attn
                dmodel_importance += torch.sum(y, dim=[0, 1])  # Sum across batch and sequence dimensions

                y = layer.attention_layer.layer(y) # attention_output
                x = x + y  # Residual connection


                y = layer.ff_layer.norm(x) # normalized_pre_ff
                dmodel_importance += torch.sum(y, dim=[0, 1])  # Sum across batch and sequence dimensions

                ff_layer = layer.ff_layer.layer
                ff_gated = ff_layer.silu(ff_layer.gate(y))

                y = ff_layer.ff_pre_act(y) # ff_pre_act
                dff_importance += torch.sum(y, dim=[0, 1])

                y = ff_layer.ff_post_act(y * ff_gated) # ff_output

                x = x + y  # Residual connection
            x = model.head(x)

            # assert x == model(calibration_data), "Model output does not match expected output"
            assert x == model(document), "Model output does not match expected output"
    print("Model output matches expected output")

    return dmodel_importance, dff_importance

def prune(model: nn.Module, dmodel_indices, dff_indices):
    """
    Prune the model by selecting the top k neurons based on their importance scores.
    """
    # Embedding
    embedding_weight = model.embedding.embedding.weight.data
    model.embedding.embedding.weight.data = embedding_weight[:, dmodel_indices]

    # Head
    head_weight = model.head.linear.weight.data
    model.head.linear.weight.data = head_weight[:, dmodel_indices]

    for block in model.encoder.blocks:
        block_state_dict = block.state_dict()

        # Attention layer
        block.attention_layer.norm.weight.data = block_state_dict['attention_layer.layer.norm.weight'][dmodel_indices]
        block.attention_layer.layer.q_proj.weight.data = block_state_dict['attention_layer.layer.q_proj.weight'][:, dmodel_indices]
        block.attention_layer.layer.k_proj.weight.data = block_state_dict['attention_layer.layer.k_proj.weight'][:, dmodel_indices]
        block.attention_layer.layer.v_proj.weight.data = block_state_dict['attention_layer.layer.v_proj.weight'][:, dmodel_indices]
        block.attention_layer.layer.o_proj.weight.data = block_state_dict['attention_layer.layer.o_proj.weight'][dmodel_indices, :]

        # Feed-forward layer
        block.ff_layer.norm.weight.data = block_state_dict['ff_layer.layer.norm.weight'][dmodel_indices]
        block.ff_layer.layer.ff_pre_act.weight.data = block_state_dict['ff_layer.layer.ff_pre_act.weight'][dff_indices, dmodel_indices]
        block.ff_layer.layer.gate.weight.data = block_state_dict['ff_layer.layer.gate.weight'][dff_indices, dmodel_indices]
        block.ff_layer.layer.ff_post_act.weight.data = block_state_dict['ff_layer.layer.ff_post_act.weight'][dmodel_indices, dff_indices]

    return model

def minitron_prune(model: nn.Module, dataloader, dmodel, dff, calibration_dataset_size, seq_len):
    # get 1024 samples calibration data
    calibration_data = torch.zeros(calibration_dataset_size, seq_len, dtype=torch.long, device=device)
    for i, batch in enumerate(dataloader):
        if i >= calibration_dataset_size:
            break
        calibration_data[i] = batch[0][:seq_len]

    importance_neurons, importance_dmodel = calculate_dimension_importances(
        model, calibration_data, dmodel, dff
    )

    # select top k neurons
    topk_dmodel = 3072
    topk_dff = 9216

    dmodel_top_indices = torch.topk(importance_neurons, dim=0, largest=True, k=topk_dmodel).indices
    dff_top_indices = torch.topk(importance_dmodel, dim=0, largest=True, k=topk_dff).indices

    model = prune(model, dmodel_top_indices, dff_top_indices)

    return model

