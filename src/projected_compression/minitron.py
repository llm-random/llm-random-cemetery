import torch
import torch.nn as nn



# Hidden dimension: 4096 → 3072
# MLP hidden dimension: 14336 → 9216

def calculate_dimension_importances(model: nn.Module, calibration_data, dmodel, dff):
    """
    Calculate the importance of each neuron in the model based on the calibration data.
    Returns a list of importance scores for dmodel and dff dimensions.
    """
    dmodel_importance = torch.zeros(dmodel)
    dff_importance = torch.zeros(dff)

    # Forward pass through the model with calibration data
    x = model.embedding(calibration_data)
    for layer in model.encoder.blocks:
        normalized_attn = layer.attention_layer.norm(x)
        attention_output = layer.attention_layer.layer(normalized_attn)
        x = x + attention_output  # Residual connection

        normalized_ff = layer.ff_layer.norm(x)
        ff_output = layer.ff_layer.layer(normalized_ff)
        x = x + ff_output  # Residual connection
    x = model.head(x)

    assert x == model(calibration_data), "Model output does not match expected output"
    print("Model output matches expected output")

    return dmodel_importance, dff_importance

# self.attention_layer = Residual(
#             norm=norm_fn(),
#             layer=attention_fn(),
#             log_name=f"{self.log_name}/residual_attention",
#         )
#         self.ff_layer = Residual(
#             norm=norm_fn(),
#             layer=ff_layer_fn(),
#             log_name=f"{self.log_name}/residual_feedforward",
#         )

def mintron_prune(model: nn.Module, dataloader, dmodel, dff):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # get 1024 samples calibration data
    calibration_data = []
    for i, batch in enumerate(dataloader):
        if i >= 1024:
            break
        calibration_data.append(batch)

    importance_neurons, importance_dmodel = calculate_dimension_importances(
        model, calibration_data, dmodel, dff
    )

