from collections import OrderedDict
import re
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def remap_nano_state_dict_to_hf(nano_dict):

    replacement_mappings = [
        # Embedding
        (r"embedding_layer\.layers\.0\.weight", "embedding.embedding.weight"),
        # Attention projections
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.pre_norm\.weight",
            r"encoder.blocks.\1.attention_layer.norm.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_q\.weight",
            r"encoder.blocks.\1.attention_layer.layer.q_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_k\.weight",
            r"encoder.blocks.\1.attention_layer.layer.k_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_v\.weight",
            r"encoder.blocks.\1.attention_layer.layer.v_proj.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.output_projection\.weight",
            r"encoder.blocks.\1.attention_layer.layer.o_proj.weight",
        ),
        # Feed-forward
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.pre_norm\.weight",
            r"encoder.blocks.\1.ff_layer.norm.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_pre_relu\.weight",
            r"encoder.blocks.\1.ff_layer.layer.ff_pre_act.weight",
        ),
        (
            r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_post_relu\.weight",
            r"encoder.blocks.\1.ff_layer.layer.ff_post_act.weight",
        ),
        # Head
        (r"head\.unembedding\.head_norm\.weight", "head.norm.weight"),
        (r"head\.unembedding\.head\.weight", "head.linear.weight"),
    ]

    remapped = {}
    for key, value in nano_dict.items():
        new_key = key

        if "residual_attention.layer.attention.rope" in new_key:
            continue

        for pattern, replacement in replacement_mappings:
            new_key = re.sub(pattern, replacement, new_key)

        remapped[new_key] = value

    return OrderedDict(remapped)


def save_to_hf(nano_model, save_dir, hf_model_type:str, dmodel:int, dff:int, att_nheads:int, nlayers:int):
    # config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
    config = AutoConfig.from_pretrained(hf_model_type)
    hf_model = AutoModelForCausalLM.from_config(config)

    config.hidden_size = dmodel
    config.intermediate_size = dff
    config.num_attention_heads = att_nheads
    config.num_hidden_layers = nlayers

    hf_state_dict = remap_nano_state_dict_to_hf(nano_model.state_dict())
    hf_model.load_state_dict(hf_state_dict, strict=True)

    print(f"Saving HF model with the following config {config}") #dev

    hf_model.save_pretrained(save_dir) 