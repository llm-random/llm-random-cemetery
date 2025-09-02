from collections import OrderedDict
import os
from pathlib import Path
import re
import torch

from src.core.utils import print_state_dict_info


def remap_pc_finalized_to_nano(llmrandom_dict):
    ...
    # replacement_mappings = [
    #     # Embedding
    #     (r"embedding_layer\.layers\.0\.weight", "embedding.embedding.weight"),
    #     # Attention projections
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.pre_norm\.weight",
    #         r"encoder.blocks.\1.attention_layer.norm.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_q\.weight",
    #         r"encoder.blocks.\1.attention_layer.layer.q_proj.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_k\.weight",
    #         r"encoder.blocks.\1.attention_layer.layer.k_proj.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.input_projection_v\.weight",
    #         r"encoder.blocks.\1.attention_layer.layer.v_proj.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_attention\.layer\.attention\.output_projection\.weight",
    #         r"encoder.blocks.\1.attention_layer.layer.o_proj.weight",
    #     ),
    #     # Feed-forward
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.pre_norm\.weight",
    #         r"encoder.blocks.\1.ff_layer.norm.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_pre_relu\.weight",
    #         r"encoder.blocks.\1.ff_layer.layer.ff_pre_act.weight",
    #     ),
    #     (
    #         r"encoder\.blocks\.block_(\d+)\.block\.residual_feedforward\.layer\.feedforward\.logging_ff_post_relu\.weight",
    #         r"encoder.blocks.\1.ff_layer.layer.ff_post_act.weight",
    #     ),
    #     # Head
    #     (r"head\.unembedding\.head_norm\.weight", "head.norm.weight"),
    #     (r"head\.unembedding\.head\.weight", "head.linear.weight"),
    # ]

    # remapped = {}
    # for key, value in llmrandom_dict.items():
    #     new_key = key

    #     if "residual_attention.layer.attention.rope" in new_key:
    #         continue

    #     for pattern, replacement in replacement_mappings:
    #         new_key = re.sub(pattern, replacement, new_key)

    #     remapped[new_key] = value

    # return OrderedDict(remapped)

def load_finalized_pc_checkpoint(model, load_config):
    checkpoint = torch.load(str(Path(load_config.path, load_config.model_checkpoint_filename)))
    if os.environ["RANK"] == "0":
        print_state_dict_info(checkpoint["model"])
    raise Exception("FIN")
    # fix_qkv_from_llmrandom(model, remapped_state_dict)
    # model.load_state_dict(remapped_state_dict)