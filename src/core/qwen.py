from collections import OrderedDict
import re

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM

from .llama import remap_llamahf_state_dict_to_nano


def remap_qwen3hf_state_dict_to_nano(qwen_state_dict):
    remapped = remap_llamahf_state_dict_to_nano(qwen_state_dict)
    # Qwen3 adds per-head QK-norm that Llama lacks
    renamed = {}
    for key, value in remapped.items():
        key = re.sub(
            r"(encoder\.blocks\.\d+\.)self_attn\.q_norm\.weight",
            r"\1attention_layer.layer.q_norm.weight",
            key,
        )
        key = re.sub(
            r"(encoder\.blocks\.\d+\.)self_attn\.k_norm\.weight",
            r"\1attention_layer.layer.k_norm.weight",
            key,
        )
        renamed[key] = value
    return OrderedDict(renamed)


def save_pretrained_qwen_as_nano(cfg: OmegaConf, metric_logger=None):

    with torch.device("meta"):
        model = instantiate(cfg.model)

    hf_model = AutoModelForCausalLM.from_pretrained(cfg.trainer.checkpoint.load.path)
    nano_sd = remap_qwen3hf_state_dict_to_nano(hf_model.state_dict())

    weights = {k for k in model.state_dict() if not k.endswith((".sin", ".cos"))}
    missing = weights - set(nano_sd)
    if missing:
        raise RuntimeError(f"Qwen->nano remap left weights unfilled: {sorted(missing)}")

    model.load_state_dict(nano_sd, strict=False, assign=True)

    torch.save(model.state_dict(), cfg.trainer.checkpoint.save.path)

    return None, None, None, None, None
