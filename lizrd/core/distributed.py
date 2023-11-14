from typing import Optional
from functools import partial

from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

from lizrd.core import llm


def custom_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int,
) -> bool:
    return nonwrapped_numel >= min_num_params


def wrap_in_fsdp(
    module: nn.Module,
    rank: Optional[int],
    param_precision: torch.dtype,
    cast_inputs: bool,
    mixed_precision_ignore_classes: list,
    offload_params: bool,
    print_model: bool,
    min_num_params: int,
):
    wrapped = FSDP(
        module,
        device_id=rank,
        mixed_precision=MixedPrecision(
            param_dtype=param_precision,
            reduce_dtype=torch.float32,
            cast_forward_inputs=cast_inputs,
            _module_classes_to_ignore=mixed_precision_ignore_classes,
        ),
        cpu_offload=CPUOffload(offload_params=offload_params),
        auto_wrap_policy=partial(
            custom_auto_wrap_policy, min_num_params=min_num_params
        ),
    )
    activation_checkpointing_check_fn = (
        lambda module: isinstance(module, llm.EmbeddingLayer)
        or isinstance(module, llm.TransformerBlock)
        or isinstance(module, llm.PredictionHead)
    )
    apply_activation_checkpointing(wrapped, check_fn=activation_checkpointing_check_fn)

    if print_model:
        print("------- MODEL AFTER WRAPPING IN FSDP -------")
        print(wrapped)
        print("--------------------------------------------")

    return wrapped


def wrap_in_ddp(
    module: nn.Module,
    rank: int,
):
    return DDP(module=module.to(f"cuda:{rank}"), device_ids=[rank])
