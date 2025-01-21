from collections import OrderedDict
from typing import Callable, Optional, Union, Type

from lizrd.core.initialization import get_init_weight
from lizrd.core.misc import Linear
from research.projected_distillation.llm import ProjectedPositionalEmbedding, ProjectedTokenEmbedding
from research.projected_distillation.load_and_save_model import load_projected_weights
from research.projected_distillation.utils import freeze_projected_params, initialize_projections
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights


def get_model(
    max_length: int,
    vocab_size: int,
    block_modules: dict[str, Callable[[], torch.nn.Module]],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    ddp_enabled: bool, 
    fsdp_enabled: bool,
    fsdp_param_precision: torch.dtype,
    fsdp_mixed_precision_ignore_classes: list[Type[torch.nn.Module]],
    fsdp_offload_params: bool,
    fsdp_min_num_params: int,
    fsdp_modules_to_wrap: Union[tuple[Type[torch.nn.Module]], None],
    activation_checkpointing_modules: Union[tuple[Type[torch.nn.Module]], None],
    is_logging_process: bool,
    local_rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
    projected_checkpoint: dict[str, torch.Tensor] = None,
    projected_dmodel:int = None,
    projection_init_type:str = None
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    if projected_checkpoint:
        embedding_components = [
            ProjectedTokenEmbedding(vocab_size, dm, projected_dmodel, init_type=init_type, init_scale=init_scale)
        ]
    else:
        embedding_components = [
            llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)
        ]

    if include_positional_embedding:
        if projected_checkpoint:
            embedding_components.append(
                ProjectedPositionalEmbedding(
                    max_length, dm, projected_dmodel, init_type=init_type, init_scale=init_scale
                )
            )
        else:
            embedding_components.append(
                llm.PositionalEmbedding(
                    max_length, dm, init_type=init_type, init_scale=init_scale
                )
            )

    embedding_layer = llm.EmbeddingLayer(*embedding_components).to(first_gpu)

    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        block_modules,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn,
    )

    if projected_checkpoint:
        head = llm.PredictionHead(
            projected_dmodel, vocab_size, init_type=init_type, init_scale=init_scale
        ).to(last_gpu)
        head = torch.nn.Sequential(
            OrderedDict([
                (
                    "head_p",
                    Linear(
                        dm, #xs
                        projected_dmodel, #xb
                        bias=False,
                        init_type=init_type,
                        init_scale=init_scale,
                    ).to(last_gpu),
                ),
                (
                    "head",
                    head,
                )
            ])
        )
    else:
        head = llm.PredictionHead(
            dm, vocab_size, init_type=init_type, init_scale=init_scale
        ).to(last_gpu)
        

    model = llm.LLM(embedding_layer, encoder_tower, head)

    if checkpoint is not None:
        load_model_weights(model, checkpoint)

    if projected_checkpoint is not None:
        if not projection_init_type:
            projection = None
            print("No projection initialization")
        elif projection_init_type == "random":
            print("Random projection initialization")
            projection = get_init_weight(
                shape=(projected_dmodel, dm),
                fan_in=1,  # fan_in=1 is also default in pytorch
                init_type=init_type,
                scale=init_scale/projected_dmodel/2,
            )
        else:
            raise Exception("Wrong projection init type")
        load_projected_weights(model, projected_checkpoint["model"], projection, dm, projected_dmodel, init_scale, device)
        initialize_projections(model, dm, projected_dmodel, projection) #dev
        freeze_projected_params(model)
        
    for name, param in model.named_parameters(): #dev
        print(f"{name}, shape: {param.shape} requires_grad: {param.requires_grad}, {param.device}")
        
    if ddp_enabled:
        model = wrap_in_ddp(module=model, local_rank=local_rank)
    elif fsdp_enabled:
        model = wrap_in_fsdp(
            module=model,
            local_rank=local_rank,
            param_precision=fsdp_param_precision,
            cast_inputs=True,
            mixed_precision_ignored_classes=fsdp_mixed_precision_ignore_classes,
            offload_params=fsdp_offload_params,
            print_model=True,
            min_num_params=fsdp_min_num_params,
            modules_to_wrap=fsdp_modules_to_wrap,
            is_logging_process=is_logging_process,
        )

    if activation_checkpointing_modules is not None:
        check_fn = lambda x: isinstance(x, activation_checkpointing_modules)
        apply_activation_checkpointing(
            model,
            check_fn=check_fn,
            checkpoint_wrapper_fn=make_checkpoint_wrapper_function(),
        )

    return model
