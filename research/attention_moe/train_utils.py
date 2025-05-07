from typing import Callable, Optional, Union, Type

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from research.attention_moe.diff_attn.rms_norm import RMSNorm

from . import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights

from lizrd.train.load_and_save_model import (
    get_checkpoint_from_path,
    load_optimizer_state,
    prepare_save_weights_path,
)

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
    use_final_norm: bool,
    args,
    rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    norm_fn: Callable[[int], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_components = [
        llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)
    ]

    if include_positional_embedding:
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

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    if use_final_norm:
        output_norm = norm_fn(dm).to(last_gpu)
    else:
        output_norm = None

    model = llm.LLM(embedding_layer, encoder_tower, head, output_norm=output_norm)

    if checkpoint is not None:
        load_model_weights(model, checkpoint)

    param_grops, ratios_in_group_order = make_param_groups_and_lr_ratios(args, model)

    optimizer = torch.optim.AdamW(
        param_grops,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )


    if checkpoint is not None and not args.reset_optimizer:
        load_optimizer_state(optimizer, checkpoint, model, rank)

    for m in model.modules():
        if getattr(m, "post_load_hook", None) is not None:
            m.post_load_hook(args)

    if ddp_enabled:
        model = wrap_in_ddp(module=model, rank=rank)
    elif fsdp_enabled:
        model = wrap_in_fsdp(
            module=model,
            rank=rank,
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

    return model, optimizer, ratios_in_group_order


from collections import defaultdict

def make_param_groups_and_lr_ratios(args, model):
    lr = args.learning_rate
    if args.relative_lr is None:
        return [{"params": model.parameters(), "lr": lr}], [1.0]

    relative_lr: dict = args.relative_lr

    lr_to_params = defaultdict(list)
    for name, param in model.named_parameters():
        ratio = 1.0
        for possible_name in relative_lr.keys():
            if possible_name in name:
                ratio = relative_lr[possible_name]
                break
        lr_to_params[ratio * lr].append(param)
    param_grops = [
        {"params": params, "lr": lr_group} for lr_group, params in lr_to_params.items()
    ]
    ratios_in_group_order = [param_group["lr"] / lr for param_group in param_grops]
    return param_grops, ratios_in_group_order
