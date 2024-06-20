from typing import Callable, Optional, Union, Type

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from research.subtoken.embedding import SubtokenEmbedding
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights


class SubtokenEmbeddingBlock(torch.nn.Module):

    def __init__(
        self,
        max_length,
        vocab_size,
        dm,
        init_type,
        init_scale,
        include_positional_embedding,
        max_n_bytes_per_token: int,
        include_subtoken_embedding: bool,  # Assuming this parameter should also be configurable
    ):
        super().__init__()
        self.token_embedding = llm.TokenEmbedding(
            vocab_size, dm, init_type=init_type, init_scale=init_scale
        )

        if include_positional_embedding:
            self.positional_embedding = llm.PositionalEmbedding(
                max_length, dm, init_type=init_type, init_scale=init_scale
            )
        else:
            self.positional_embedding = None

        if include_subtoken_embedding:
            self.subtoken_embedding = SubtokenEmbedding(
                dm, max_n_bytes_per_token, init_type=init_type, init_scale=init_scale
            )
        else:
            self.subtoken_embedding = None

    def forward(self, input_ids: torch.Tensor, input_bytes: torch.Tensor):
        tokens_embeddings = self.token_embedding(input_ids)
        if self.positional_embedding is not None:
            positional_embeddings = self.positional_embedding(input_ids)
            tokens_embeddings += positional_embeddings

        if self.subtoken_embedding is not None:
            subtoken_embeddings = self.subtoken_embedding(input_bytes)
            tokens_embeddings += subtoken_embeddings

        return tokens_embeddings


def get_model(
    max_length: int,
    vocab_size: int,
    block_modules: dict[str, Callable[[], torch.nn.Module]],
    dm: int,
    n_blocks: int,
    device: torch.device,
    max_n_bytes_per_token: int,
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
    include_subtoken_embedding: bool,
    rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
):
    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_layer = SubtokenEmbeddingBlock(
        max_length,
        vocab_size,
        dm,
        init_type,
        init_scale,
        include_positional_embedding,
        include_subtoken_embedding=include_subtoken_embedding,
        max_n_bytes_per_token=max_n_bytes_per_token,
    ).to(first_gpu)

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

    model = llm.LLM(embedding_layer, encoder_tower, head)

    if checkpoint is not None:
        load_model_weights(model, checkpoint)

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

    return model
