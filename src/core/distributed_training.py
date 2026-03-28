from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from functools import partial
import importlib
import torch
import os
import logging
import sys
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


def _iter_modules_post_order(module):
    for child in module.children():
        yield from _iter_modules_post_order(child)
    yield module


def _should_activation_checkpoint(module):
    return bool(getattr(module, "moe_activation_checkpointing", False))


def _prepare_activation_checkpointing(model):
    return [
        module for module in model.modules() if _should_activation_checkpoint(module)
    ]


def _apply_activation_checkpoint_wrappers(model):
    checkpointed_modules = _prepare_activation_checkpointing(model)
    if not checkpointed_modules:
        return

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=_should_activation_checkpoint,
    )
    logger.info(
        f"Applied activation checkpointing wrappers to {len(checkpointed_modules)} module(s)."
    )


def get_classes_from_dotted_path(paths):
    return [dynamic_import(path) for path in paths]


def dynamic_import(dotted_path):
    module_path, _, obj_name = dotted_path.rpartition(".")
    if not module_path or not obj_name:
        raise ValueError(f"Invalid path: {dotted_path}")
    module = importlib.import_module(module_path)

    return getattr(module, obj_name)


def setup_fsdp1_model(model, fsdp_config):
    classes_to_wrap = get_classes_from_dotted_path(fsdp_config.modules_to_wrap)
    classes_to_wrap.extend(
        sorted(
            {
                module.__class__
                for module in model.modules()
                if _should_activation_checkpoint(module)
            },
            key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
        )
    )
    logger.info(f"[FSDP1] Wrapping model with classes: {classes_to_wrap}")

    ignore_mixed_precision_classes = get_classes_from_dotted_path(
        fsdp_config.mixed_precision.ignored_classes
    )
    logger.info(
        f"[FSDP1] Ignoring mixed precision for classes: {ignore_mixed_precision_classes}"
    )

    mixed_precision_dtype = getattr(
        sys.modules["torch"], fsdp_config.mixed_precision.dtype
    )
    logger.info(f"[FSDP1] Using mixed precision dtype: {mixed_precision_dtype}")

    wrapped_model = FSDP(
        model,
        device_id=int(os.environ["RANK"]),
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            cast_forward_inputs=True,
            _module_classes_to_ignore=ignore_mixed_precision_classes,
        ),
        auto_wrap_policy=ModuleWrapPolicy(classes_to_wrap),
        use_orig_params=True,
    )
    _apply_activation_checkpoint_wrappers(model)
    return wrapped_model


def setup_fsdp2_model(model, fsdp_config):
    modules_to_shard = get_classes_from_dotted_path(fsdp_config.modules_to_shard)
    logger.debug(f"[FSDP2] Sharding model with classes: {modules_to_shard}")
    device_mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))

    fsdp2_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }

    modules_to_shard_tuple = tuple(modules_to_shard)
    for module in _iter_modules_post_order(model):
        if module is model:
            continue
        if isinstance(module, modules_to_shard_tuple) or _should_activation_checkpoint(
            module
        ):
            fully_shard(module, mesh=device_mesh, **fsdp2_kwargs)

    fully_shard(model, mesh=device_mesh, **fsdp2_kwargs)
    _apply_activation_checkpoint_wrappers(model)
    logger.info(f"Sharding done.")
    return model


def setup_distributed_training(model, distributed_config):
    if distributed_config is not None:
        if torch.cuda.is_available():
            if distributed_config.get("fsdp2"):
                model = setup_fsdp2_model(model, distributed_config.fsdp2)
            elif distributed_config.get("fsdp"):
                model = setup_fsdp1_model(model, distributed_config.fsdp)
            else:
                raise ValueError(f"Unknown distributed config.")
        else:
            logger.info(
                "CUDA is not available. Skipping distributed training - using unwrapped model for single-process CPU training."
            )

    return model
