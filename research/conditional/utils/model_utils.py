from functools import partial
import torch
import torch.nn.functional as F

from lizrd.core import llm
from lizrd.text.data import LLMBatch
from lizrd.core.llm import Parallel
from research.conditional.moe_layers.cont_moe_designs.common_weighted_parameter_matrices import (
    ContinuousMoECommonWeightedParameters,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature_positive import (
    ContinuousMoEAdaTempPositive,
)
from research.conditional.moe_layers.cont_moe_designs.random_grouping import (
    ContinuousMoERandomGroups,
)
from research.conditional.moe_layers.cont_moe_designs.learn_temp_and_common_base import (
    ContinuousMoEFinal,
)
from research.conditional.moe_layers.cont_moe_designs.learnable_temperature import (
    ContinuousMoEAdaTemp,
)
from research.conditional.moe_layers.cont_moe_designs.add_layernorms import (
    ContinuousMoELayernorm,
)
from research.conditional.moe_layers.cont_moe_designs.no_softmax_on_weights import (
    ContinuousMoENosoftmax,
)
from research.conditional.moe_layers.cont_moe_designs.send_result_only_to_top1_token import (
    ContinuousMoETopmerge,
)
from research.conditional.moe_layers.cont_moe_designs.merge_without_weights import (
    ContinuousMoERawmerge,
)
from research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights_common_base import (
    ContinuousMoEMergeDifferentlyCommonBase,
)
from research.conditional.moe_layers.cont_moe_designs.separate_merge_emit_weights import (
    ContinuousMoEMergeDifferentlySimple,
)
from research.conditional.moe_layers.cont_moe_designs.separate_weighted_parameter_matrices import (
    ContinuousMoESeparateWeightedParameters,
)
from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    LegacyContinuousMoE,
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.moe_layers.token_choice import TokenChoiceFF
from research.conditional.moe_layers.ff_timed import FeedForwardTimed
from torch.nn.parallel import DistributedDataParallel as DDP


def make_loss_and_backprop_function(loss_checkpoint_chungs: int):
    if loss_checkpoint_chungs == 0:
        return calculate_llm_loss_and_backward_pass
    else:
        return partial(
            chungized_llm_loss_and_backward_pass, n_chungs=loss_checkpoint_chungs
        )


def chungized_llm_loss_and_backward_pass(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    vocab_size: int,
    scaler: torch.cuda.amp.GradScaler,
    n_chungs: int,
    gradient_accumulation_steps: int,
):
    do_backward_pass = model.training
    if isinstance(model, DDP):
        model = model.module
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    gt_tokens = gt_tokens.to(model.head.device)
    mask = batch.should_calculate_loss
    mask = mask.to(model.head.device)
    num_masked_tokens = mask.sum()

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        encoder_output: torch.Tensor = model.encoder(input_tokens)
        encoder_output_det = encoder_output.detach()
        if do_backward_pass:
            encoder_output_det.requires_grad = True
        chunged_inputs = torch.chunk(encoder_output_det, n_chungs, dim=0)
        chunged_non_masked_inputs = torch.chunk(gt_tokens, n_chungs, dim=0)
        chunged_non_masked_masks = torch.chunk(mask, n_chungs, dim=0)

        total_loss = 0
        total_correct_tokens = 0
        for chunged_input, chunged_gt, chunged_mask in zip(
            chunged_inputs, chunged_non_masked_inputs, chunged_non_masked_masks
        ):
            partial_output = model.head(chunged_input)
            with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float16):
                partial_loss_unmasked = F.cross_entropy(
                    partial_output.reshape(-1, vocab_size),
                    chunged_gt.reshape(-1).long(),
                    reduction="none",
                )
                partial_loss = partial_loss_unmasked[chunged_mask.reshape(-1) == 1]

                partial_correct_tokens = chunged_gt.long() == partial_output.argmax(
                    dim=-1
                )
                partial_correct_tokens = partial_correct_tokens.long().reshape(
                    -1
                ) * chunged_mask.reshape(-1)
                partial_correct_tokens = partial_correct_tokens.sum()

            if do_backward_pass:
                loss = (
                    partial_loss.sum() / num_masked_tokens / gradient_accumulation_steps
                )
                with torch.autocast(
                    device_type="cuda", enabled=False, dtype=torch.float16
                ):
                    scaler.scale(loss).backward()
            total_loss += partial_loss.sum()
            total_correct_tokens += partial_correct_tokens

    if do_backward_pass:
        encoder_output.backward(encoder_output_det.grad)

    aux_info = {
        "correct_tokens": total_correct_tokens,
        "total_masked_tokens": num_masked_tokens,
        "losses": retrieve_additional_losses(model),
    }

    return total_loss / num_masked_tokens / gradient_accumulation_steps, aux_info


def calculate_llm_loss_and_backward_pass(
    batch: LLMBatch,
    model: torch.nn.Module,
    mixed_precision: bool,
    scaler: torch.cuda.amp.GradScaler,
    vocab_size: int,
    gradient_accumulation_steps: int,
):
    do_backward_pass = model.training
    input_tokens = batch.input_ids
    gt_tokens = batch.target_ids
    mask = batch.should_calculate_loss

    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        model_output = model(input_tokens)

    # move the gt tokens and mask to the same device as the model output - they should be on the same device for loss calculation
    gt_tokens = gt_tokens.to(model_output.device)
    mask = mask.to(model_output.device)

    mask_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        gt_tokens.reshape(-1).long(),
        reduction="none",
    )
    mask_loss = mask_loss[mask.reshape(-1) == 1]
    loss = mask_loss.mean() / gradient_accumulation_steps
    if do_backward_pass:
        scaler.scale(loss).backward()

    correct_tokens = gt_tokens.long() == model_output.argmax(dim=-1)
    correct_tokens = correct_tokens.long().reshape(-1) * mask.reshape(-1)
    correct_tokens = correct_tokens.sum()
    total_masked_tokens = mask.sum()

    aux_info = {
        "correct_tokens": correct_tokens,
        "total_masked_tokens": total_masked_tokens,
        "losses": retrieve_additional_losses(model),
    }

    return loss, aux_info


def get_attention_layer(args):
    causal = args.model_type == "gpt"

    attention_layer_fun = lambda: llm.Attention(
        dmodel=args.dmodel,
        heads=args.n_att_heads,
        causal=causal,
        dhead=args.dhead,
        flash=args.flash_attention,
        init_type=args.init_type,
        init_scale=args.init_scale,
    )

    return attention_layer_fun


def get_residual_layer(args):
    if args.residual_mode == "pre_norm":
        return partial(llm.PreNormBlock, dmodel=args.dmodel)
    elif args.residual_mode == "post_norm":
        return partial(llm.PostNormBlock, dmodel=args.dmodel)
    elif args.residual_mode == "rezero":
        return partial(llm.RezeroBlock, dmodel=args.dmodel)
    else:
        raise NotImplementedError(f"Residual type {args.residual_mode} not implemented")


def get_expert_choice_args(args):
    set_arguments_option1 = (
        args.total_experts_width is not None
        and args.effective_dff is not None
        and args.n_experts is not None
    ) and (args.expert_size is None and args.topk_fraction is None)
    set_arguments_option2 = (
        args.expert_size is not None
        and args.topk_fraction is not None
        and args.n_experts is not None
    ) and (args.effective_dff is None and args.total_experts_width is None)
    set_arguments_option3 = (  # this should be the default
        args.granularity is not None
        and args.expansion_rate is not None
        and args.effective_dff_x is not None
    )

    if (
        not set_arguments_option1
        and not set_arguments_option2
        and not set_arguments_option3
    ):
        raise AssertionError(
            "You must specify either total_experts_width, effective_dff, and n_experts "
            "or expert_size, topk_fraction, and n_experts "
            "or granularity, expansion_rate, and effective_dff_x "
        )

    if set_arguments_option3:
        # 4 is the standard dff_x, we assume it's defined relative to that
        dff_x = 4
        args.total_experts_width = args.dmodel * dff_x * args.expansion_rate
        args.n_experts = args.expansion_rate * args.granularity
        args.effective_dff = args.effective_dff_x * args.dmodel

    if args.total_experts_width is not None:
        expert_size = args.total_experts_width / args.n_experts
        assert expert_size == int(expert_size)
        args.expert_size = int(expert_size)

        experts_per_token = args.effective_dff / expert_size

        topk_fraction = experts_per_token / args.n_experts
        assert 0.0 <= topk_fraction <= 1.0
        args.topk_fraction = topk_fraction
    else:
        experts_per_token = args.topk_fraction * args.n_experts
        args.effective_dff = experts_per_token * args.expert_size
        args.total_experts_width = args.expert_size * args.n_experts

    return {
        "dmodel": args.dmodel,
        "n_experts": args.n_experts,
        "expert_size": args.expert_size,
        "topk_fraction": args.topk_fraction,
        "random_perm": args.expert_random_perm,
        "group_by_batch": args.group_granular_moe_by_batch,
        "softmax_ungrouped": args.softmax_ungrouped,
        "one_hot_impl": args.granular_moe_one_hot_impl,
        "softmax_over": args.softmax_over,
        "use_full_einsum": args.use_full_einsum,
        "group_size": args.simulate_group_size,
        "init_type": args.init_type,
        "init_scale": args.init_scale,
        "use_torch_bmm": args.use_torch_bmm,
    }


def get_expert_choice_with_parallel_ff_args(args):
    expert_choice_params = get_expert_choice_args(args)
    n_experts = expert_choice_params["n_experts"]
    expert_size = expert_choice_params["expert_size"]
    top_k_fraction = expert_choice_params["topk_fraction"]

    def calculate_effective_expert_dff(_expert_size, _n_experts, _topk_fraction):
        return _topk_fraction * _n_experts * _expert_size

    if args.ff_parallel_mode == "modify_expert_size":
        expert_size = int(
            expert_choice_params["expert_size"]
            * (1 - args.ff_parallel_compute_fraction)
        )
        expert_choice_params["expert_size"] = expert_size

    elif args.ff_parallel_mode == "modify_topk_fraction":
        top_k_fraction = expert_choice_params["topk_fraction"] * (
            1 - args.ff_parallel_compute_fraction
        )

        expert_choice_params["topk_fraction"] = top_k_fraction

    elif args.ff_parallel_mode == "modify_n_experts":
        n_experts = int(
            expert_choice_params["n_experts"] * (1 - args.ff_parallel_compute_fraction)
        )
        expert_choice_params["n_experts"] = n_experts
    else:
        raise ValueError(
            f"Invalid ff_parallel_mode {args.ff_parallel_mode}. Possible values are modify_expert_size, modify_topk_fraction, modify_n_experts"
        )

    dff_expert = int(
        calculate_effective_expert_dff(expert_size, n_experts, top_k_fraction)
    )
    dff_parallel = args.effective_dff - dff_expert
    return {
        "expert_choice_kwargs": expert_choice_params,
        "parallel_ff_args": (args.dmodel, dff_parallel),
    }


def retrieve_additional_losses(model: torch.nn.Module):
    losses = {}
    if not hasattr(model, "forward_pass_cache"):
        return losses

    if "load_balancing_losses" in model.forward_pass_cache:
        load_balancing_losses = model.forward_pass_cache["load_balancing_losses"]
        load_balancing_losses = torch.stack(load_balancing_losses)
        load_balancing_loss = torch.mean(load_balancing_losses)
        losses["load_balancing_loss"] = load_balancing_loss

    return losses


def get_common_mot_kwargs(args):
    return {
        "dm": args.dmodel,
        "dff": args.dff,
        "n_experts": args.n_experts,
        "group_size": args.group_size,
        "sparsity_dim": args.sparsity_dim,
        "temperature": args.temperature,
        "expert_size": args.expert_size,
        "use_opt_einsum": args.use_opt_einsum,
        "flop_matched": args.flop_matched,
        "init_type": args.init_type,
        "init_scale": args.init_scale,
        "emit_softmax_over_experts": args.emit_softmax_over_experts,
    }


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(
            args.dmodel, args.dff, init_type=args.init_type, init_scale=args.init_scale
        )
    elif args.ff_mode == "vanilla_timed":
        return_fn = lambda: FeedForwardTimed(
            args.dmodel, args.dff, args.activation_type, args.no_ff
        )
    elif args.ff_mode == "cont_moe" or args.ff_mode == "cont_moe_quick":
        return_fn = lambda: ContinuousMoE(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_merge_diff_simple":
        return_fn = lambda: ContinuousMoEMergeDifferentlySimple(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_merge_diff_comm_base":
        return_fn = lambda: ContinuousMoEMergeDifferentlyCommonBase(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_rawmerge":
        return_fn = lambda: ContinuousMoERawmerge(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_topmerge":
        return_fn = lambda: ContinuousMoETopmerge(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_nosoft":
        return_fn = lambda: ContinuousMoENosoftmax(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_adatemp":
        return_fn = lambda: ContinuousMoEAdaTemp(
            **get_common_mot_kwargs(args),
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
        )
    elif args.ff_mode == "cont_moe_adatemp_positive":
        return_fn = lambda: ContinuousMoEAdaTempPositive(
            **get_common_mot_kwargs(args),
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
        )
    elif args.ff_mode == "cont_moe_ln":
        return_fn = lambda: ContinuousMoELayernorm(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_final":
        return_fn = lambda: ContinuousMoEFinal(**get_common_mot_kwargs(args))
    elif args.ff_mode == "cont_moe_random_groups":
        return_fn = lambda: ContinuousMoERandomGroups(
            **get_common_mot_kwargs(args),
            batch_size=args.batch_size,
            seqlen=args.cutoff,
            mix_whole_batch=args.mix_whole_batch,
        )
    elif args.ff_mode == "cont_moe_common_weighted_parameters":
        return_fn = lambda: ContinuousMoECommonWeightedParameters(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_separate_weighted_parameters":
        return_fn = lambda: ContinuousMoESeparateWeightedParameters(
            **get_common_mot_kwargs(args)
        )
    elif args.ff_mode == "cont_moe_legacy":
        return_fn = lambda: LegacyContinuousMoE(**get_common_mot_kwargs(args))
    elif args.ff_mode == "expert_choice":
        ff_args = get_expert_choice_args(args)
        return_fn = partial(ExpertChoiceFF, **ff_args)
    elif args.ff_mode == "expert_choice_with_parallel_ff":
        expert_choice_kwargs = get_expert_choice_with_parallel_ff_args(args)[
            "expert_choice_kwargs"
        ]
        parallel_ff_args = get_expert_choice_with_parallel_ff_args(args)[
            "parallel_ff_args"
        ]
        return_fn = lambda: Parallel(
            ExpertChoiceFF(**expert_choice_kwargs),
            llm.FeedForward(*parallel_ff_args),
        )
    elif args.ff_mode == "token_choice":
        return_fn = lambda: TokenChoiceFF(
            dmodel=args.dmodel,
            n_experts=args.n_experts,
            expert_size=args.effective_dff,
            capacity_factor=args.capacity_factor,
            load_balancing_loss_weight=args.load_balancing_loss_weight,
        )
    elif args.ff_mode == "kernelized_fc":
        from research.conditional.moe_layers.kernelized import FCKernelized

        return_fn = lambda: FCKernelized(
            dmodel=args.dmodel,
            dff=args.dff,
            kernel_r=args.kernel_r,
            kernel_type=args.kernel_type,
            redraw_projections_interval=args.redraw_projections_interval,
            no_kernel_norm=args.no_kernel_norm,
            no_average_attn=args.no_average_attn,
            nystrom=args.nystrom,
            xfavor=args.xfavor,
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    if args.every_other_layer:
        if args.standard_ff_first:
            return_fn = llm.EveryOtherLayer(
                lambda: llm.FeedForward(args.dmodel, args.dff), return_fn
            )
        else:
            return_fn = llm.EveryOtherLayer(
                return_fn, lambda: llm.FeedForward(args.dmodel, args.dff)
            )

    return return_fn
