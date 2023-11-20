import argparse
import secrets

import torch

from lizrd.core import llm, misc
from lizrd.datasets.wikibookdata import get_processed_dataset
from lizrd.scripts.grid_utils import get_machine_backend, MachineBackend
from research.reinitialization.core import linears, linears_loss, linears_plusminus
from research.reinitialization.core import linears_recycle
from research.reinitialization.core import linears_noise
from research.reinitialization.core.pruner import Pruner
from lizrd.train.train_utils import (
    get_model,
    Trainer,
    RetrainTrainer,
)
from research.reinitialization.core.scheduler import DelayedConstScheduler
import os
from lizrd.support.logging import (
    get_logger,
    make_concise_datetime,
)

parser = argparse.ArgumentParser()

parser.add_argument("--testing_regular", action="store_true")
parser.add_argument("--testing_recycle", action="store_true")
parser.add_argument("--use_neptune", action="store_true")
parser.add_argument("--use_clearml", action="store_true")
parser.add_argument("--use_pruner", action="store_true")
parser.add_argument("--mixed_precision", action="store_true")
parser.add_argument("--x_flop", action="store_true")
parser.add_argument("--x_logarithmic", action="store_true")

parser.add_argument("--pruner_prob", type=float, default=None)
parser.add_argument("--pruner_n_steps", type=int, default=None)
parser.add_argument("--project_name", type=str)

parser.add_argument("--name", type=str, default="")
parser.add_argument("--pruner_delay", type=int, default=0)
parser.add_argument("--pruner_n_steps_retrain", type=int, default=None)
parser.add_argument("--ff_layer", type=str, default="regular")
parser.add_argument("--bias", type=str, default="none")
parser.add_argument("--trainer_type", type=str, default="regular")
parser.add_argument("--tags", nargs="*", type=str, default=None)
parser.add_argument("--ds_seed", type=int, default=42)
parser.add_argument("--eval_ds_seed", type=int, default=1984)
parser.add_argument("--retrain_ds_seed", type=int, default=1998)

parser.add_argument("--batch_size", type=str, default=64)
parser.add_argument("--batch_size_buffer", type=float, default=None)
parser.add_argument("--cutoff", type=int, default=128)
parser.add_argument("--dmodel", type=int, default=256)
parser.add_argument("--dff", type=str, default="auto")
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--dhead", type=int, default=None)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--mask_loss_weight", type=float, default=1.0)
parser.add_argument("--class_loss_weight", type=float, default=1.0)
parser.add_argument("--midpoint_loss_weight", type=float, default=0.0)
parser.add_argument("--decay_loss_weight", type=float, default=0.0)
parser.add_argument("--mask_percent", type=float, default=0.15)
parser.add_argument("--n_steps", type=int, default=100_001)
parser.add_argument("--n_steps_eval", type=int, default=100)
parser.add_argument("--n_steps_log", type=int, default=5_000)
parser.add_argument("--immunity", type=int, default=10)
parser.add_argument("--reinit_dist", type=str, default="init")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--sep_dir_mag_magnitude_requires_grad", action="store_true")
parser.add_argument("--sep_dir_mag_small_grad", action="store_true")
parser.add_argument("--n_log_light_steps", type=int, default=100)
parser.add_argument("--n_log_heavy_steps", type=int, default=5000)
parser.add_argument("--log_acc_steps", type=int, default=100)
parser.add_argument("--retrain_warmup_steps", type=int, default=None)
parser.add_argument("--log_neuron_diff", action="store_true")
parser.add_argument("--log_neuron_diff_sample_size", type=int, default=1)
parser.add_argument("--log_neuron_diff_n_samples", type=int, default=100)
parser.add_argument("--neuron_diff_ds_seed", type=int, default=511)
parser.add_argument("--neuron_diff_batches", type=int, default=10)
parser.add_argument("--retrain_without_reinit", action="store_true")
parser.add_argument("--random_indexes", action="store_true")
parser.add_argument("--highest_magnitudes", action="store_true")

parser.add_argument("--mpl_reg_pow", type=float, required=False)
parser.add_argument("--mpl_midpoint_type", type=str, default="mean")
parser.add_argument("--mpl_transform_type", type=str, default="linear")
parser.add_argument("--mpl_only_smaller_neurons", action="store_true")


parser.add_argument("--mpl_aggregation_type", type=str, default="regular")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--model_load_path", type=str, default=None)

parser.add_argument("--noise_ff_prune_ratio", type=float, required=False)
parser.add_argument("--noise_ff_n_steps", type=int, required=False)
parser.add_argument("--noise_interpolation_delay", type=float, default=0.0)
parser.add_argument("--noise_ff_weight_init", type=str, default="random")
parser.add_argument("--lr_warmup_steps", type=int, default=10_000)
parser.add_argument("--model_type", type=str, default="bert")
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--pruning_criterion", type=str, default="smallest")

args = parser.parse_args()

if args.n_gpus > 1:
    raise NotImplementedError("Multi-GPU training is not supported yet.")

VOCAB_SIZE = 30522 if args.model_type == "bert" else 50257

if args.dff == "auto":
    args.dff = args.dmodel * 4
else:
    args.dff = int(args.dff)

ATHENA_MEMORY_CONST = 2.192e7
ENTROPY_MEMORY_CONST = 5.48e6
LOCAL_MEMORY_CONST = 4e5


def batch_size_heuristic(memory_const: float) -> int:
    buffer = args.batch_size_buffer or 1
    return max(
        1, int(memory_const / (args.dmodel * args.n_blocks * 12 + VOCAB_SIZE)) * buffer
    )


if args.batch_size == "auto":
    if get_machine_backend() == MachineBackend.ENTROPY:
        args.batch_size = batch_size_heuristic(ATHENA_MEMORY_CONST)
    elif get_machine_backend() == MachineBackend.ENTROPY:
        args.batch_size = batch_size_heuristic(ENTROPY_MEMORY_CONST)
    else:
        args.batch_size = batch_size_heuristic(LOCAL_MEMORY_CONST)
else:
    args.batch_size = int(args.batch_size)

# basic validation of args
if args.use_pruner and (args.pruner_n_steps is None or args.pruner_prob is None):
    raise ValueError(
        "use_pruner set but pruner_n_steps or pruner_prob or pruner_delay not set"
    )
if args.trainer_type == "retrain" and args.pruner_n_steps_retrain is None:
    raise ValueError("trainer_type is retrain but pruner_n_steps_retrain not set")
if args.trainer_type == "retrain" and not args.use_pruner:
    raise ValueError("trainer_type is retrain but use_pruner not set")
if not args.use_pruner and (
    args.pruner_n_steps is not None
    or args.pruner_prob is not None
    or args.pruner_delay > 0
):
    raise ValueError(
        "use_pruner not set but pruner_n_steps or pruner_prob or pruner_delay set"
    )
if args.ff_layer == "noise" and (
    args.noise_ff_prune_ratio is None or args.noise_ff_n_steps is None
):
    raise ValueError(
        "ff_layer is noise but noise_ff_prune_ratio or noise_ff_n_steps not set"
    )
if (
    args.noise_ff_prune_ratio is not None or args.noise_ff_n_steps is not None
) and args.ff_layer != "noise":
    raise ValueError(
        "ff_layer is not noise but noise_ff_prune_ratio or noise_ff_n_steps set"
    )
if args.ff_layer == "iwd_baseline" and args.iwd_aggregation_type is None:
    raise ValueError("ff_layer is iwd_baseline but iwd_aggregation_type not set")

print("BEGINNING OF FILE")
print("cuda available:")
print(torch.cuda.is_available())

# constants
DEVICE = misc.get_default_device()

unique_timestamp = f"{make_concise_datetime()}{secrets.token_urlsafe(1)}"
modelpath = f"models/{unique_timestamp}"
os.makedirs(modelpath, exist_ok=True)

# set pruner if needed
if args.use_pruner and args.pruner_n_steps:
    pruner = Pruner()
    scheduler = DelayedConstScheduler(
        n_steps_prune=args.pruner_n_steps,
        prob=args.pruner_prob,
        delay=args.pruner_delay,
        n_steps_retrain=args.pruner_n_steps_retrain,
    )
else:
    pruner = None
    scheduler = None

print(pruner)
print(scheduler)
# set ff layer
if args.ff_layer == "regular":
    ff_layer_fun = lambda: llm.FeedForward(args.dmodel, args.dff, bias=args.bias)
elif args.ff_layer == "unstruct_prune":
    ff_layer_fun = lambda: linears.UnstructPruneFF(args.dmodel, args.dff, pruner)
elif args.ff_layer == "struct_prune":
    ff_layer_fun = lambda: linears.StructPruneFF(args.dmodel, args.dff, pruner)
elif args.ff_layer == "unstruct_magnitude_prune":
    ff_layer_fun = lambda: linears.UnstructMagnitudePruneFF(
        args.dmodel, args.dff, pruner
    )
elif args.ff_layer == "struct_magnitude_prune":
    ff_layer_fun = lambda: linears.StructMagnitudePruneFF(
        args.dmodel, args.dff, pruner, args.pruning_criterion
    )
elif args.ff_layer == "unstruct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.UnstructMagnitudeRecycleFF(
        args.dmodel, args.dff, pruner
    )
elif args.ff_layer == "struct_magnitude_recycle":
    ff_layer_fun = lambda: linears_recycle.StructMagnitudeRecycleFF(
        args.dmodel, args.dff, pruner
    )
elif args.ff_layer == "retrain_recycle":
    ff_layer_fun = lambda: linears_recycle.RetrainRecycleFF(
        dmodel=args.dmodel,
        dff=args.dff,
        pruner=pruner,
        retrain_without_reinit=args.retrain_without_reinit,
        random_indexes=args.random_indexes,
        highest_magnitudes=args.highest_magnitudes,
    )
elif args.ff_layer == "struct_magnitude_recycle_with_immunity":
    ff_layer_fun = lambda: linears_recycle.StructMagnitudeRecycleImmunityFF(
        args.dmodel, args.dff, pruner, args.immunity, args.reinit_dist
    )
elif args.ff_layer == "masked_ff":
    ff_layer_fun = linears.MaskedFF
elif args.ff_layer == "separate_direction_magnitude_ff":
    ff_layer_fun = lambda: linears.SeparateDirectionMagnitudeFF(
        args.dmodel,
        args.dff,
        magnitude_requires_grad=args.sep_dir_mag_magnitude_requires_grad,
        small_grad=args.sep_dir_mag_small_grad,
        bias=args.bias,
    )
elif args.ff_layer == "log_ff":
    ff_layer_fun = lambda: linears.LogFF(args.dmodel, args.dff, pruner)
elif args.ff_layer == "plusminus_ff":
    ff_layer_fun = lambda: linears_plusminus.PlusMinusFF(args.dm, args.dff)
elif args.ff_layer == "loss_ff":
    ff_layer_fun = lambda: linears_loss.BaseLossFF(
        dmodel=args.dmodel,
        dff=args.dff,
        only_smaller_neurons=args.mpl_only_smaller_neurons,
        reg_pow=args.mpl_reg_pow,
        midpoint_type=args.mpl_midpoint_type,
        transform_type=args.mpl_transform_type,
        pruner=pruner,
        aggregation_type=args.mpl_aggregation_type,
    )
elif args.ff_layer == "noise":
    ff_layer_fun = lambda: linears_noise.NoiseFF(
        dmodel=args.dmodel,
        dff=args.dff,
        pruner=pruner,
        prune_ratio=args.noise_ff_prune_ratio,
        n_steps_interpolate=args.noise_ff_n_steps,
    )
elif args.ff_layer == "noise":
    ff_layer_fun = lambda: linears_noise.NoiseFF(
        dmodel=args.dmodel,
        dff=args.dff,
        pruner=pruner,
        prune_ratio=args.noise_ff_prune_ratio,
        n_steps_interpolate=args.noise_ff_n_steps,
        new_weight_init=args.noise_ff_new_weight_init,
    )
else:
    raise ValueError(f"ff_layer {args.ff_layer} not recognized")

misc.print_available_gpus()
pdataset = get_processed_dataset(
    batch_size=args.batch_size,
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=args.num_workers,
    seed=args.ds_seed,
    model_type=args.model_type,
)
eval_pdataset = get_processed_dataset(
    batch_size=args.batch_size,
    max_total_length=args.cutoff,
    mask_percent=args.mask_percent,
    device=DEVICE,
    num_workers=1,
    seed=args.eval_ds_seed,
    model_type=args.model_type,
)

if args.model_type == "bert":
    attention_layer_fun = lambda: llm.Attention(
        args.dmodel, args.heads, dhead=args.dhead
    )
elif args.model_type == "gpt":
    attention_layer_fun = lambda: llm.CausalAttention(
        args.dmodel, args.heads, dhead=args.dhead
    )

model = get_model(
    max_length=args.cutoff,
    vocab_size=VOCAB_SIZE,
    ff_layer_fun=ff_layer_fun,
    dm=args.dmodel,
    n_blocks=args.n_blocks,
    device=DEVICE,
    attention_layer_fun=attention_layer_fun,
)
if args.model_load_path:
    print(f"Loading model from {args.model_load_path}")
    print("Make sure that parameters of the saved model are the same as current")
    model.load_state_dict(torch.load(args.model_load_path))

logger = get_logger(args, model, VOCAB_SIZE)

# set optimizer
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

# dataset for neuron diff
if args.log_neuron_diff:
    pdataset_neuron_diff = get_processed_dataset(
        batch_size=args.batch_size,
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        seed=args.neuron_diff_ds_seed,
    )
else:
    pdataset_neuron_diff = None

base_trainer_params = dict(
    model=model,
    optimizer=optimizer,
    pdataset=pdataset,
    pdataset_eval=eval_pdataset,
    batch_size=args.batch_size,
    vocab_size=VOCAB_SIZE,
    mask_percent=args.mask_percent,
    modelpath=modelpath,
    pruner=pruner,
    logger=logger,
    scheduler=scheduler,
    mixed_precision=args.mixed_precision,
    n_log_light_steps=args.n_log_light_steps,
    n_log_heavy_steps=args.n_log_heavy_steps,
    log_acc_steps=args.log_acc_steps,
    neuron_diff_dataset=pdataset_neuron_diff,
    neuron_diff_sample_size=args.log_neuron_diff_sample_size,
    neuron_diff_n_samples=args.log_neuron_diff_n_samples,
    neuron_diff_n_batches=args.neuron_diff_batches,
    losses_weights={
        "mask": args.mask_loss_weight,
        "midpoint": args.midpoint_loss_weight,
        "decay": args.decay_loss_weight,
    },
    noise_interpolation_delay=args.noise_interpolation_delay,
    lr_warmup_steps=args.lr_warmup_steps,
    model_type=args.model_type,
)

if args.trainer_type == "retrain":
    pdataset_retrain = get_processed_dataset(
        batch_size=args.batch_size,
        max_total_length=args.cutoff,
        mask_percent=args.mask_percent,
        device=DEVICE,
        num_workers=args.num_workers,
        seed=args.retrain_ds_seed,
    )
    trainer = RetrainTrainer(
        **base_trainer_params,
        pdataset_retrain=pdataset_retrain,
        retrain_warmup_steps=args.retrain_warmup_steps,
    )
elif args.trainer_type == "regular":
    trainer = Trainer(**base_trainer_params)
else:
    raise ValueError(f"trainer_type {args.trainer_type} not recognized")

trainer.train(args.n_steps, args.n_steps_eval)
