from lizrd.core import bert


def introduce_parser_arguments(parser):
    # core hyperparameters, fixed for all experiments; needs a good reason to change

    parser.add_argument("--use_clearml", action="store_true")
    parser.add_argument("--use_neptune", action="store_false")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cutoff", type=int, default=128)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--dff", type=int, default=1024)
    parser.add_argument("--n_att_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_false")
    parser.add_argument("--log_distributions", action="store_true")
    parser.add_argument("--logging_frequency", type=int, default=1000)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument("--n_steps", type=int, default=100_001)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--tags", nargs="*", type=str, default=None)

    # parameters usually changed for experiments

    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--attention_mode", type=str, default="vanilla")
    parser.add_argument(
        "--project_name", type=str, default="nonlinearities/initial_tests"
    )
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--learning_rate_ff", type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--hack_for_batch_size", action="store_true")

    # experimental/legacy parameters

    parser.add_argument("--save_model_checkpoints", action="store_true")
    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    return parser


def get_attention_layer(args):
    if args.attention_mode == "vanilla":
        attention_layer_fun = lambda: bert.Attention(args.dmodel, args.n_att_heads)
    else:
        raise NotImplementedError(
            f"Attention mode {args.attention_mode} not implemented"
        )
    return attention_layer_fun


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        ff_layer_fun = lambda: bert.FeedForward(args.dmodel, args.dff)
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")
    return ff_layer_fun
