"""muP coordinate check for the optimizer options (plan/lines/line2_optimizer.md, tasks 2, 4).

Trains the tiny_local model at several widths (common.h) and one fixed depth for a few
steps on one fixed batch, and prints, per parameter role, the RMS of the role's module
outputs and of their change since step 0. Without a width rule (adamw_groups, muon) the
change grows with width; with one (adamw_simplep, muon_mup) it does not. With --moe the
router gets its own row, its logits.

Usage, from the repo root:

    pixi run python scripts/mup_coord_check.py
    pixi run python scripts/mup_coord_check.py --widths 2 4 8 16 --steps 4 --lr 7
    pixi run python scripts/mup_coord_check.py --options muon muon_mup --moe
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
import src.definitions  # noqa: F401  (registers base_schema, used by tiny_local)
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.core.moe import MoE
from src.core.optimizers import ROLES, build_optimizer_and_scheduler, param_roles
from src.core.utils import solve_config_lr

# Table rows: the roles, then MoE routers (role hidden) on their own.
ROWS = (*ROLES, "router")


def build(option, h, n_blocks, lr, seq, moe):
    """tiny_local's model at width h, and its optimizer, as tests/test_optimizers.py builds them."""
    warnings.filterwarnings("ignore")
    overrides = [
        f"_trainer/_optimizer@optimizer={option}",
        # Constant LR: WSD with no warmup or decay.
        "_trainer/_scheduler@trainer.scheduler=wsd",
        "trainer.scheduler.warmup_steps=0",
        "trainer.scheduler.decay_steps=0",
        f"common.h={h}",
        f"common.n_blocks={n_blocks}",
        f"common.sequence_length={seq}",
    ]
    if moe:
        overrides.append("+_ff_layer@model.encoder.block_fn.ff_layer_fn=moe")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.abspath("configs")
    ):
        cfg = compose("tiny_local", overrides)
    torch.manual_seed(0)
    model = instantiate(cfg.model, _convert_="all")
    optimizer, _ = build_optimizer_and_scheduler(cfg, model, lr)
    return model, optimizer, cfg.common.vocab_size


def role_modules(model):
    """Modules that own a parameter directly, keyed by that parameter's role; an MoE's own
    parameter is its router."""
    role_of = {id(p): role for role, ps in param_roles(model).items() for p in ps}
    out = {row: [] for row in ROWS}
    for module in model.modules():
        own = list(module.parameters(recurse=False))
        if own:
            row = "router" if isinstance(module, MoE) else role_of[id(own[0])]
            out[row].append(module)
    return out


def produced(module, args, out):
    """What a module's own parameter produces: an MoE's router logits, else its output."""
    if isinstance(module, MoE):
        return F.linear(args[0], module.router_weight).detach()
    return out.detach()


def rms(x):
    return x.float().pow(2).mean().sqrt().item()


def coord_check(option, h, n_blocks, lr, steps, batch, seq, moe):
    """Per row, after `steps` updates: mean over its modules of RMS(output), RMS(output - output at step 0)."""
    model, optimizer, vocab_size = build(option, h, n_blocks, lr, seq, moe)
    outputs = {}
    for row, modules in role_modules(model).items():
        for i, module in enumerate(modules):
            module.register_forward_hook(
                lambda m, args, out, key=(row, i): outputs.__setitem__(
                    key, produced(m, args, out)
                )
            )
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, vocab_size, (batch, seq + 1), generator=generator)
    model(tokens[:, :-1])
    first = dict(outputs)
    for _ in range(steps):
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model(tokens[:, :-1])
    cells = {}
    for row in ROWS:
        keys = [k for k in outputs if k[0] == row]
        if keys:
            act = sum(rms(outputs[k]) for k in keys) / len(keys)
            delta = sum(rms(outputs[k] - first[k]) for k in keys) / len(keys)
            cells[row] = (act, delta)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--lr", type=float, default=7, help="2**-lr if >= 1, as configs"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument(
        "--options", nargs="+", default=["adamw_groups", "adamw_simplep"]
    )
    parser.add_argument(
        "--moe", action="store_true", help="the MoE feed-forward, with a router row"
    )
    args = parser.parse_args()
    lr, _ = solve_config_lr(args.lr)
    print(
        f"n_blocks={args.n_blocks}, lr={lr:.3g}, step {args.steps}, batch "
        f"{args.batch}x{args.seq}{', MoE' if args.moe else ''}. "
        "Cells: activation RMS / change since step 0."
    )
    header = "| option | role | " + " | ".join(f"h{h}" for h in args.widths) + " |"
    print(header)
    print("|" + " --- |" * (2 + len(args.widths)))
    for option in args.options:
        by_width = {
            h: coord_check(
                option, h, args.n_blocks, lr, args.steps, args.batch, args.seq, args.moe
            )
            for h in args.widths
        }
        for row in ROWS:
            if row in by_width[args.widths[0]]:
                cells = [
                    "{:.3g} / {:.3g}".format(*by_width[h][row]) for h in args.widths
                ]
                print(f"| {option} | {row} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
