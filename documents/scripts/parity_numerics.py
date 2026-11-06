#!/usr/bin/env python
"""Parity audit, Phase 1: offline numerics on the two h12 runs (no training).

The plan is documents/notes/ame_efficiency_parity_plan.md; results go to
documents/notes/ame_efficiency_parity_report.md. Run from the root of a
`parity_audit` checkout, so that every model is built by the code the runs used
(`main.get_model_optimizer_scheduler`, the same Hydra configs):

    python documents/scripts/parity_numerics.py CHECK --out DIR [--data FILE]

CHECK is one of:

    dump_data   the first training batches of seed 123 and the whole held-out
                set, exactly as the Trainer's dataloaders yield them (CPU only)
    final_ce    1.1: final_ce from each checkpoint through the Trainer's own
                final_eval, plus the precise (fp64-accumulated) evaluators
    budget      1.6 + 1.3: whole-model gradients of every path against its fp32
                reference, at init and at the final checkpoints; the bank
                gradient's structural checks
    layers      1.2 + 1.4: one AME layer at a time at the production shape, with
                the trained weights and the layer's real input and upstream
                gradient: kernel, LSE gradient, sink, prep precision
    optim       1.5: one optimizer step, Tier 2 against the ordinary path, fed
                identical gradients
    machinery   1.7: the cached BlockMask, and the instrumentation's side effects

Every path is a *variant*: the run's config plus Hydra overrides (VARIANTS below).
Checkpoints are read from --ckpt-root (default ~/parity_ckpt). `--smoke` shrinks
the model (h2, vocab 512, short sequences) so the logic runs on a laptop CPU;
smoke numbers mean nothing.
"""

import argparse
import contextlib
import gc
import json
import math
import os
import sys
import time
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BASES = {"ame": "s2_ps_gate_base", "dense": "s1_d_gate_base"}
CKPT_DIRS = {"ame": "ame_s2ps_6256577", "dense": "dense_s1d_6252398"}
SEQ = 2049  # tokens per sample, as the runs read them (the model sees 2048)

MP_NULL = "trainer.mem_parallel_size=null"
NO_COMPILE = "compile_blocks_full=false"
REFERENCE = "++common.attention.ame.backend=reference"
FP32 = "trainer.autocast_bf16=false"

# name: (arm, overrides, autocast, micro-batch used for gradients)
VARIANTS = {
    # S2-PS as run: Tier 2 banks (bf16 compute copy), compiled blocks, flex.
    "ame_prod": ("ame", [], True, 32),
    # The same without compile_blocks_full: A8 isolated (flex is still compiled).
    # Micro-batch 16: eager blocks at 32 need more than an 80 GB H100 (job 36320).
    "ame_prod_eager": ("ame", [NO_COMPILE], True, 16),
    # Standard mixed-precision AME: banks as fp32 nn.Embedding, compiled, flex.
    "ame_std": ("ame", [MP_NULL], True, 32),
    # A plain bf16 AME: reference attention under autocast, eager.
    "ame_ref16": ("ame", [MP_NULL, NO_COMPILE, REFERENCE], True, 8),
    # The oracle: fp32 everywhere, reference attention, eager.
    "ame_ref32": ("ame", [MP_NULL, NO_COMPILE, REFERENCE, FP32], False, 4),
    # S1-D as run: compiled blocks, SDPA flash under autocast.
    "dense_prod": ("dense", [], True, 32),
    "dense_prod_eager": ("dense", [NO_COMPILE], True, 32),
    # The oracle: fp32, SDPA math, eager.
    "dense_ref32": ("dense", [NO_COMPILE, FP32], False, 8),
    # Production with the cross-entropy taken on fp32 logits (micro-batch 16,
    # room for the fp32 copy of the logits): X3 (CE on bf16
    # logits, shared by both arms) isolated from everything AME-specific.
    "ame_prod_ce32": ("ame", [], True, 16),
    "dense_prod_ce32": ("dense", [], True, 16),
}
CE32 = {"ame_prod_ce32", "dense_prod_ce32"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    log(f"wrote {path}")


# ---- configs and models ------------------------------------------------------


def compose(arm, overrides=(), smoke=False):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import resolver  # noqa: F401  registers ${eval:...}
    import src.definitions  # noqa: F401  registers base_schema

    extra = ["infrastructure.metric_logger.name=parity_numerics"]
    if smoke:
        extra += ["common.h=2", "common.vocab_size=512", "common.sequence_length=129"]
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=os.path.join(REPO, "configs", "parity"), version_base=None
    ):
        return hydra_compose(config_name=BASES[arm], overrides=extra + list(overrides))


class _Quiet(types.SimpleNamespace):
    heavy_metrics_calculation_interval = 100


def quiet_logger():
    """A RecorderLogger at step 1 of interval 100: Residual's metric hook is
    called (as in the runs, where it broke the graph twice per block) but
    computes nothing."""
    from src.core.metric_loggers import RecorderLogger

    logger = RecorderLogger(_Quiet())
    logger.set_step(1)
    return logger


def build(variant, device, smoke=False):
    """Model, optimizer, scheduler and bank controller, built the way main.py
    builds them (seed, instantiate, controller, compile, AdamW groups)."""
    from hydra.utils import instantiate

    from main import get_model_optimizer_scheduler
    from src.core.memory_parallel import get_controller
    from src.core.model import Residual
    from src.core.utils import solve_config_lr

    arm, overrides, autocast, _ = VARIANTS[variant]
    cfg = compose(arm, overrides, smoke)
    if smoke and device.type == "cpu":
        # inductor miscompiles the bank scatter on CPU (§1.18) and compiled flex
        # cannot lower there: smoke runs never compile.
        cfg.compile_blocks_full = False
    torch._dynamo.reset()
    torch.manual_seed(cfg.trainer.train_dataloader.dataset.seed)
    model = instantiate(cfg.model, _convert_="all").to(device)
    logger = quiet_logger()
    for module in model.modules():
        if isinstance(module, Residual):
            module.set_metric_logger(logger)
    lr, _ = solve_config_lr(float(cfg.trainer.learning_rate))
    model, optimizer, scheduler = get_model_optimizer_scheduler(cfg, model, lr)
    banks = get_controller(model)
    if banks is not None:
        banks.materialize()
    return types.SimpleNamespace(
        variant=variant,
        arm=arm,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        banks=banks,
        autocast=autocast and device.type == "cuda",
        micro=VARIANTS[variant][3],
        logger=logger,
    )


def bank_slots(model):
    """[(module, attr, fully-qualified weight name)] in the controller's order."""
    from src.core.memory_parallel import ShardedBank

    slots = []
    for name, module in model.named_modules():
        for attr in ("q_bank", "k_bank"):
            bank = getattr(module, attr, None)
            if isinstance(bank, (nn.Embedding, ShardedBank)):
                slots.append((module, attr, f"{name}.{attr}.weight"))
    return slots


_CKPT_CACHE = {}


def _read(path):
    if path not in _CKPT_CACHE:
        t0 = time.time()
        _CKPT_CACHE[path] = torch.load(path, map_location="cpu", weights_only=False)
        log(f"read {path} ({time.time() - t0:.0f} s)")
    return _CKPT_CACHE[path]


def load_checkpoint(built, ckpt_root, with_optimizer=False):
    """The run's final checkpoint into any variant.

    Tier 2 (mp 1) reads the model state dict and the bank sidecar the way
    `load_checkpoint_from_file` does. The ordinary path (mp null) has the banks
    as nn.Embedding weights the state dict does not carry, so they come from
    the sidecar's fp32 masters, in the controller's bank order.
    """
    arm = built.arm
    folder = os.path.join(ckpt_root, CKPT_DIRS[arm])
    state = _read(os.path.join(folder, "__model_checkpoint_filename.pt"))
    model = built.model
    if arm == "dense":
        model.load_state_dict(state["model"])
    elif built.banks is not None:
        model.load_state_dict(state["model"])
        side = _read(os.path.join(folder, "memory_parallel_rank0.pt"))
        built.banks.load_state_dict(side)
        built.sidecar = side
    else:
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        slots = bank_slots(model)
        expected = {name for _, _, name in slots}
        assert not unexpected, unexpected
        assert set(missing) == expected, (sorted(set(missing) ^ expected))[:6]
        side = _read(os.path.join(folder, "memory_parallel_rank0.pt"))
        assert len(side["masters"]) == len(slots)
        with torch.no_grad():
            for (module, attr, _), master in zip(slots, side["masters"]):
                getattr(module, attr).weight.copy_(master)
        built.sidecar = side
    if with_optimizer:
        built.checkpoint_state = state
    return state


def free(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- data ------------------------------------------------------------------------


def dump_data(args):
    """The first training batches of seed 123 and the full held-out set."""
    from hydra.utils import instantiate

    from src.core.utils import create_batch_fingerprint

    cfg = compose("ame")
    train = instantiate(cfg.trainer.train_dataloader)
    held = instantiate(cfg.trainer.eval_dataloader)
    it = iter(train)
    train_batches = [next(it) for _ in range(args.train_batches)]
    held_batches = list(held)
    out = {
        "train": torch.cat(train_batches).to(torch.int32),
        "heldout": torch.cat(held_batches).to(torch.int32),
        "train_fingerprint": create_batch_fingerprint(train_batches[0]),
        "heldout_fingerprint": create_batch_fingerprint(held_batches[0]),
        "heldout_batches": len(held_batches),
    }
    path = os.path.join(args.out, "parity_batches.pt")
    torch.save(out, path)
    log(
        f"train {tuple(out['train'].shape)}, heldout {tuple(out['heldout'].shape)} "
        f"in {len(held_batches)} batches -> {path}"
    )
    save_json(
        os.path.join(args.out, "data.json"),
        {k: v for k, v in out.items() if not torch.is_tensor(v)},
    )


def load_data(args, device):
    if args.smoke:
        g = torch.Generator().manual_seed(0)
        seq = 129
        return {
            "train": torch.randint(0, 512, (64, seq), generator=g).to(device),
            "heldout": torch.randint(0, 512, (64, seq), generator=g).to(device),
        }
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    return {k: data[k].to(torch.int64).to(device) for k in ("train", "heldout")}


# ---- forward, loss, gradients ------------------------------------------------------


def autocast_ctx(on):
    return torch.autocast("cuda", dtype=torch.bfloat16) if on else contextlib.nullcontext()


def loss_ctx(on):
    # Trainer._loss_ctx: the cross-entropy runs with autocast off, on bf16 logits.
    return torch.autocast("cuda", enabled=False) if on else contextlib.nullcontext()


def forward_loss(built, chunk, precise=None):
    """Trainer._forward_loss + _cross_entropy for one micro-batch: mean CE.

    The returned mean is the Trainer's (a bf16 scalar on the production paths)
    and is what gets backpropagated. `precise`, if a list, receives the fp64
    sum of the per-token losses and their count: the Trainer's bf16 mean is
    quantized (ulp 2^-6 near 3, 2^-4 near 11), too coarse for comparing paths."""
    inp, tgt = chunk[:, :-1], chunk[:, 1:]
    with autocast_ctx(built.autocast):
        logits = built.model(inp)
        with loss_ctx(built.autocast):
            if built.variant in CE32:
                logits = logits.float()
            per_token = F.cross_entropy(
                logits.flatten(0, -2), tgt.reshape(-1).long(), reduction="none"
            )
            loss = per_token.mean()
    if precise is not None:
        precise.append((per_token.detach().double().sum().item(), per_token.numel()))
    return loss


def zero_all(built):
    built.model.zero_grad(set_to_none=True)
    if built.banks is not None:
        built.banks.materialize()
        built.banks.zero_grad()


def gradients(built, sequences):
    """Mean CE over `sequences` and its gradient, accumulated in micro-batches
    of `built.micro` exactly as the Trainer does (each micro-batch's mean CE
    scaled by its share). Bank gradients come from `main_grad` on Tier 2."""
    zero_all(built)
    n = sequences.shape[0]
    precise = []
    for start in range(0, n, built.micro):
        chunk = sequences[start : start + built.micro]
        loss = forward_loss(built, chunk, precise)
        (loss * (chunk.shape[0] / n)).backward()
    total = sum(s for s, _ in precise) / sum(c for _, c in precise)
    return total, collect_grads(built)


def collect_grads(built):
    grads = {}
    slot_names = {}
    if built.banks is not None:
        for module, attr, name in bank_slots(built.model):
            slot_names[id(getattr(module, attr))] = name
        for bank in built.banks.banks:
            grads[slot_names[id(bank)]] = bank.main_grad.detach().float().cpu().clone()
    for name, p in built.model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().float().cpu().clone()
    return grads


GROUP_KEYS = ("tau", "alpha", "beta", "k_sink", "v_sink", "q_proj", "k_proj", "v_proj", "o_proj")


def group_of(name):
    if ".q_bank." in name:
        return "bank_q"
    if ".k_bank." in name:
        return "bank_k"
    last = name.split(".")
    for key in GROUP_KEYS:
        if key in last:
            return key
    if ".ff_layer." in name:
        return "ff"
    if name.startswith("embedding"):
        return "embedding"
    if name.startswith("head.linear"):
        return "head"
    if "norm" in name:
        return "norm"
    return "other"


def layer_of(name):
    parts = name.split(".")
    return int(parts[2]) if len(parts) > 2 and parts[1] == "blocks" else None


def compare(test, ref, by_layer=False):
    """Per parameter group: relative error, cosine, and the signed projection
    <test - ref, ref> / |ref|^2 (a systematic shrink or stretch along the true
    gradient), all accumulated in float64."""
    acc = {}
    for name, r in ref.items():
        if name not in test:
            continue
        t = test[name]
        keys = [group_of(name)]
        if by_layer and layer_of(name) is not None:
            keys.append(f"{group_of(name)}@{layer_of(name)}")
        r64, t64 = r.double(), t.double()
        d = t64 - r64
        sums = (
            float((d * d).sum()),
            float((r64 * r64).sum()),
            float((t64 * t64).sum()),
            float((t64 * r64).sum()),
            float((d * r64).sum()),
            float(d.abs().max()),
            r.numel(),
        )
        for key in keys:
            a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
            for i in range(5):
                a[i] += sums[i]
            a[5] = max(a[5], sums[5])
            a[6] += sums[6]
    out = {}
    for key, (dd, rr, tt, tr, dr, mx, n) in sorted(acc.items()):
        out[key] = {
            "rel_err": math.sqrt(dd / rr) if rr > 0 else float("nan"),
            "cos": tr / math.sqrt(rr * tt) if rr > 0 and tt > 0 else float("nan"),
            "proj": dr / rr if rr > 0 else float("nan"),
            "max_abs_err": mx,
            "ref_norm": math.sqrt(rr),
            "numel": n,
        }
    return out


def fmt_table(cmp, keys=None):
    rows = []
    for key, m in cmp.items():
        if keys is not None and key not in keys:
            continue
        rows.append(
            f"  {key:<14} rel {m['rel_err']:.3e}  cos {m['cos']:.8f}  "
            f"proj {m['proj']:+.3e}  |ref| {m['ref_norm']:.3e}"
        )
    return "\n".join(rows)


# ---- 1.6 + 1.3: the whole-model error budget, and the bank gradient's structure ----


def bank_structure(main_grads, bank_names, vocab, sequences, ref_grads):
    """1.3's exact items on a Tier 2 `main_grad` computed over `sequences`:
    the rows of absent ids are exactly zero, the nonzero rows are exactly the
    input ids (not the targets), and nothing leaks into the row after a present
    one -- inductor's CPU miscompile spread row id onto id + 1 (§1.18). With a
    reference gradient, also the error on present rows and its correlation with
    the previous row's reference gradient (the same bug's signature)."""
    inputs = sequences[:, :-1].reshape(-1).cpu()
    targets = sequences[:, 1:].reshape(-1).cpu()
    present = torch.zeros(vocab, dtype=torch.bool)
    present[inputs] = True
    only_target = torch.zeros(vocab, dtype=torch.bool)
    only_target[targets] = True
    only_target &= ~present
    nxt = torch.zeros(vocab, dtype=torch.bool)
    nxt[1:] = present[:-1]
    nxt &= ~present
    results = []
    for g, name in zip(main_grads, bank_names):
        nonzero = g.abs().amax(dim=1) > 0
        entry = {
            "bank": name,
            "rows_present": int(present.sum()),
            "nonzero_rows": int(nonzero.sum()),
            "nonzero_absent_rows": int((nonzero & ~present).sum()),
            "zero_present_rows": int((~nonzero & present).sum()),
            "nonzero_only_target_rows": int((nonzero & only_target).sum()),
            "max_abs_in_next_absent_rows": float(g[nxt].abs().max()) if nxt.any() else 0.0,
        }
        ref = ref_grads.get(name)
        if ref is not None:
            e = (g - ref)[present]
            r = ref[present]
            entry["rel_err_present"] = float(e.norm() / r.norm())
            prev = torch.zeros_like(ref)
            prev[1:] = ref[:-1]
            p = prev[present]
            entry["corr_err_with_prev_row_ref"] = float(
                (e * p).sum() / (e.norm() * p.norm() + 1e-30)
            )
            # Controls: the next row (the bug cannot put row id + 1's gradient
            # into row id) and the row's own gradient (a systematic scale error).
            nxt_ref = torch.zeros_like(ref)
            nxt_ref[:-1] = ref[1:]
            q = nxt_ref[present]
            entry["corr_err_with_next_row_ref"] = float(
                (e * q).sum() / (e.norm() * q.norm() + 1e-30)
            )
            entry["corr_err_with_own_row_ref"] = float(
                (e * r).sum() / (e.norm() * r.norm() + 1e-30)
            )
            # And the reference gradients' own neighbour similarity, which any
            # proportional error inherits.
            entry["corr_ref_with_prev_row_ref"] = float(
                (r * p).sum() / (r.norm() * p.norm() + 1e-30)
            )
            entry["ref_nonzero_rows"] = int((ref.abs().amax(dim=1) > 0).sum())
        results.append(entry)
    return results


def run_budget(args, device):
    """1.6: each path's whole-model gradient against its fp32 reference, on 8
    micro-batches (256 sequences) of the first training batch and of the
    held-out set, at init and at the final checkpoint. The reference runs
    first; every other variant is compared as soon as it is computed (against
    the reference and against production), so at most three gradient sets are
    held at once. 1.3's structural checks run on ame_prod's main_grad."""
    data = load_data(args, device)
    n_seq = 256 if not args.smoke else 8
    splits = {"train": data["train"][:n_seq], "heldout": data["heldout"][:n_seq]}
    which = set(args.variants.split(",")) if args.variants else None
    tag = f"{args.arms.replace(',', '+')}_{args.states.replace(',', '+')}"
    path = os.path.join(args.out, f"budget_{tag}.json")
    report = {}
    for state in args.states.split(","):
        for arm in args.arms.split(","):
            names = [v for v in VARIANTS if VARIANTS[v][0] == arm and (which is None or v in which)]
            ref_name, prod_name = f"{arm}_ref32", f"{arm}_prod"
            order = [n for n in (ref_name, prod_name) if n in names] + [
                n for n in names if n not in (ref_name, prod_name)
            ]
            keep = {split: {} for split in splits}
            for split in splits:
                report[f"{arm}/{state}/{split}"] = {"loss": {}, "vs_ref32": {}, "vs_prod": {}}
            for variant in order:
                t0 = time.time()
                built = build(variant, device, args.smoke)
                if state == "final" and not args.smoke:
                    load_checkpoint(built, args.ckpt_root)
                for split, seqs in splits.items():
                    section = report[f"{arm}/{state}/{split}"]
                    with sdpa_math(variant == "dense_ref32"):
                        loss, g = gradients(built, seqs)
                    section["loss"][variant] = loss
                    if variant == prod_name and built.banks is not None:
                        slot = {id(getattr(m, a)): n for m, a, n in bank_slots(built.model)}
                        section["bank_structure"] = bank_structure(
                            [b.main_grad.detach().float().cpu() for b in built.banks.banks],
                            [slot[id(b)] for b in built.banks.banks],
                            built.cfg.common.vocab_size,
                            seqs,
                            keep[split].get(ref_name, {}),
                        )
                        log(f"{arm}/{state}/{split} bank structure: "
                            + json.dumps(summarize_structure(section["bank_structure"])))
                    if ref_name in keep[split] and variant != ref_name:
                        cmp = compare(g, keep[split][ref_name], by_layer=True)
                        section["vs_ref32"][variant] = cmp
                        log(f"{arm}/{state}/{split} {variant} vs {ref_name} (loss {loss:.6f} vs "
                            f"{section['loss'][ref_name]:.6f}):\n"
                            + fmt_table(cmp, [k for k in cmp if "@" not in k]))
                    if prod_name in keep[split] and variant not in (prod_name, ref_name):
                        section["vs_prod"][variant] = compare(g, keep[split][prod_name])
                    if variant in (ref_name, prod_name):
                        keep[split][variant] = g
                    else:
                        del g
                log(f"{variant} at {state}: {time.time() - t0:.0f} s")
                free(built)
                save_json(path, report)
            del keep
            free()
    save_json(path, report)


def summarize_structure(rows):
    keys = (
        "nonzero_absent_rows",
        "zero_present_rows",
        "nonzero_only_target_rows",
        "max_abs_in_next_absent_rows",
    )
    out = {k: max(r[k] for r in rows) for k in keys}
    if "rel_err_present" in rows[0]:
        out["rel_err_present_max"] = max(r["rel_err_present"] for r in rows)
        out["corr_err_prev_row_absmax"] = max(
            abs(r["corr_err_with_prev_row_ref"]) for r in rows
        )
    out["rows_present"] = rows[0]["rows_present"]
    return out


# ---- 1.1: final_ce, as the Trainer computes it, and precisely ------------------------


def run_final_ce(args, device):
    from hydra.utils import instantiate

    from main import initialize_training_components
    from src.core.utils import create_batch_fingerprint

    data = load_data(args, device)
    held = data["heldout"]
    path = os.path.join(args.out, "final_ce.json")
    report = {}
    for arm in args.arms.split(","):
        # (a) The Trainer's own final_eval on the checkpoint, loaded the way a
        # resume loads it (load_checkpoint_from_file: model, optimizer,
        # scheduler and, for AME, the bank sidecar).
        cfg = compose(
            arm,
            [f"trainer.checkpoint.load.path={os.path.join(args.ckpt_root, CKPT_DIRS[arm])}"],
        )
        torch._dynamo.reset()
        logger = quiet_logger()
        logger.set_step(None)
        model, optimizer, scheduler, training_state, logger = initialize_training_components(
            cfg, metric_logger=logger
        )
        trainer = instantiate(cfg.trainer)(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            metric_logger=logger,
        )
        # The dumped held-out set must be the batches final_eval reads.
        eval_batches = list(trainer.eval_dataloader)
        stacked = torch.cat(eval_batches).to(device)
        same = bool(torch.equal(stacked, held))
        t0 = time.time()
        trainer.final_eval()
        final_ce = logger.data["final_ce"][-1][0]
        log(f"{arm}: Trainer.final_eval final_ce {final_ce!r} ({time.time() - t0:.0f} s); dump identical: {same}")
        entry = {
            "final_ce_trainer": final_ce,
            "dump_equals_eval_dataloader": same,
            "eval_fingerprint": create_batch_fingerprint(eval_batches[0]),
        }
        # (b) The same model, precisely: bf16 logits as run, the CE in fp32 on
        # the upcast logits, accumulated in fp64 per token.
        model.eval()
        entry["ce_prod_fp32_loss"] = precise_ce(model, held, autocast=True, micro=32)
        log(f"{arm}: production forward, fp32 CE, fp64 sum: {entry['ce_prod_fp32_loss']:.6f}")
        free(trainer, model, optimizer, scheduler)
        # (c) The fp32 reference path on the same checkpoint: the common evaluator.
        ref = build(f"{arm}_ref32", device)
        load_checkpoint(ref, args.ckpt_root)
        ref.model.eval()
        with sdpa_math(arm == "dense"):
            entry["ce_ref32"] = precise_ce(ref.model, held, autocast=False, micro=8)
        log(f"{arm}: fp32 reference: {entry['ce_ref32']:.6f}")
        free(ref)
        report[arm] = entry
        save_json(path, report)
    if "ame" in report and "dense" in report:
        report["gap"] = {
            k: report["ame"][k] - report["dense"][k]
            for k in ("final_ce_trainer", "ce_prod_fp32_loss", "ce_ref32")
        }
        log(f"gap AME - dense: {report['gap']}")
    save_json(path, report)


@torch.no_grad()
def precise_ce(model, held, autocast, micro):
    total = torch.zeros((), dtype=torch.float64, device=held.device)
    count = 0
    for start in range(0, held.shape[0], micro):
        chunk = held[start : start + micro]
        inp, tgt = chunk[:, :-1], chunk[:, 1:]
        with autocast_ctx(autocast):
            logits = model(inp)
        losses = F.cross_entropy(
            logits.float().flatten(0, -2), tgt.reshape(-1).long(), reduction="none"
        )
        total += losses.double().sum()
        count += losses.numel()
    return float(total / count)


@contextlib.contextmanager
def sdpa_math(on):
    """Dense's fp32 oracle: SDPA's math backend. `attention_mechanism` pins its
    own backend list inside the call, so the module-level function is swapped."""
    if not on:
        yield
        return
    import src.core.model as model_module
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original = model_module.attention_mechanism

    def math_attention(query, key, value, causal):
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(query, key, value, is_causal=causal)

    model_module.attention_mechanism = math_attention
    try:
        yield
    finally:
        model_module.attention_mechanism = original


# ---- 1.2 + 1.4: one AME layer at a time ---------------------------------------------


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm()) if b.norm() > 0 else float("nan")


def proj(a, b):
    a, b = a.double(), b.double()
    return float(((a - b) * b).sum() / (b * b).sum()) if b.norm() > 0 else float("nan")


def capture_layers(built, chunk):
    """Each AME layer's real input (the normalized residual), the ids, and the
    upstream gradient at its output, from one eager production forward and
    backward on `chunk`."""
    store = {}
    handles = []
    for index, block in enumerate(built.model.encoder.blocks):
        layer = block.attention_layer.layer

        def hook(mod, args, out, index=index):
            store[index] = {"x": args[0].detach().float().cpu()}
            out.register_hook(lambda g, index=index: store[index].__setitem__("dy", g.detach().cpu()))

        handles.append(layer.register_forward_hook(hook))
    # In halves: the eager model does not fit a micro-batch of 32 (job 36320).
    # Each sequence's activations and upstream gradient are independent of the
    # others in its micro-batch; only the mean's 1/N differs, and halving N
    # doubles every gradient exactly (a power of two), so dy * 0.5 is what the
    # micro-batch of 32 produces, bit for bit.
    parts = {}
    for half in chunk.split(chunk.shape[0] // 2):
        store.clear()
        zero_all(built)
        loss = forward_loss(built, half)
        loss.backward()
        for index, item in store.items():
            parts.setdefault(index, []).append(item)
    for h in handles:
        h.remove()
    return {
        index: {
            "x": torch.cat([p["x"] for p in items]),
            "dy": torch.cat([p["dy"] for p in items]) * 0.5,
        }
        for index, items in parts.items()
    }


def layer_params(layer):
    names = ["alpha", "beta", "tau", "k_sink", "v_sink", "v_proj.weight", "o_proj.weight"]
    out = {}
    for name in names:
        obj = layer
        for part in name.split("."):
            obj = getattr(obj, part)
        out[name] = obj
    return out


def bank_rows_grad(layer, ids_unique):
    """The layer's bank gradient on the rows the batch touched: Tier 2's
    main_grad, or nn.Embedding's .grad."""
    out = {}
    for attr in ("q_bank", "k_bank"):
        bank = getattr(layer, attr)
        g = bank.main_grad if hasattr(bank, "main_grad") and bank.main_grad is not None else bank.weight.grad
        out[attr] = g[ids_unique].detach().float().cpu().clone()
    return out


def run_layer_path(layer, x, ids, dy, autocast, compiled, banks=None):
    """Forward and backward of one attention layer; returns output and grads."""
    for p in layer.parameters():
        p.grad = None
    if banks is not None:
        banks.zero_grad()
    x = x.clone().requires_grad_(True)
    fn = torch.compile(layer) if compiled else layer
    with autocast_ctx(autocast):
        out = fn(x, ids)
    out.backward(dy.to(out.dtype))
    ids_unique = torch.unique(ids)
    result = {
        "out": out.detach().float().cpu(),
        "dx": x.grad.detach().float().cpu(),
        "params": {k: (v.grad.detach().float().cpu().clone() if v.grad is not None else None) for k, v in layer_params(layer).items()},
        "banks": bank_rows_grad(layer, ids_unique),
    }
    return result


def kernel_checks(prod_layer, std_layer, x, ids, dy, device):
    """A4-A6 alone: the kernel inputs are the production prep's q, k, v
    (bf16), the upstream gradient is the true one at the kernel's output."""
    from src.core.ame import _flex_attention

    res = {}
    with torch.no_grad(), autocast_ctx(True):
        q, k, v = prod_layer._mixed_qkv(x, ids)
    q, k, v = (t.detach().contiguous() for t in (q, k, v))
    B, H, n, D = q.shape
    w_o = prod_layer.o_proj.weight.detach().float()
    d_attn = (dy.float() @ w_o).view(B, n, H, D).transpose(1, 2).contiguous()
    d_attn16 = d_attn.to(torch.bfloat16)

    def leaves(dtype):
        return [t.to(dtype).clone().requires_grad_(True) for t in (q, k, v)]

    def sink_grads(layer):
        return {
            "k_sink": layer.k_sink.grad.detach().float().cpu().clone(),
            "v_sink": layer.v_sink.grad.detach().float().cpu().clone(),
        }

    # flex + the LSE merge, compiled together as inside the block.
    for p in (prod_layer.k_sink, prod_layer.v_sink):
        p.grad = None
    qf, kf, vf = leaves(torch.bfloat16)
    flex_fn = torch.compile(prod_layer._attend_flex)
    out_f = flex_fn(qf, kf, vf)
    out_f.backward(d_attn16)
    flex = {"out": out_f.detach().float(), "dq": qf.grad.float(), "dk": kf.grad.float(), "dv": vf.grad.float(), **{k: v.to(device) for k, v in sink_grads(prod_layer).items()}}

    # The exact function of the same bf16 inputs, in fp32 (reference math).
    for p in (std_layer.k_sink, std_layer.v_sink):
        p.grad = None
    q3, k3, v3 = leaves(torch.float32)
    out_r = std_layer._attend_reference(q3, k3, v3)
    out_r.backward(d_attn16.float())
    ref = {"out": out_r.detach(), "dq": q3.grad, "dk": k3.grad, "dv": v3.grad, **{k: v.to(device) for k, v in sink_grads(std_layer).items()}}

    # A plain bf16 implementation of the same attention (bf16 einsums, fp32 softmax).
    for p in (std_layer.k_sink, std_layer.v_sink):
        p.grad = None
    q1, k1, v1 = leaves(torch.bfloat16)
    out_1 = std_layer._attend_reference(q1, k1, v1)
    out_1.backward(d_attn16)
    ref16 = {"out": out_1.detach().float(), "dq": q1.grad.float(), "dk": k1.grad.float(), "dv": v1.grad.float(), **{k: v.to(device) for k, v in sink_grads(std_layer).items()}}

    res["flex_vs_ref32"] = {key: rel(flex[key], ref[key]) for key in ref}
    res["flex_vs_ref32_proj"] = {key: proj(flex[key], ref[key]) for key in ref}
    res["ref16_vs_ref32"] = {key: rel(ref16[key], ref[key]) for key in ref}
    res["ref16_vs_ref32_proj"] = {key: proj(ref16[key], ref[key]) for key in ref}
    # Per head, the worst dq: a head-local backward bug would show here first.
    res["flex_vs_ref32_dq_per_head"] = [rel(flex["dq"][:, h], ref["dq"][:, h]) for h in range(H)]

    # Dense's kernel on the same inputs (causal, no bias, no sink): SDPA flash
    # in bf16 against SDPA math in fp32. The rounding budget of the standard
    # kernel on AME's actual q, k, v.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    qa, ka, va = leaves(torch.bfloat16)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out_a = F.scaled_dot_product_attention(qa, ka, va, is_causal=True, scale=1.0)
    out_a.backward(d_attn16)
    qb, kb, vb = leaves(torch.float32)
    with sdpa_kernel([SDPBackend.MATH]):
        out_b = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, scale=1.0)
    out_b.backward(d_attn16.float())
    res["sdpa_flash_vs_math32"] = {
        "out": rel(out_a.float(), out_b),
        "dq": rel(qa.grad.float(), qb.grad),
        "dk": rel(ka.grad.float(), kb.grad),
        "dv": rel(va.grad.float(), vb.grad),
    }

    # The LSE gradient alone: L = sum(w * lse). If flex's LSE carried no
    # gradient, dq and dk here would be zero.
    g = torch.Generator(device=device).manual_seed(1)
    w = torch.randn(B, H, n, device=device, generator=g)
    ql, kl, vl = leaves(torch.bfloat16)
    seq_fn = torch.compile(lambda a, b, c: prod_layer._attend_flex_sequence(a, b, c, return_lse=True))
    _, lse = seq_fn(ql, kl, vl)
    (w * lse.float()).sum().backward()
    qm, km, vm = leaves(torch.float32)
    scores = torch.einsum("bhid,bhjd->bhij", qm, km)
    scores = scores + std_layer.score_bias(n, device, torch.float32, 0).unsqueeze(0)
    lse_ref = torch.logsumexp(scores, dim=-1)
    (w * lse_ref).sum().backward()
    res["lse"] = {
        "lse": rel(lse.float().detach(), lse_ref.detach()),
        "dq": rel(ql.grad.float(), qm.grad),
        "dk": rel(kl.grad.float(), km.grad),
        "dq_norm_flex": float(ql.grad.float().norm()),
        "dq_norm_ref": float(qm.grad.norm()),
    }
    del scores

    # The sink's weight per head, from the exact computation.
    with torch.no_grad():
        k_sink = std_layer.k_sink.float().view(1, H, 1, D)
        s = (q.float() * k_sink).sum(-1)
        sc = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) + std_layer.score_bias(n, device, torch.float32, 0).unsqueeze(0)
        lse_seq = torch.logsumexp(sc, dim=-1)
        sink_w = torch.sigmoid(s - lse_seq)
        res["sink_weight_per_head"] = sink_w.mean(dim=(0, 2)).cpu().tolist()
        res["sink_weight_mean"] = float(sink_w.mean())
        del sc
    return res


def prep_checks(prod_layer, std_layer, x, ids, banks):
    """1.4: the prep (gather, normalize, mix, tau) in bf16 (Tier 2's compute
    copy) against the same prep in fp32 (fp32 banks), and both against fp32
    without autocast; then the backward through each with one upstream."""
    res = {}
    with autocast_ctx(True):
        qp, kp, vp = torch.compile(prod_layer._mixed_qkv)(x, ids)
        qs, ks, vs = std_layer._mixed_qkv(x, ids)
    q32, k32, v32 = std_layer._mixed_qkv(x, ids)
    for name, (a, b, c) in {"q": (qp, qs, q32), "k": (kp, ks, k32)}.items():
        res[f"{name}_prod_vs_fp32"] = rel(a.detach().float(), c.detach())
        res[f"{name}_std_vs_fp32"] = rel(b.detach().float(), c.detach())
        res[f"{name}_prod_ne_std_fraction"] = float((a.detach() != b.detach()).float().mean())
    tau = std_layer.tau.detach().float()
    tau16 = tau.to(torch.bfloat16).float()
    res["tau"] = tau.cpu().tolist()
    res["tau_bf16_rel_err_max"] = float(((tau16 - tau).abs() / tau).max())
    # Backward: one fixed upstream for q and k (unit-variance noise scaled to
    # each tensor), through the prod prep and the fp32 prep.
    # bf16-representable, so every path receives the identical upstream.
    g = torch.Generator(device=x.device).manual_seed(2)
    up_q = torch.randn(q32.shape, device=x.device, generator=g).to(torch.bfloat16).float()
    up_k = torch.randn(k32.shape, device=x.device, generator=g).to(torch.bfloat16).float()
    out = {}
    for label, layer, auto, compiled in (
        ("prod", prod_layer, True, True),
        ("std", std_layer, True, False),
        ("fp32", std_layer, False, False),
    ):
        for p in layer.parameters():
            p.grad = None
        if label == "prod":
            banks.zero_grad()
        fn = torch.compile(layer._mixed_qkv) if compiled else layer._mixed_qkv
        with autocast_ctx(auto):
            q, k, _ = fn(x, ids)
        torch.autograd.backward([q, k], [up_q.to(q.dtype), up_k.to(k.dtype)])
        ids_unique = torch.unique(ids)
        out[label] = {
            "alpha": layer.alpha.grad.detach().float().clone(),
            "tau": layer.tau.grad.detach().float().clone(),
            **{a: v.to(x.device) for a, v in bank_rows_grad(layer, ids_unique).items()},
        }
    for label in ("prod", "std"):
        res[f"grad_{label}_vs_fp32"] = {k: rel(out[label][k], out["fp32"][k]) for k in out["fp32"]}
        res[f"grad_{label}_vs_fp32_proj"] = {k: proj(out[label][k], out["fp32"][k]) for k in out["fp32"]}
    return res


def run_layers(args, device):
    data = load_data(args, device)
    micro = 32 if not args.smoke else 4
    chunk = data["heldout"][:micro]
    prod = build("ame_prod_eager", device, args.smoke)
    std = build("ame_std", device, args.smoke)  # flex on CUDA; reference set per path
    if not args.smoke:
        load_checkpoint(prod, args.ckpt_root)
        load_checkpoint(std, args.ckpt_root)
    store = capture_layers(prod, chunk)
    ids = chunk[:, :-1]
    layers = range(len(prod.model.encoder.blocks)) if args.layers is None else [int(i) for i in args.layers.split(",")]
    report = {}
    for index in layers:
        t0 = time.time()
        x = store[index]["x"].to(device)
        dy = store[index]["dy"].to(device)
        pl = prod.model.encoder.blocks[index].attention_layer.layer
        sl = std.model.encoder.blocks[index].attention_layer.layer
        entry = {}
        paths = {}
        paths["prod_compiled"] = run_layer_path(pl, x, ids, dy, True, device.type == "cuda", prod.banks)
        paths["prod_eager"] = run_layer_path(pl, x, ids, dy, True, False, prod.banks)
        sl.backend = "auto"
        paths["std_compiled"] = run_layer_path(sl, x, ids, dy, True, device.type == "cuda")
        sl.backend = "reference"
        paths["ref16"] = run_layer_path(sl, x, ids, dy, True, False)
        paths["ref32"] = run_layer_path(sl, x, ids, dy, False, False)
        sl.backend = "auto"
        ref = paths["ref32"]
        for label, path in paths.items():
            if label == "ref32":
                continue
            entry[label] = {
                "out": rel(path["out"], ref["out"]),
                "dx": rel(path["dx"], ref["dx"]),
                "dx_proj": proj(path["dx"], ref["dx"]),
                **{f"d{k}": rel(v, ref["params"][k]) for k, v in path["params"].items() if v is not None},
                **{f"d{k}_proj": proj(v, ref["params"][k]) for k, v in path["params"].items() if v is not None},
                **{f"d{k}": rel(v, ref["banks"][k]) for k, v in path["banks"].items()},
                **{f"d{k}_proj": proj(v, ref["banks"][k]) for k, v in path["banks"].items()},
            }
        if device.type == "cuda":
            entry["kernel"] = kernel_checks(pl, sl, x, ids, dy, device)
            entry["prep"] = prep_checks(pl, sl, x, ids, prod.banks)
        report[index] = entry
        log(
            f"layer {index} ({time.time() - t0:.0f} s): "
            + ", ".join(f"{k}: out {v['out']:.2e} dx {v['dx']:.2e} dq_bank {v.get('dq_bank', float('nan')):.2e} dtau {v.get('dtau', float('nan')):.2e}" for k, v in entry.items() if isinstance(v, dict) and "out" in v and k != "kernel")
        )
        if "kernel" in entry:
            kc = entry["kernel"]
            log(
                f"  kernel flex vs ref32: {json.dumps({k: round(v, 6) for k, v in kc['flex_vs_ref32'].items()})}; "
                f"ref16: {json.dumps({k: round(v, 6) for k, v in kc['ref16_vs_ref32'].items()})}; "
                f"sdpa flash: {json.dumps({k: round(v, 6) for k, v in kc['sdpa_flash_vs_math32'].items()})}; "
                f"lse: {json.dumps({k: round(v, 6) for k, v in kc['lse'].items()})}; sink weight {kc['sink_weight_mean']:.4f}"
            )
            pc = entry["prep"]
            log(f"  prep: {json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pc.items() if not isinstance(v, (list, dict))})}")
            log(f"  prep grads prod: {json.dumps({k: round(v, 6) for k, v in pc['grad_prod_vs_fp32'].items()})}; std: {json.dumps({k: round(v, 6) for k, v in pc['grad_std_vs_fp32'].items()})}")
        save_json(os.path.join(args.out, "layers.json"), report)
        del paths
        free()


# ---- 1.5: one optimizer step, identical gradients ------------------------------------


def run_optim(args, device):
    """Tier 2 and the ordinary path take one optimizer step from the same state
    with the same gradients. Everything downstream of the gradient -- the joint
    clip, the two AdamWs (foreach on the masters, fused on the backbone), the
    master -> compute-copy refresh -- must agree to fp32 rounding."""
    from src.core.memory_parallel import _find_banks
    from src.core.trainer import Trainer

    data = load_data(args, device)
    seqs = data["train"][: (64 if not args.smoke else 8)]
    report = {}
    for state in args.states.split(","):
        if args.smoke and state == "final":
            break
        log(f"=== optim {state}")
        t2 = build("ame_prod", device, args.smoke)
        ordn = build("ame_std", device, args.smoke)
        if state == "final":
            load_checkpoint(t2, args.ckpt_root, with_optimizer=True)
            load_checkpoint(ordn, args.ckpt_root)
            copy_optimizer_state(t2, ordn)
        else:
            with torch.no_grad():
                for (m, a, _), bank in zip(bank_slots(ordn.model), t2.banks.banks):
                    assert torch.equal(getattr(m, a).weight, bank.master)
        # One set of gradients, from the production path.
        loss, grads = gradients(t2, seqs)
        log(f"loss {loss:.5f}")
        ordinary_slots = {name: getattr(m, a).weight for m, a, name in bank_slots(ordn.model)}
        with torch.no_grad():
            for name, p in ordn.model.named_parameters():
                p.grad = grads[name].to(device).clone()
        # The clip, exactly as Trainer.clip_gradient runs it on each path.
        stub_t2 = types.SimpleNamespace(model=t2.model, gradient_clipping=1.0, _banks=t2.banks)
        stub_or = types.SimpleNamespace(model=ordn.model, gradient_clipping=1.0, _banks=None)
        norm_t2 = Trainer.clip_gradient(stub_t2)
        norm_or = Trainer.clip_gradient(stub_or)
        entry = {"total_norm_tier2": float(norm_t2), "total_norm_ordinary": float(norm_or)}
        entry["total_norm_rel_diff"] = abs(float(norm_t2) - float(norm_or)) / float(norm_or)
        lr = t2.optimizer.param_groups[0]["lr"]
        assert lr == ordn.optimizer.param_groups[0]["lr"]
        t2.optimizer.step()
        t2.banks.step(lr=lr)
        ordn.optimizer.step()
        # Compare parameters, moments, and the compute copy.
        params_t2 = dict(t2.model.named_parameters())
        worst = {"param": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
        worst_name = {}
        slot_of = {id(b): n for (m, a, n), b in zip(bank_slots(t2.model), t2.banks.banks)}
        for name, p_or in ordn.model.named_parameters():
            if name in ordinary_slots and ordinary_slots[name] is p_or:
                bank = next(b for b in t2.banks.banks if slot_of[id(b)] == name)
                p_t2 = bank.master
                st_t2 = t2.banks.optimizer.state[bank.master]
            else:
                p_t2 = params_t2[name]
                st_t2 = t2.optimizer.state[p_t2]
            st_or = ordn.optimizer.state[p_or]
            for key, a, b in (
                ("param", p_t2, p_or),
                ("exp_avg", st_t2["exp_avg"], st_or["exp_avg"]),
                ("exp_avg_sq", st_t2["exp_avg_sq"], st_or["exp_avg_sq"]),
            ):
                denom = b.double().abs().max().clamp_min(1e-30)
                diff = float((a.double() - b.double()).abs().max() / denom)
                if diff > worst[key]:
                    worst[key], worst_name[key] = diff, name
        entry["max_rel_diff"] = worst
        entry["max_rel_diff_where"] = worst_name
        copies = [torch.equal(b.weight, b.master.to(b.weight.dtype)) for b in t2.banks.banks]
        entry["compute_copy_equals_bf16_master"] = all(copies)
        entry["lr"] = lr
        log(json.dumps(entry))
        report[state] = entry
        free(t2, ordn, grads)
        save_json(os.path.join(args.out, "optim.json"), report)


def copy_optimizer_state(t2, ordn):
    """The final checkpoint's moments into the ordinary path's fused AdamW:
    backbone moments by parameter name, bank moments from the sidecar."""
    state = t2.checkpoint_state["optim"]
    # Rebuild the Tier 2 backbone optimizer's state from the checkpoint (the
    # sidecar already restored the bank optimizer).
    t2.optimizer.load_state_dict(state)
    named_t2 = {id(p): n for n, p in t2.model.named_parameters()}
    by_name = {}
    for p, st in t2.optimizer.state.items():
        by_name[named_t2[id(p)]] = st
    slots = bank_slots(t2.model)
    bank_name = {id(b): n for (m, a, n), b in zip(slots, t2.banks.banks)}
    for b in t2.banks.banks:
        by_name[bank_name[id(b)]] = t2.banks.optimizer.state[b.master]
    for name, p in ordn.model.named_parameters():
        st = by_name[name]
        ordn.optimizer.state[p] = {
            "step": torch.tensor(float(st["step"]), dtype=torch.float32, device=p.device),
            "exp_avg": st["exp_avg"].detach().clone().to(p.device),
            "exp_avg_sq": st["exp_avg_sq"].detach().clone().to(p.device),
        }
    # Same LR on both: the scheduler state of the checkpoint.
    t2.scheduler.load_state_dict(t2.checkpoint_state["scheduler"])
    ordn.scheduler.load_state_dict(t2.checkpoint_state["scheduler"])
    for opt_from, opt_to in ((t2.optimizer, ordn.optimizer),):
        lr = opt_from.param_groups[0]["lr"]
        for group in opt_to.param_groups:
            group["lr"] = lr


# ---- 1.7: machinery -------------------------------------------------------------------


def run_machinery(args, device):
    from torch.nn.attention.flex_attention import create_block_mask

    data = load_data(args, device)
    built = build("ame_prod", device, args.smoke)
    report = {"state": "init"}
    if not args.smoke and "final" in args.states.split(","):
        load_checkpoint(built, args.ckpt_root)
        report["state"] = "final"
    seq_len = data["heldout"].shape[1] - 1
    # Warm the cache the way the compiled run did: a training micro-step.
    loss = forward_loss(built, data["heldout"][:2])
    loss.backward()
    layer = built.model.encoder.blocks[0].attention_layer.layer
    cached = layer._block_mask_cache.get((seq_len, 0, str(device)))
    if device.type == "cuda":
        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx

        fresh = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
        same = cached is not None and all(
            torch.equal(getattr(cached, f), getattr(fresh, f))
            for f in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices")
        )
        dense = fresh.to_dense()
        report["block_mask_cached_equals_fresh"] = bool(same)
        report["block_mask_cache_keys"] = [str(k) for k in layer._block_mask_cache]
        report["block_mask_dense_shape"] = list(dense.shape)
    # The instrumentation: no RNG, no parameter, buffer or cache change.
    ids = data["heldout"][:4, :-1]
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state() if device.type == "cuda" else None
    before = {n: t.detach().clone() for n, t in built.model.state_dict().items()}
    masters = [b.master.clone() for b in built.banks.banks]
    main_grads = [b.main_grad.clone() for b in built.banks.banks]
    cache_keys = {i: dict(b.attention_layer.layer._block_mask_cache) for i, b in enumerate(built.model.encoder.blocks)}
    readouts = {}
    for i, block in enumerate(built.model.encoder.blocks):
        readouts[i] = block.attention_layer.layer.instrumentation(ids)
    after = built.model.state_dict()
    report["instrumentation_state_unchanged"] = all(torch.equal(before[n], after[n]) for n in before)
    report["instrumentation_masters_unchanged"] = all(torch.equal(a, b.master) for a, b in zip(masters, built.banks.banks))
    report["instrumentation_main_grad_unchanged"] = all(torch.equal(a, b.main_grad) for a, b in zip(main_grads, built.banks.banks))
    report["instrumentation_rng_cpu_unchanged"] = bool(torch.equal(rng_cpu, torch.get_rng_state()))
    if rng_cuda is not None:
        report["instrumentation_rng_cuda_unchanged"] = bool(torch.equal(rng_cuda, torch.cuda.get_rng_state()))
    report["instrumentation_mask_cache_unchanged"] = all(
        set(cache_keys[i]) == set(b.attention_layer.layer._block_mask_cache)
        and all(cache_keys[i][k] is b.attention_layer.layer._block_mask_cache[k] for k in cache_keys[i])
        for i, b in enumerate(built.model.encoder.blocks)
    )
    report["readout_layer0"] = readouts[0]
    log(json.dumps({k: v for k, v in report.items() if k != "readout_layer0"}))
    save_json(os.path.join(args.out, "machinery.json"), report)


# ---- main -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["dump_data", "final_ce", "budget", "layers", "optim", "machinery"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=os.path.expanduser("~/parity/data/parity_batches.pt"))
    parser.add_argument("--ckpt-root", default=os.path.expanduser("~/parity_ckpt"))
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--variants", default=None, help="budget: comma-separated subset")
    parser.add_argument("--arms", default="dense,ame", help="budget: dense, ame or both")
    parser.add_argument("--states", default="init,final", help="budget: init, final or both")
    parser.add_argument("--layers", default=None, help="layers: comma-separated subset")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import main as main_module

    main_module.setup_enviroment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import subprocess

    commit_file = os.path.join(REPO, "PARITY_COMMIT")
    if os.path.exists(commit_file):
        commit = open(commit_file).read().strip()
    else:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    meta = {
        "check": args.check,
        "commit": commit,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    log(json.dumps(meta))
    assert not torch.backends.cuda.matmul.allow_tf32, "fp32 references need TF32 off"
    save_json(os.path.join(args.out, f"meta_{args.check}.json"), meta)
    torch._dynamo.config.cache_size_limit = 64
    {
        "dump_data": dump_data,
        "final_ce": run_final_ce,
        "budget": run_budget,
        "layers": run_layers,
        "optim": run_optim,
        "machinery": run_machinery,
    }[args.check](*( (args,) if args.check == "dump_data" else (args, device)))
    log("done")


if __name__ == "__main__":
    main()
