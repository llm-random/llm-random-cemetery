"""Rebuttal E4: how much a trained PC(D) model relies on the channels its initial
selection excluded.

Loads a PC/PCD nano checkpoint (DCP `model/` folder written by PCTrainer.save_checkpoint)
into the model built from the training job's own config, and scores held-out FineWeb-Edu
cross-entropy twice:
  (i)  W_c as trained;
  (ii) W_c rebuilt after zeroing every projection entry that reads a channel the initial
       one-hot selection excluded (P_in rows, P_out columns and embedding-projection
       columns outside the top-k importances): the projections may then only mix
       retained channels, as at initialisation, while W_r and everything else keep their
       trained values.
It also reports the share of each projection's squared norm that sits on excluded
channels. This measures reliance, not causation; E3 is the causal test. On an E3
(discarded-block-masked) checkpoint (i) and (ii) must agree exactly: a sanity check.

    torchrun --nproc-per-node=1 scripts/rebuttal/e4_knockout.py \
        --job-config rebuttal_jobs/<job>/config_0.yaml --ckpt <save>/<jobid>/0/step_2047 \
        --out <json> [--n-samples 2048]
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.distributed.checkpoint as dcp  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.distributed.tensor import DTensor, distribute_tensor  # noqa: E402

import resolver  # noqa: E402,F401
from main import distributed_setup, setup_enviroment  # noqa: E402
from src.core.datasets import get_dataloader  # noqa: E402
from src.projected_compression.initialization import (  # noqa: E402
    create_model,
    get_topk_indices,
)


def keep_vectors(cfg, model):
    proj = model.projections
    d_idx, f_idx = get_topk_indices(
        cfg.projected_compression.path_to_importances, proj.target_dmodel, proj.target_dff
    )
    base_d = proj.embedding.shape[1]
    base_f = proj.blocks[0].compressible_ff_pre.base_out_features
    keep_d = torch.zeros(base_d, dtype=torch.bool)
    keep_d[d_idx.cpu()] = True
    keep_f = []
    for idx in f_idx:
        k = torch.zeros(base_f, dtype=torch.bool)
        k[idx.cpu()] = True
        keep_f.append(k)
    return keep_d, keep_f


def projection_sides(model, keep_d, keep_f):
    """(name, parameter, kind, keep) for every projection matrix.
    kind: "rows" = base dimension on rows (P_in), "cols" = base dimension on columns
    (P_out, embedding projection)."""
    out = [("embedding", model.projections.embedding, "cols", keep_d)]
    for i, b in enumerate(model.projections.blocks):
        for name, lin, k_in, k_out in [
            ("q", b.compressible_q, keep_d, None),
            ("k", b.compressible_k, keep_d, None),
            ("v", b.compressible_v, keep_d, None),
            ("o", b.compressible_o, None, keep_d),
            ("ff_pre", b.compressible_ff_pre, keep_d, keep_f[i]),
            ("ff_gate", b.compressible_ff_gate, keep_d, keep_f[i]),
            ("ff_post", b.compressible_ff_post, keep_f[i], keep_d),
        ]:
            if hasattr(lin, "projection_in_weight"):
                out.append((f"{i}.{name}.in", lin.projection_in_weight, "rows", k_in))
            if hasattr(lin, "projection_out_weight"):
                out.append((f"{i}.{name}.out", lin.projection_out_weight, "cols", k_out))
    out.append(("head.in", model.projections.head.projection_in_weight, "rows", keep_d))
    return out


def _full(t):
    return t.full_tensor() if isinstance(t, DTensor) else t


def keep_matrix(p, kind, keep):
    local = p.to_local() if isinstance(p, DTensor) else p
    k = keep.to(local.device, torch.float32)
    m = (k[:, None] if kind == "rows" else k[None, :]).expand(p.shape).contiguous()
    m = m.to(p.dtype)
    if isinstance(p, DTensor):
        m = distribute_tensor(m, p.device_mesh, p.placements)
    return m


@torch.no_grad()
def excluded_mass(sides):
    """Share of squared projection norm on excluded channels, per matrix type."""
    acc = {}
    for name, p, kind, keep in sides:
        full = _full(p.detach()).float().cpu()
        k = keep[:, None] if kind == "rows" else keep[None, :]
        excl = (full[~k.expand(full.shape)] ** 2).sum().item()
        tot = (full**2).sum().item()
        typ = name.split(".")[1] if name[0].isdigit() else name.split(".")[0]
        a = acc.setdefault(typ, [0.0, 0.0])
        a[0] += excl
        a[1] += tot
    return {t: (e / s if s else 0.0) for t, (e, s) in acc.items()}


@torch.no_grad()
def heldout_ce(model, loader, n_batches, micro):
    model.eval()
    model.prepare_compressed_weights()
    loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
    n_tok = torch.zeros((), dtype=torch.float64, device="cuda")
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = batch.to("cuda")
        for chunk in batch.split(micro):
            inp, tgt = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
            logits = model(inp)
            loss_sum += F.cross_entropy(
                logits.float().flatten(0, 1), tgt.flatten(), reduction="sum"
            ).double()
            n_tok += tgt.numel()
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(n_tok)
    return (loss_sum / n_tok).item(), int(n_tok.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-config", required=True)
    ap.add_argument("--ckpt", required=True, help="step folder holding model/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=16)
    args = ap.parse_args()

    setup_enviroment()
    distributed_setup()
    t0 = time.time()
    cfg = OmegaConf.load(args.job_config)
    model = create_model(
        cfg.model,
        cfg.projected_compression,
        cfg.projected_compression.source_model_for_distillation,
    )
    dcp.load(model.state_dict(), checkpoint_id=os.path.join(args.ckpt, "model"))

    ds = OmegaConf.to_container(cfg.trainer.train_dataloader.dataset, resolve=True)
    ds.update(
        holdout="eval",
        holdout_rows=262144,
        max_samples=args.n_samples,
        world_size_independent=True,
        shuffle=True,
        seed=123,
    )
    dataset = instantiate(ds)
    world = int(os.environ.get("WORLD_SIZE", 1))
    loader = get_dataloader(dataset, total_batch_size=args.batch * world, num_workers=0)
    n_batches = args.n_samples // (args.batch * world)

    keep_d, keep_f = keep_vectors(cfg, model)
    sides = projection_sides(model, keep_d, keep_f)
    mass = excluded_mass(sides)
    ce_trained, ntok = heldout_ce(model, loader, n_batches, args.micro)

    with torch.no_grad():
        for _, p, kind, keep in sides:
            p.mul_(keep_matrix(p, kind, keep))
    ce_knockout, _ = heldout_ce(model, loader, n_batches, args.micro)

    res = dict(
        ckpt=args.ckpt,
        job_config=args.job_config,
        n_samples=args.n_samples,
        n_tokens=ntok,
        ce_trained=ce_trained,
        ce_knockout=ce_knockout,
        delta=ce_knockout - ce_trained,
        excluded_mass_share=mass,
        seconds=time.time() - t0,
    )
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
