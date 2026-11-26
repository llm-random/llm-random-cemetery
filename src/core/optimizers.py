"""Parameter roles and the one optimizer builder main.py calls (line 2 of the plan)."""

import logging
import os
from collections import defaultdict

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.distributed.tensor import DTensor, Shard
from torch.optim.adam import adam

from src.core.model import TransformerHead

logger = logging.getLogger(__name__)

# Order of the param groups; the role table in the plan.
ROLES = ("vector", "embedding", "head", "hidden")


def param_roles(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Each parameter's role from its shape and owning module; first rule wins."""
    embedding = {
        id(p)
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    head = {
        id(p)
        for m in model.modules()
        if isinstance(m, TransformerHead)
        for p in m.parameters()
    }
    roles = {role: [] for role in ROLES}
    for p in model.parameters():
        if p.ndim < 2:
            roles["vector"].append(p)
        elif id(p) in embedding:
            roles["embedding"].append(p)
        elif id(p) in head:
            roles["head"].append(p)
        else:
            roles["hidden"].append(p)
    return roles


def newton_schulz(G, steps):
    """Keller Jordan's quintic Newton-Schulz, in G's dtype: roughly G's orthogonal factor U V^T.

    Singular values come out near 1, not at 1 (github.com/KellerJordan/Muon). G may be a
    stack of matrices along dim 0.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tall else X


def _orthogonalize(matrices, steps):
    """newton_schulz on each matrix, one batched call per shape."""
    by_shape = defaultdict(list)
    for i, m in enumerate(matrices):
        by_shape[m.shape].append(i)
    out = [None] * len(matrices)
    for indices in by_shape.values():
        batch = newton_schulz(torch.stack([matrices[i] for i in indices]), steps)
        for i, m in zip(indices, batch):
            out[i] = m
    return out


def _local(t):
    """A DTensor's own shard (in place under no_grad), or t."""
    return t.to_local() if isinstance(t, DTensor) else t


def _gather(local, p):
    """The whole matrix from every rank's rows of it (one all-gather), shaped like p."""
    if not isinstance(p, DTensor):
        return local
    return DTensor.from_local(
        local, p.device_mesh, p.placements, shape=p.shape, stride=p.stride()
    ).full_tensor()


def _own_rows(full, p):
    """This rank's rows of a whole matrix, split as Shard(0) splits p (torch.chunk)."""
    if not isinstance(p, DTensor):
        return full
    mesh = p.device_mesh
    chunks = full.chunk(mesh.size(), dim=0)
    rank = mesh.get_local_rank()
    return chunks[rank] if rank < len(chunks) else full[:0]


class MuonWithAdamW(torch.optim.AdamW):
    """Muon on groups with `use_muon`, torch's AdamW on the rest, as one optimizer.

    Muon is Keller Jordan's (Nesterov momentum, quintic Newton-Schulz in bf16 on GPU) with
    Moonlight's update scale 0.2 * sqrt(max(rows, cols)) (arXiv 2502.16982): it matches
    AdamW's update RMS, so Muon groups take AdamW's LR and decoupled weight decay.
    Under FSDP2 each rank gathers every whole update, orthogonalizes it and keeps its rows.
    """

    def __init__(self, params, lr, weight_decay, momentum, ns_steps, fused):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        # Read by add_param_group, which AdamW's __init__ calls for each group.
        self._muon_defaults = dict(use_muon=False, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, lr=lr, weight_decay=weight_decay, fused=fused)

    def add_param_group(self, param_group):
        for key, value in self._muon_defaults.items():
            param_group.setdefault(key, value)
        if param_group["use_muon"]:
            for p in param_group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon takes 2D matrices, got {tuple(p.shape)}")
                if isinstance(p, DTensor) and p.placements != (Shard(0),):
                    raise ValueError(f"Muon takes FSDP2's Shard(0), got {p.placements}")
        super().add_param_group(param_group)

    def __setstate__(self, state):
        # Skip Adam's fix-up for old checkpoints: it reads state["step"], which Muon has not.
        torch.optim.Optimizer.__setstate__(self, state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _adamw_step(self, group):
        """torch 2.7.1's Adam.step for one group, as torch.optim.AdamW runs it."""
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = (
            [] for _ in range(6)
        )
        has_complex = self._init_group(
            group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps
        )
        beta1, beta2 = group["betas"]
        adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            steps,
            amsgrad=group["amsgrad"],
            has_complex=has_complex,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )

    def _muon_step(self, group):
        lr, beta = group["lr"], group["momentum"]
        params = [p for p in group["params"] if p.grad is not None]
        updates = []
        for p in params:
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            grad, buf = _local(p.grad), _local(state["momentum_buffer"])
            buf.lerp_(grad, 1 - beta)
            # Nesterov, cast before the gather; CPU bf16 matmuls are ~40x slower than fp32.
            dtype = torch.bfloat16 if grad.is_cuda else torch.float32
            updates.append(_gather(grad.lerp(buf, beta).to(dtype), p))
        for p, update in zip(params, _orthogonalize(updates, group["ns_steps"])):
            weight = _local(p)
            weight.mul_(1 - lr * group["weight_decay"])
            weight.add_(_own_rows(update, p), alpha=-lr * 0.2 * max(p.shape) ** 0.5)


def log_param_groups(model, groups):
    """Each group's optimizer, LR, decay and parameters, once, on rank 0."""
    if os.environ.get("RANK", "0") != "0":
        return
    names = {id(p): name for name, p in model.named_parameters()}
    for group in groups:
        n_params = sum(p.numel() for p in group["params"])
        logger.info(
            f"param group {group['role']}: "
            f"{'muon' if group.get('use_muon') else 'adamw'}, lr {group['lr']:.4g}, "
            f"weight_decay {group['weight_decay']:g}, {len(group['params'])} tensors, "
            f"{n_params:,} params"
        )
        for p in group["params"]:
            logger.info(f"  {names[id(p)]} {tuple(p.shape)}")


def build_optimizer_and_scheduler(cfg, model, learning_rate):
    roles = cfg.optimizer.roles
    base_dmodel = cfg.optimizer.base_dmodel
    muon = cfg.optimizer.muon
    if roles is None:
        assert base_dmodel is None, "simpleP (optimizer.base_dmodel) needs param groups"
        assert muon is None, "Muon (optimizer.muon) needs param groups"
        params = model.parameters()
    else:
        # simpleP's LR / m is AdamW's rule; Muon's width rule is line 2, task 4.
        assert (
            muon is None or base_dmodel is None
        ), "Muon takes no optimizer.base_dmodel"
        # simpleP (muP's LR rule): hidden and head LR / m, m = dmodel / base_dmodel.
        scale = 1 if base_dmodel is None else base_dmodel / cfg.common.dmodel
        params = [
            {
                "params": ps,
                "role": role,
                "lr": learning_rate * (scale if role in ("hidden", "head") else 1),
                "weight_decay": roles[role].weight_decay,
            }
            for role, ps in param_roles(model).items()
            if ps
        ]
        if muon is not None:
            for group in params:
                group["use_muon"] = group["role"] == "hidden"
        log_param_groups(model, params)
    if muon is None:
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            # The fused kernel exists only for CUDA tensors; CPU runs keep the
            # default path, which also keeps the seeded CPU gates bit-identical.
            fused=torch.cuda.is_available(),
        )
    else:
        optimizer = MuonWithAdamW(
            params,
            lr=learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            momentum=muon.momentum,
            ns_steps=muon.ns_steps,
            fused=torch.cuda.is_available(),
        )
    scheduler = instantiate(cfg.trainer.scheduler)(
        optimizer=optimizer, n_steps=cfg.trainer.n_steps
    )
    return optimizer, scheduler
