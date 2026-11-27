"""Parameter roles and the one optimizer builder main.py calls (line 2 of the plan)."""

import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from torch.distributed.tensor import DTensor, Shard
from torch.optim.adam import adam
from torch.profiler import record_function

from src.core.model import RoPEAttention, TransformerHead

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
    """Keller Jordan's quintic Newton-Schulz on a stack of matrices, in G's dtype.

    Roughly each matrix's orthogonal factor U V^T: singular values come out near 1, not at
    1 (github.com/KellerJordan/Muon). baddbmm folds the scalings into the matmuls.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    tall = G.size(-2) > G.size(-1)
    X = G.mT if tall else G
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)
    return X.mT if tall else X


def _local(t):
    """A DTensor's own shard (in place under no_grad), or t."""
    return t.to_local() if isinstance(t, DTensor) else t


def _gather_rows(local, p):
    """Whole matrices from every rank's rows of a stack shaped like p: one all-gather.

    Shard(0) splits rows as torch.chunk does, so short last shards are padded to equal size
    and the padding, all at the end, is cut off.
    """
    if not isinstance(p, DTensor):
        return local
    mesh = p.device_mesh
    world, rows = mesh.size(), p.shape[0]
    chunk = -(-rows // world)
    n, local_rows, cols = local.shape
    if local_rows < chunk:
        local = torch.nn.functional.pad(local, (0, 0, 0, chunk - local_rows))
    out = local.new_empty((world * n, chunk, cols))
    dist.all_gather_into_tensor(out, local.contiguous(), group=mesh.get_group())
    full = out.view(world, n, chunk, cols).transpose(0, 1)
    return full.reshape(n, world * chunk, cols)[:, :rows]


def _own_rows(full, p, dim=0):
    """This rank's rows (along dim) of whole matrices shaped like p, as Shard(0) splits it."""
    if not isinstance(p, DTensor):
        return full
    mesh = p.device_mesh
    chunk = -(-p.shape[0] // mesh.size())
    start = min(mesh.get_local_rank() * chunk, p.shape[0])
    return full.narrow(dim, start, _local(p).shape[0])


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
        """Elementwise work as foreach calls; per shape, one gather and one Newton-Schulz."""
        lr, beta = group["lr"], group["momentum"]
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        for p in params:
            if not self.state[p]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
        weights = [_local(p) for p in params]
        grads = [_local(p.grad) for p in params]
        bufs = [_local(self.state[p]["momentum_buffer"]) for p in params]
        with record_function("muon:momentum"):
            torch._foreach_lerp_(bufs, grads, 1 - beta)
            updates = torch._foreach_lerp(grads, bufs, beta)  # Nesterov
            torch._foreach_mul_(weights, 1 - lr * group["weight_decay"])
        # Newton-Schulz in bf16 on GPU; CPU bf16 matmuls are ~40x slower than fp32.
        dtype = torch.bfloat16 if grads[0].is_cuda else torch.float32
        by_shape = defaultdict(list)
        for i, p in enumerate(params):
            by_shape[p.shape].append(i)
        for shape, indices in by_shape.items():
            p = params[indices[0]]
            with record_function("muon:gather"):
                stack = torch.stack([updates[i] for i in indices]).to(dtype)
                full = _gather_rows(stack, p)
            with record_function("muon:newton_schulz"):
                orthogonal = newton_schulz(full, group["ns_steps"])
            with record_function("muon:update"):
                own = _own_rows(orthogonal, p, dim=1).to(weights[indices[0]].dtype)
                torch._foreach_add_(
                    [weights[i] for i in indices],
                    list(own.unbind(0)),
                    alpha=-lr * 0.2 * max(shape) ** 0.5,
                )


@torch.no_grad()
def qk_clip_(layers):
    """Kimi K2's QK-Clip (arXiv 2507.20534): scale W_q and W_k so no head's logits top tau.

    A query head whose max logit s > tau gets its logits scaled by gamma = tau / s. Its kv
    head's W_k rows take sqrt of the smallest gamma among the query heads sharing it, and
    W_q's rows the rest; with one query head per kv head both take sqrt(gamma), as in K2.
    Returns the largest logit before clipping, over all ranks.
    """
    max_logits = torch.stack([m.max_logits for m in layers])
    if dist.is_initialized():
        dist.all_reduce(max_logits, op=dist.ReduceOp.MAX)
    for m, logits in zip(layers, max_logits):
        gamma = m.qk_clip / logits.clamp(min=m.qk_clip)
        k_gamma = gamma.view(m.kv_heads, -1).amin(dim=1)
        q_scale = gamma / k_gamma.sqrt().repeat_interleave(m.q_heads // m.kv_heads)
        for weight, scale in (
            (m.q_proj.weight, q_scale),
            (m.k_proj.weight, k_gamma.sqrt()),
        ):
            rows = scale.repeat_interleave(m.dhead).to(weight.dtype)
            _local(weight).mul_(_own_rows(rows, weight)[:, None])
        m.max_logits.fill_(float("-inf"))
    return max_logits.max()


class _QKClipHook:
    """Optimizer post-step hook: QK-Clip every step; logs the window's max logit.

    The max stays on the GPU between logs: reading it every step would make the host wait
    for each optimizer step to finish.
    """

    def __init__(self, layers, metric_logger, log_every):
        self.layers = layers
        self.metric_logger = metric_logger
        self.log_every = log_every
        self.steps, self.window_max = 0, None

    def __call__(self, optimizer, args, kwargs):
        with record_function("qk_clip"):
            top = qk_clip_(self.layers)
        if self.window_max is not None:
            top = torch.maximum(self.window_max, top)
        self.window_max = top
        self.steps += 1
        if self.steps % self.log_every == 0:
            self.metric_logger.log("train/max_attention_logit", self.window_max.item())
            self.window_max = None


def add_qk_clip(optimizer, model, metric_logger, log_every=100):
    """QK-Clip after every optimizer step, if any attention layer has a threshold.

    Every log_every steps it logs train/max_attention_logit, the largest logit before
    clipping over those steps.
    """
    layers = [
        m
        for m in model.modules()
        if isinstance(m, RoPEAttention) and m.qk_clip is not None
    ]
    if layers:
        optimizer.register_step_post_hook(_QKClipHook(layers, metric_logger, log_every))


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
