"""The bank path under torch.compile: `main_grad` must equal eager.

Why this file exists (2026-11-05). `FusedBankPair.forward` carries no
`@torch.compiler.disable`, so under `compile_blocks_full` dynamo traces
`_FusedBankGather` -- its forward, its backward and the in-place `index_add_`
into `main_grad` -- with no graph break (`torch._dynamo.explain` on a toy block:
1 graph, 0 breaks). Every AME run since D2 (job 32666) takes that path, at
every GPU count, and nothing checked its gradient under compile:
`test_fusing_the_qk_pair_changes_nothing` runs eager.

What was found on CPU, torch 2.7.1:

- `aot_eager` is exact. AOTAutograd keeps the mutation: it functionalizes
  `index_add_` into `index_put(..., accumulate=True)` and copies the result back
  into `main_grad`.
- **inductor's C++ backend is wrong.** It fuses the k-bank scatter into a tiled,
  vectorised kernel whose target index uses the wrong tile variable
  (`x0 + 2*x0_inner + 64*id` inside a loop over `x1_inner`), so each k-bank row
  gradient is spread over rows `id` and `id + 1`. The q bank is exact. Max error
  7.3e-3 on the toy model below -- the number D1 (2026-11-04) measured and
  attributed to "the side effect does not survive AOTAutograd". It is a CPU
  codegen bug, not an AOTAutograd one.

Production runs on CUDA, where inductor generates Triton instead, so the CUDA
tests below are the gate that matters. The CPU inductor case is not tested: it
is a known torch bug, slow to compile, and no run takes that path.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.core.memory_parallel import MemoryParallelController
from tests.test_memory_parallel import VOCAB, build_model, micro_batches

# Relative to the largest reference entry. fp32 reassociation in fused kernels
# sits around 1e-6; the CPU bug above is O(1) relative on the rows it hits.
RTOL = 1e-4


def _one_backward(model, banks, ids, backend):
    """One forward/backward, returning every bank's main_grad and every
    backbone gradient, flattened per tensor."""
    import torch._dynamo

    torch._dynamo.reset()
    if backend is not None:
        for block in model.blocks:
            block.compile(backend=backend)
    banks.materialize()
    logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    loss.backward()
    main = [bank.main_grad.detach().clone() for bank in banks.banks]
    backbone = [
        p.grad.detach().clone() for p in model.parameters() if p.grad is not None
    ]
    return main, backbone, loss.detach()


def _relative_error(reference, other):
    worst = 0.0
    for a, b in zip(reference, other):
        scale = a.abs().max().clamp_min(1e-30)
        worst = max(worst, ((a - b).abs().max() / scale).item())
    return worst


def _compare(device, backend):
    ids = micro_batches()[0].to(device)
    results = []
    for which in (None, backend):
        model = build_model(dtype=torch.float32, seed=0).to(device)
        banks = MemoryParallelController(
            model, mem_parallel_size=1, compute_dtype=torch.float32, fuse_qk=True
        )
        assert len(banks.fused_pairs) == len(model.blocks)
        results.append(_one_backward(model, banks, ids, which))
    (main_e, back_e, loss_e), (main_c, back_c, loss_c) = results
    assert max(g.abs().max().item() for g in main_e) > 0, "vacuous: no bank gradient"
    return _relative_error(main_e, main_c), _relative_error(back_e, back_c)


def test_aot_autograd_keeps_the_main_grad_mutation():
    """The traced graph itself is right; any error comes from codegen."""
    bank_err, backbone_err = _compare("cpu", "aot_eager")
    assert bank_err == 0.0, f"aot_eager main_grad differs: {bank_err:.3e}"
    assert backbone_err < RTOL


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_fused_pair_main_grad_matches_eager_on_cuda():
    bank_err, backbone_err = _compare("cuda", "inductor")
    assert bank_err < RTOL, (
        f"compiled main_grad differs from eager by {bank_err:.3e} (relative) -- "
        "the compiled bank path is computing a wrong bank gradient"
    )
    assert backbone_err < RTOL, f"backbone gradients differ by {backbone_err:.3e}"


def _nccl_worker(rank, world_size, port, out_path):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device("cuda", rank)
        ids = micro_batches(rank)[0].to(device)

        per_backend = {}
        for which in (None, "inductor"):
            model = build_model(dtype=torch.float32, seed=0).to(device)
            banks = MemoryParallelController(
                model,
                mem_parallel_size=world_size,
                compute_dtype=torch.float32,
                fuse_qk=True,
            )
            main, backbone, _ = _one_backward(model, banks, ids, which)
            full = []
            for grad in main:
                slices = [torch.empty_like(grad) for _ in range(world_size)]
                dist.all_gather(slices, grad.contiguous(), group=banks.mp_group)
                full.append(torch.cat(slices, dim=1).cpu())
            per_backend[which] = (full, [g.cpu() for g in backbone])

        (main_e, back_e), (main_c, back_c) = per_backend[None], per_backend["inductor"]
        bank_err = torch.tensor(_relative_error(main_e, main_c), device=device)
        backbone_err = torch.tensor(_relative_error(back_e, back_c), device=device)
        dist.all_reduce(bank_err, op=dist.ReduceOp.MAX)
        dist.all_reduce(backbone_err, op=dist.ReduceOp.MAX)
        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "bank_err": bank_err.item(),
                    "backbone_err": backbone_err.item(),
                    "bank_moved": max(g.abs().max().item() for g in main_e),
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs at least 2 CUDA devices",
)
def test_compiled_fused_pair_main_grad_matches_eager_over_nccl(tmp_path):
    """The production shape of the path: all-gather of ids, all-to-all, permute,
    traced by dynamo, at mem_parallel_size = world size."""
    from conftest import spawn_on_free_port

    world_size = min(4, torch.cuda.device_count())
    out = str(tmp_path / "nccl_compile.pt")
    spawn_on_free_port(_nccl_worker, lambda port: (world_size, port, out), world_size)
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])
    assert payload["bank_moved"] > 0, "vacuous: no bank gradient"
    assert payload["bank_err"] < RTOL, (
        f"compiled main_grad differs from eager by {payload['bank_err']:.3e} "
        f"(relative) at mem_parallel_size {world_size}"
    )
    assert payload["backbone_err"] < RTOL
