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
# Ported to parity_audit (documents/notes/ame_efficiency_parity_plan.md, check 1.0)
# from ame_optimize at 922838c1: the CPU aot_eager case and the CUDA mp-1 case,
# which run unchanged against f9680709's bank path. The NCCL mp>=2 case is left
# out: it needs conftest.isolate_compile_caches, which exists only on
# ame_optimize, and both runs used mem_parallel_size 1.

import pytest
import torch
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
