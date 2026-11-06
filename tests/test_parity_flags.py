"""The parity audit's opt-in switches leave the default bit-identical.

`parity_bank_compute_dtype` and `parity_bank_eager` exist only on the
`parity_audit` branch, for the Phase 3 bisect of
documents/notes/ame_efficiency_parity_plan.md. The plan admits new code only as
opt-in flags whose default a test proves bit-identical: that is this file.
"""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from main import build_memory_parallel_controller
from tests.test_memory_parallel import VOCAB, build_model, micro_batches


def _cfg(**top):
    cfg = {"trainer": {"mem_parallel_size": 1, "distributed": None, "autocast_bf16": False}}
    cfg.update(top)
    return OmegaConf.create(cfg)


def _step(cfg):
    model = build_model(dtype=torch.float32, seed=0)
    banks = build_memory_parallel_controller(cfg, model)
    banks.materialize()
    banks.zero_grad()
    for ids in micro_batches(n=2):
        logits = model(ids)
        F.cross_entropy(logits.flatten(0, -2), ids.reshape(-1) % VOCAB).backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return banks, grads, [b.main_grad.clone() for b in banks.banks]


def test_unset_and_explicitly_off_are_the_same_path():
    plain_banks, plain_grads, plain_main = _step(_cfg())
    off_banks, off_grads, off_main = _step(
        _cfg(parity_bank_compute_dtype=None, parity_bank_eager=False)
    )
    assert plain_banks.compute_dtype == off_banks.compute_dtype
    for pair in plain_banks.fused_pairs + off_banks.fused_pairs:
        assert "forward" not in vars(pair)  # the class's forward, untouched
    assert plain_grads.keys() == off_grads.keys()
    for name in plain_grads:
        assert torch.equal(plain_grads[name], off_grads[name]), name
    for a, b in zip(plain_main, off_main):
        assert torch.equal(a, b)


def test_the_compute_dtype_override_takes_effect():
    banks = build_memory_parallel_controller(
        _cfg(parity_bank_compute_dtype="bfloat16"), build_model(dtype=torch.float32)
    )
    assert banks.compute_dtype == torch.bfloat16
    assert all(b.weight.dtype == torch.bfloat16 for b in banks.banks)
    assert all(b.master.dtype == torch.float32 for b in banks.banks)


def test_the_eager_switch_wraps_every_fused_pair_and_computes_the_same():
    _, grads, main = _step(_cfg())
    banks, eager_grads, eager_main = _step(_cfg(parity_bank_eager=True))
    assert banks.fused_pairs and all("forward" in vars(p) for p in banks.fused_pairs)
    for name in grads:
        assert torch.equal(grads[name], eager_grads[name]), name
    for a, b in zip(main, eager_main):
        assert torch.equal(a, b)
