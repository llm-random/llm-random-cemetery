"""Line 7 schedules: WSD's final_lr_fraction and batch size warmup."""

import os
import warnings

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.core.schedulers import BatchWarmupLRScale, WSDScheduler
from src.core.trainer import Trainer
from src.definitions import check_point

ROOT = os.path.dirname(os.path.dirname(__file__))
FEATURE_DIR = os.path.join(ROOT, "configs", "jm", "feature")
# 8 steps at batch 2 and 4 at batch 4 (two loader batches of 8 each), then batch 8.
WARMUP = {
    "batch_sizes": [2, 4],
    "steps": [8, 4],
    "final_batch_size": 8,
    "scale_lr": True,
    "rescale_adam_v": False,
}


def _compose(name, overrides=()):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(version_base=None, config_dir=FEATURE_DIR):
        return compose(config_name=name, overrides=list(overrides))


def test_wsd_decays_each_group_to_its_fraction_of_peak():
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [{"params": [params[0]], "lr": 1.0}, {"params": [params[1]], "lr": 0.5}]
    )
    scheduler = WSDScheduler(
        optimizer, n_steps=10, warmup_steps=2, decay_steps=4, final_lr_fraction=0.1
    )
    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr())
    # Decay starts after step 6; halfway through it the factor is 1 - 0.9 / 2.
    assert lrs[7] == pytest.approx([0.55, 0.275])
    assert lrs[9] == pytest.approx([0.1, 0.05])


def test_wsd_final_lr_fraction_outside_0_1_fails_check_point():
    cfg = _compose(
        "tiny_local_wsd_final_lr", ["trainer.scheduler.final_lr_fraction=1.5"]
    )
    with pytest.raises(ValueError, match="final_lr_fraction must be in"):
        check_point(cfg)


class _WarmupStub:
    """The Trainer's batch size warmup methods on the smallest object they need."""

    _warmup_split = Trainer._warmup_split
    _warmup_batches = Trainer._warmup_batches

    def __init__(self, rows, n_batches, start_step=0):
        self.batch_size_warmup = OmegaConf.create(WARMUP)
        self.start_step = start_step
        # Row i holds sample i, so the stream order is visible in the values.
        self.train_dataloader = [
            torch.arange(i * rows, (i + 1) * rows).unsqueeze(1)
            for i in range(n_batches)
        ]


@pytest.mark.parametrize("world_size, rows", [(1, 8), (2, 4)])
def test_warmup_slices_the_loader_stream_in_order(world_size, rows):
    stub = _WarmupStub(rows=rows, n_batches=6)
    pieces = list(stub._warmup_batches())
    per_device = [b // world_size for b in [2] * 8 + [4] * 4 + [8] * 2]
    assert [len(p) for p in pieces] == per_device
    assert torch.equal(torch.cat(pieces), torch.cat(stub.train_dataloader))


@pytest.mark.parametrize("start_step", [1, 5, 8, 11, 12, 13])
def test_resumed_warmup_yields_the_uninterrupted_steps(start_step):
    full = list(_WarmupStub(rows=8, n_batches=6)._warmup_batches())
    resumed = list(
        _WarmupStub(rows=8, n_batches=6, start_step=start_step)._warmup_batches()
    )
    assert len(resumed) == len(full) - start_step
    assert all(torch.equal(a, b) for a, b in zip(full[start_step:], resumed))


def _scaled_wsd():
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.SGD(
        [{"params": [params[0]], "lr": 1.0}, {"params": [params[1]], "lr": 0.5}]
    )
    wsd = WSDScheduler(
        optimizer, n_steps=20, warmup_steps=4, decay_steps=4, final_lr_fraction=0.0
    )
    return optimizer, BatchWarmupLRScale(wsd, OmegaConf.create(WARMUP))


def _lrs_in_effect(optimizer, scheduler, n_steps):
    lrs = []
    for _ in range(n_steps):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return lrs


def test_lr_scale_is_sqrt_of_stage_over_final_batch():
    plain_opt = torch.optim.SGD(
        [
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1.0},
            {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 0.5},
        ]
    )
    plain = WSDScheduler(
        plain_opt, n_steps=20, warmup_steps=4, decay_steps=4, final_lr_fraction=0.0
    )
    optimizer, scaled = _scaled_wsd()
    # Batch 2 for steps 0-7, 4 for 8-11, then 8.
    scales = [0.5] * 8 + [2**-0.5] * 4 + [1.0] * 8
    expected = [
        [lr * s for lr in lrs]
        for lrs, s in zip(_lrs_in_effect(plain_opt, plain, 20), scales)
    ]
    np.testing.assert_allclose(_lrs_in_effect(optimizer, scaled, 20), expected)


@pytest.mark.parametrize("stop", [3, 8, 10, 12, 15])
def test_resumed_lr_scale_gives_the_uninterrupted_lrs(stop):
    optimizer, scheduler = _scaled_wsd()
    full = _lrs_in_effect(optimizer, scheduler, 20)

    optimizer, scheduler = _scaled_wsd()
    _lrs_in_effect(optimizer, scheduler, stop)
    saved_optim, saved_sched = optimizer.state_dict(), scheduler.state_dict()
    optimizer, scheduler = _scaled_wsd()
    # checkpointing.py's order: optimizer first, then scheduler.
    optimizer.load_state_dict(saved_optim)
    scheduler.load_state_dict(saved_sched)
    np.testing.assert_allclose(
        _lrs_in_effect(optimizer, scheduler, 20 - stop), full[stop:]
    )


def _adam_wsd():
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    wsd = WSDScheduler(
        optimizer, n_steps=20, warmup_steps=4, decay_steps=4, final_lr_fraction=0.0
    )
    warmup = OmegaConf.create({**WARMUP, "rescale_adam_v": True})
    return param, optimizer, BatchWarmupLRScale(wsd, warmup)


def _adam_steps(param, optimizer, scheduler, start, stop):
    """Steps start..stop-1 on fixed per-step gradients; lr/sqrt(v) before and after each scheduler step."""
    before, after = [], []
    for step in range(start, stop):
        param.grad = torch.randn(3, generator=torch.Generator().manual_seed(step))
        optimizer.step()
        v = optimizer.state[param]["exp_avg_sq"]
        before.append(optimizer.param_groups[0]["lr"] / v.sqrt())
        scheduler.step()
        after.append(optimizer.param_groups[0]["lr"] / v.sqrt())
    return before, after


def test_rescale_adam_v_keeps_lr_over_sqrt_v_across_switches():
    param, optimizer, scheduler = _adam_wsd()
    before, after = _adam_steps(param, optimizer, scheduler, 0, 20)
    ratios = [float((a / b).mean()) for b, a in zip(before, after)]
    # Steps 7 and 11 leave a stage: without the v rescale lr/sqrt(v) would jump sqrt(2) there.
    # Elsewhere the ratio is WSD's own step-to-step LR change.
    lrs = _lrs_in_effect(*_scaled_wsd(), 21)
    wsd_ratios = [lrs[s + 1][0] / lrs[s][0] for s in range(20)]
    wsd_ratios[7] /= 2**0.5
    wsd_ratios[11] /= 2**0.5
    np.testing.assert_allclose(ratios, wsd_ratios, rtol=1e-6)


@pytest.mark.parametrize("stop", [7, 8, 9, 12])
def test_resumed_rescale_adam_v_gives_the_uninterrupted_run(stop):
    param, optimizer, scheduler = _adam_wsd()
    _adam_steps(param, optimizer, scheduler, 0, 20)
    full = param.detach().clone()

    param, optimizer, scheduler = _adam_wsd()
    _adam_steps(param, optimizer, scheduler, 0, stop)
    saved = param.detach().clone(), optimizer.state_dict(), scheduler.state_dict()
    param, optimizer, scheduler = _adam_wsd()
    with torch.no_grad():
        param.copy_(saved[0])
    optimizer.load_state_dict(saved[1])
    scheduler.load_state_dict(saved[2])
    _adam_steps(param, optimizer, scheduler, stop, 20)
    torch.testing.assert_close(param.detach(), full)


@pytest.mark.parametrize(
    "name",
    [
        "tiny_local_batch_warmup",
        "tiny_local_batch_warmup_lr_scale",
        "tiny_local_batch_warmup_rescale_adam_v",
    ],
)
def test_shipped_warmup_config_passes_check_point(name):
    check_point(_compose(name))


@pytest.mark.parametrize(
    "override, message",
    [
        ("common.batch_size=16", "must equal common.batch_size"),
        ("trainer.batch_size_warmup.batch_sizes=[3,4]", "must divide"),
        ("trainer.batch_size_warmup.batch_sizes=[2,8]", "must divide"),
        ("trainer.batch_size_warmup.steps=[10,20]", "must end on a loader batch"),
        ("trainer.batch_size_warmup.steps=[20]", "one steps entry per"),
        ("trainer.gradient_accumulation_steps=3", "gradient_accumulation_steps=3"),
        ("trainer.batch_size_warmup.rescale_adam_v=true", "needs scale_lr"),
    ],
)
def test_misfitting_warmup_fails_check_point(override, message):
    cfg = _compose("tiny_local_batch_warmup", [override])
    with pytest.raises(ValueError, match=message):
        check_point(cfg)
