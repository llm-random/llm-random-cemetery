"""Opt-in peak-memory logging (rebuttal E5): Trainer._log_peak_memory.

The CUDA counters are faked so the test runs on CPU; dist is not initialised, so the
reduction is the identity.
"""

import types

import pytest
import torch

from src.core.trainer import Trainer

GIB = 2**30


class _Logger:
    def __init__(self):
        self.logged = {}

    def log(self, name, value):
        self.logged[name] = value


def _fake_trainer(step, n_steps=30):
    return types.SimpleNamespace(
        step=step, n_steps=n_steps, device=torch.device("cpu"), metric_logger=_Logger()
    )


@pytest.fixture
def fake_cuda(monkeypatch):
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 5 * GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 6 * GIB)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: resets.append(1))
    return resets


def test_off_by_default(monkeypatch, fake_cuda):
    monkeypatch.delenv("NANO_PEAK_MEMORY_EVERY", raising=False)
    t = _fake_trainer(step=0)
    Trainer._log_peak_memory(t)
    assert t.metric_logger.logged == {}
    assert fake_cuda == []


def test_step0_logs_init_peak_and_resets(monkeypatch, fake_cuda):
    monkeypatch.setenv("NANO_PEAK_MEMORY_EVERY", "10")
    t = _fake_trainer(step=0)
    Trainer._log_peak_memory(t)
    logged = t.metric_logger.logged
    assert logged["memory/init_and_step0/peak_allocated_gib_max_rank"] == 5.0
    assert logged["memory/init_and_step0/peak_reserved_gib_max_rank"] == 6.0
    assert logged["memory/init_and_step0/peak_host_rss_gib_max_rank"] > 0
    assert fake_cuda == [1]


def test_cadence_and_last_step(monkeypatch, fake_cuda):
    monkeypatch.setenv("NANO_PEAK_MEMORY_EVERY", "10")
    for step, expect in [(5, False), (10, True), (29, True)]:
        t = _fake_trainer(step=step, n_steps=30)
        Trainer._log_peak_memory(t)
        assert bool(t.metric_logger.logged) == expect, step
        if expect:
            assert "memory/training/peak_allocated_gib_max_rank" in t.metric_logger.logged
    assert fake_cuda == []  # only step 0 resets the counters
