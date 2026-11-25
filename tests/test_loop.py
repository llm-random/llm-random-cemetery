"""The looped-depth switch (common.loop): which blocks run how often, and its guards."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401
from src.core.model import ConcatInjection

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _h4(*overrides):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4", overrides=["common.sequence_length=16", *overrides]
        )
    return instantiate(cfg.model, _convert_="all")


def _block_calls(model):
    calls = [0] * len(model.encoder.blocks)
    for i, block in enumerate(model.encoder.blocks):
        block.register_forward_hook(lambda *_, i=i: calls.__setitem__(i, calls[i] + 1))
    model(torch.randint(0, 100, (2, 16)))
    return calls


def test_off_runs_every_block_once():
    assert _block_calls(_h4()) == [1, 1, 1, 1]


def test_the_core_runs_n_loops_times_between_prelude_and_coda():
    model = _h4(
        "common.loop.n_loops=3", "common.loop.n_prelude=1", "common.loop.n_coda=1"
    )
    assert _block_calls(model) == [1, 3, 3, 1]


def test_concat_injection_starts_as_the_identity_on_the_state():
    torch.manual_seed(0)
    x, prelude_out = torch.randn(2, 16, 64), torch.randn(2, 16, 64)
    assert torch.equal(ConcatInjection(64)(x, prelude_out), x)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["common.loop.n_loops=0"], "n_loops must be >= 1"),
        (["common.loop.n_prelude=2", "common.loop.n_coda=2"], "non-empty core"),
        (["common.loop.n_prelude=-1"], "non-empty core"),
        (["common.loop.injection=add"], "none or concat"),
    ],
)
def test_a_bad_loop_config_is_a_build_error(overrides, message):
    with pytest.raises(Exception, match=message):
        _h4(*overrides)
