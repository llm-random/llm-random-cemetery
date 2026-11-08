"""The init defaults of branch ame_initialization (documents/notes/ame_init_exp.md).

Every model config builds from configs/_model/h_balanced/base.yaml, so these
tests compose tiny dense and AME models from it and check:
  - the token embedding, head and AME banks: Xavier normal, sqrt(2 / (V + d)),
    not cut;
  - the attention and FF Linears: sqrt(0.1 / fan_in), cut at +-2 std;
  - weight decay 0 on the embedding and the banks, 0.1 on the head and Linears;
  - the memory-parallel fp32 masters hold the banks' Xavier draw.
"""

import math
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)
from src.core.ame import AMEAttention
from src.core.memory_parallel import MemoryParallelController
from src.core.model import Linear, parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Large enough that each std estimate is good to about 0.1%, small enough to
# build quickly on CPU. h4: dmodel 256, dff 1024, 4 heads of 64.
VOCAB = 4096
DMODEL = 256
XAVIER_STD = math.sqrt(2 / (VOCAB + DMODEL))  # 0.0214
LINEAR_SCALE = math.sqrt(0.1)


def _cut_std_factor():
    """The std of N(0, 1) cut at +-2: 0.8796."""
    density = math.exp(-2.0) / math.sqrt(2 * math.pi)  # phi(2)
    mass = math.erf(2 / math.sqrt(2))  # P(|z| < 2)
    return math.sqrt(1 - 4 * density / mass)


CUT_STD_FACTOR = _cut_std_factor()


def _h4(attention_type, overrides=()):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=[
                "common.sequence_length=16",
                f"common.vocab_size={VOCAB}",
                f"common.attention.type={attention_type}",
                *overrides,
            ],
        )
    torch.manual_seed(0)
    return instantiate(cfg.model, _convert_="all")


@pytest.fixture(scope="module", params=["gqa", "ame"])
def model(request):
    return _h4(request.param)


def _banks(model):
    for block in model.encoder.blocks:
        layer = block.attention_layer.layer
        if isinstance(layer, AMEAttention):
            yield layer.q_bank.weight
            yield layer.k_bank.weight


def _assert_xavier(weight):
    fan_out, fan_in = weight.shape
    expected = math.sqrt(2 / (fan_in + fan_out))
    assert weight.std().item() == pytest.approx(expected, rel=0.01)
    assert weight.mean().abs().item() < 0.01 * expected
    # Not cut: a normal sample this size reaches past 4 std.
    assert weight.abs().max().item() > 4 * expected


def test_the_embedding_and_head_are_xavier_normal(model):
    assert model.embedding.weight.shape == (VOCAB, DMODEL)
    _assert_xavier(model.embedding.weight)
    assert model.head.linear.weight.shape == (VOCAB, DMODEL)
    _assert_xavier(model.head.linear.weight)
    assert model.embedding.weight.std().item() == pytest.approx(XAVIER_STD, rel=0.01)


def test_the_ame_banks_are_xavier_normal():
    model = _h4("ame")
    banks = list(_banks(model))
    assert len(banks) == 2 * 4
    for bank in banks:
        assert bank.shape == (VOCAB, DMODEL)
        _assert_xavier(bank)


def test_every_attention_and_ff_linear_is_the_cut_fan_in_normal(model):
    linears = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Linear) and not name.startswith("head.")
    ]
    # 4 blocks x (4 attention projections for gqa, v/o for ame, + 3 FF).
    assert len(linears) >= 4 * 5
    for name, module in linears:
        sigma = LINEAR_SCALE / math.sqrt(module.in_features)
        weight = module.weight
        assert weight.std().item() == pytest.approx(
            CUT_STD_FACTOR * sigma, rel=0.03
        ), name
        assert weight.abs().max().item() <= 2 * sigma, name


def test_weight_decay_spares_the_embedding_and_banks_only(model):
    decayed, undecayed = parameter_groups(model, 0.1)
    assert decayed["weight_decay"] == 0.1 and undecayed["weight_decay"] == 0.0
    decayed = {id(p) for p in decayed["params"]}
    undecayed = {id(p) for p in undecayed["params"]}

    assert id(model.embedding.weight) in undecayed
    for bank in _banks(model):
        assert id(bank) in undecayed
    assert id(model.head.linear.weight) in decayed
    for module in model.modules():
        if isinstance(module, Linear):
            assert id(module.weight) in decayed
    assert len(decayed) + len(undecayed) == len(list(model.parameters()))


def test_an_explicit_bank_init_std_still_draws_n_0_std():
    """The stage 1-2 meaning of a number, kept for configs that set one."""
    model = _h4("ame", ["+common.attention.ame={bank_init_std: 1.0}"])
    for bank in _banks(model):
        assert bank.std().item() == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_bank_init_std_is_refused(bad):
    with pytest.raises(ValueError, match="bank_init_std"):
        AMEAttention(256, 4, 2, 16, vocab_size=96, bank_init_std=bad)


def test_the_memory_parallel_masters_hold_the_xavier_draw():
    """Tier 2 (fp32 master, bf16 compute copy): the master is copied from the
    module's draw, so it must be that draw, and the compute copy its cast."""
    model = _h4("ame")
    drawn = [bank.detach().clone() for bank in _banks(model)]
    controller = MemoryParallelController(model, 1, compute_dtype=torch.bfloat16)
    assert len(controller.banks) == len(drawn)
    for bank, before in zip(controller.banks, drawn):
        assert bank.master.dtype == torch.float32
        assert torch.equal(bank.master, before)
        assert torch.equal(bank.weight, before.to(torch.bfloat16))
        _assert_xavier(bank.master)


def test_the_composed_defaults_read_as_the_plan_says():
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(config_name="h12", overrides=["common.sequence_length=16"])
    model = cfg.model
    assert model.embedding.init_fn._target_ == "torch.nn.init.xavier_normal_"
    assert (
        model.head.linear_fn.partial_init_fn._target_
        == "src.core.model.xavier_normal_weight_init"
    )
    block = model.encoder.block_fn
    assert block.attention_fn.init_scale == pytest.approx(LINEAR_SCALE)
    assert block.ff_layer_fn.ff_pre_act_fn.partial_init_fn.scale == pytest.approx(
        LINEAR_SCALE
    )
    assert block.ff_layer_fn.ff_post_act_fn.partial_init_fn.scale == pytest.approx(
        LINEAR_SCALE
    )
