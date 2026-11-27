"""AME's attention gates: A, the read gate, and B, the output gate.

documents/notes/ame_attention_gate.md specifies both. A multiplies the heads' read
`r_i` by `phi(W_A h_i)` just before the output projection `W_O`, per channel
(`elementwise`) or per head (`headwise`); B multiplies the attention's output by
`phi(W_B h_i)` just before the residual add. `phi` is sigmoid (the default) or
SiLU, `h_i` the layer's normalized input. Each gate works alone, and both together.

What these tests guard, beyond "the gated layer runs":

- **no regression:** with both gates off the model is the pre-gate model bit for
  bit -- its parameters, its state dict and its forward;
- **the score table:** neither gate enters `q.k`, so the scores stay a function
  of token ids and distance alone (feedback: the AME score-table invariant);
- **init isolation:** a gate draws from a generator of its own and moves no other
  parameter (src/core/init_rng.py);
- **the training graph:** the eval-time gate readout must not reach it, so the
  gated layer still compiles to one graph in train mode.

Elsewhere: the options through build_attention and into a mixed stack in
tests/test_attention_registry.py; init isolation in tests/test_init_rng.py
(`SWITCHES`); a gated stack through main.py in tests/test_training_smoke.py (slow).
"""

import hashlib
import math
import os

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)
from src.core.ame import GATE_ACTIVATIONS, OUTPUT_GATES, READ_GATES, AMEAttention
from src.core.mfu import model_flops_per_token
from src.core.model import parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOCAB = 64
DIMS = dict(dmodel=32, q_heads=4, kv_heads=2, seq_len=10)

# Every combination with at least one gate on, under both activations.
GATED = [
    (read_gate, output_gate, activation)
    for read_gate in READ_GATES
    for output_gate in OUTPUT_GATES
    for activation in GATE_ACTIVATIONS
    if (read_gate, output_gate) != ("none", "none")
]


def _ids(combo):
    read_gate, output_gate, activation = combo
    return f"A_{read_gate}-B_{output_gate}-{activation}"


def _layer(seed=0, **options):
    torch.manual_seed(seed)
    options.setdefault("backend", "reference")
    return AMEAttention(**DIMS, vocab_size=VOCAB, **options)


def _gated(seed, read_gate, output_gate, activation, **options):
    return _layer(
        seed,
        read_gate=read_gate,
        output_gate=output_gate,
        gate_activation=activation,
        **options,
    )


def _batch(batch=2, seed=1, dmodel=DIMS["dmodel"]):
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, VOCAB, (batch, DIMS["seq_len"]), generator=generator)
    x = torch.randn(batch, DIMS["seq_len"], dmodel, generator=generator)
    return x, token_ids


def _perturb_sink(layer, seed=9):
    """The sink starts at zero; move it, so the read the gates act on includes
    a sink term that matters."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer.k_sink.copy_(torch.randn(layer.k_sink.shape, generator=generator))
        layer.v_sink.copy_(torch.randn(layer.v_sink.shape, generator=generator))
    return layer


def _gate_weights(module):
    return {name: p for name, p in module.named_parameters() if "gate_proj" in name}


def _h4_model(overrides=(), seed=1234, device="cpu", attention="ame"):
    """An h4 model built as main.py builds one (torch.manual_seed, then
    instantiate), at test_init_rng.py's size: sequence 16, vocab 96."""
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=[
                "common.sequence_length=16",
                "common.vocab_size=96",
                f"common.attention.type={attention}",
                *overrides,
            ],
        )
    torch.manual_seed(seed)
    with torch.device(device):
        return instantiate(cfg.model, _convert_="all")


def _ame_layers(model):
    return [m for m in model.modules() if isinstance(m, AMEAttention)]


# ---------------------------------------------------------------------------
# The gated layer, transcribed from the doc's equations. Naive on purpose, like
# tests/test_ame.py's _spec_forward: it must not share code with src/core/ame.py.


def _spec_forward(layer, x, token_ids):
    batch, n = token_ids.shape
    heads, d = layer.q_heads, layer.dhead

    q = layer.q_bank.weight[token_ids].view(batch, n, heads, d)
    k = layer.k_bank.weight[token_ids].view(batch, n, heads, d)
    v = (x @ layer.v_proj.weight.T).view(batch, n, heads, d)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)

    def shift1(z):
        shifted = torch.zeros_like(z)
        shifted[:, 1:] = z[:, :-1]
        return shifted

    alpha, beta = layer.alpha, layer.beta
    k = alpha[:, 0].view(1, 1, -1, 1) * k + alpha[:, 1].view(1, 1, -1, 1) * shift1(k)
    v = beta[:, 0].view(1, 1, -1, 1) * v + beta[:, 1].view(1, 1, -1, 1) * shift1(v)
    q = q * layer.tau.view(1, 1, -1, 1)

    i = torch.arange(n).view(-1, 1)
    j = torch.arange(n).view(1, -1)
    s = torch.einsum("bihd,bjhd->bhij", q, k)
    s = s - layer.slopes.view(1, -1, 1, 1) * (i - j).float()
    s = s.masked_fill(j > i, float("-inf"))
    if layer.use_sink:
        # the sink: kv column 0, no distance, visible from every row
        s_sink = torch.einsum("bihd,hd->bhi", q, layer.k_sink).unsqueeze(-1)
        s = torch.cat([s_sink, s], dim=-1)
        v_sink = layer.v_sink.view(1, 1, heads, d).expand(batch, 1, heads, d)
        v = torch.cat([v_sink, v], dim=1)
    a = s.softmax(dim=-1)
    r = torch.einsum("bhij,bjhd->bihd", a, v)  # r_i^h, head h's read

    phi = {"sigmoid": torch.sigmoid, "silu": F.silu}.get(layer.gate_activation)
    w_o = layer.o_proj.weight  # head h owns columns h*d .. (h+1)*d - 1
    if layer.read_gate == "headwise":
        # o_i = sum_h g_ih * W_O^h r_i^h
        g = phi(x @ layer.read_gate_proj.weight.T)  # [B, n, H]
        o = sum(
            g[..., h, None] * (r[:, :, h] @ w_o[:, h * d : (h + 1) * d].T)
            for h in range(heads)
        )
    else:
        if layer.read_gate == "elementwise":
            # o_i = W_O (g_i (.) r_i)
            g = phi(x @ layer.read_gate_proj.weight.T)  # [B, n, H*d]
            r = g.view(batch, n, heads, d) * r
        o = r.reshape(batch, n, heads * d) @ w_o.T
    if layer.output_gate == "elementwise":
        # x_i' = x_i + g_i^B (.) o_i: the layer returns g^B (.) o
        o = phi(x @ layer.output_gate_proj.weight.T) * o
    return o


# ---------------------------------------------------------------------------
# No regression: with both gates off, AME is the model it was before the gates.

# The h4 AME model at seed 1234 (test_init_rng.py's build: sequence 16, vocab 96)
# on the tree before any gate code: ame_gate at 06ddbbc0, 2026-11-27, torch 2.7.1.
# The state dict is a pure function of the seed, so its digest holds on any x86
# machine. The forward's last bits depend on the CPU kernels torch dispatches to:
# its digest is compared where the CPU capability matches the recording's, and
# elsewhere its sums to 1e-6.
BASELINE_STATE_DICT_SHA256 = (
    "ab0e1527457bfada83f3bd19adc35f34f71f99a92af2b4be66be7fc67ac81d75"
)
BASELINE_STATE_DICT_ENTRIES = 63
BASELINE_FORWARD_SHA256 = (
    "bbd759203828444a65494a61bbec642e156938b402dae6fc27de1e76512b5325"
)
BASELINE_FORWARD_CPU = "AVX2"
BASELINE_FORWARD_SUM = 81.33464276691666
BASELINE_FORWARD_ABS_SUM = 2122.5115708677913


def _state_dict_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        t = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(t.dtype).encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def _fixed_ids():
    return (torch.arange(2 * 16).reshape(2, 16) * 37 + 11) % 96


def test_without_gates_the_h4_model_is_the_pre_gate_model_bit_for_bit():
    model = _h4_model()
    assert len(model.state_dict()) == BASELINE_STATE_DICT_ENTRIES
    assert _state_dict_sha256(model) == BASELINE_STATE_DICT_SHA256, (
        "the ungated h4 AME model no longer initializes as it did before the "
        "gates: a parameter moved, or the state dict changed its keys or order"
    )
    with torch.no_grad():
        logits = model(_fixed_ids())
    if torch.backends.cpu.get_cpu_capability() == BASELINE_FORWARD_CPU:
        forward = logits.contiguous().view(torch.uint8).numpy().tobytes()
        assert (
            hashlib.sha256(forward).hexdigest() == BASELINE_FORWARD_SHA256
        ), "the ungated forward is no longer the pre-gate forward bit for bit"
    else:
        assert logits.double().sum().item() == pytest.approx(
            BASELINE_FORWARD_SUM, rel=1e-6
        )
        assert logits.double().abs().sum().item() == pytest.approx(
            BASELINE_FORWARD_ABS_SUM, rel=1e-6
        )


def test_the_default_is_both_gates_none_and_adds_nothing():
    default = _perturb_sink(_layer(seed=3))
    explicit = _perturb_sink(_layer(seed=3, read_gate="none", output_gate="none"))
    assert default.read_gate == explicit.read_gate == "none"
    assert default.output_gate == explicit.output_gate == "none"
    assert default.gate_activation is explicit.gate_activation is None
    assert list(default.state_dict()) == list(explicit.state_dict())
    for name, tensor in default.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert _gate_weights(default) == {}
    x, token_ids = _batch()
    assert torch.equal(default(x, token_ids), explicit(x, token_ids))


# ---------------------------------------------------------------------------
# Options and validation.


@pytest.mark.parametrize(
    "options, valid_set",
    [
        ({"read_gate": "elementwize"}, "none, elementwise, headwise"),
        ({"read_gate": "None"}, "none, elementwise, headwise"),
        ({"output_gate": "headwise"}, "none, elementwise"),
        ({"output_gate": "channelwise"}, "none, elementwise"),
        ({"read_gate": "headwise", "gate_activation": "tanh"}, "sigmoid, silu"),
        ({"output_gate": "elementwise", "gate_activation": "none"}, "sigmoid, silu"),
        ({"gate_activation": "sigmoid"}, "read_gate (elementwise, headwise)"),
        ({"gate_activation": "silu"}, "output_gate (elementwise)"),
    ],
    ids=[
        "read_gate_typo",
        "read_gate_yaml_None",
        "output_gate_headwise",
        "output_gate_unknown",
        "activation_unknown",
        "activation_string_none",
        "activation_without_a_gate",
        "silu_without_a_gate",
    ],
)
def test_a_bad_gate_option_is_rejected_with_the_valid_set(options, valid_set):
    with pytest.raises(ValueError) as caught:
        _layer(**options)
    assert valid_set in str(caught.value)


def test_output_gate_headwise_says_why_it_does_not_exist():
    """B multiplies the output after W_O has summed the heads: there is no head
    left to gate."""
    with pytest.raises(ValueError, match="no heads"):
        _layer(output_gate="headwise")


@pytest.mark.parametrize(
    "options, activation",
    [
        ({"read_gate": "elementwise"}, "sigmoid"),
        ({"output_gate": "elementwise"}, "sigmoid"),
        ({"read_gate": "headwise", "gate_activation": "silu"}, "silu"),
        ({"read_gate": "headwise", "gate_activation": "sigmoid"}, "sigmoid"),
        ({}, None),
    ],
)
def test_an_unset_activation_is_sigmoid_once_a_gate_is_on(options, activation):
    assert _layer(**options).gate_activation == activation


def test_extra_repr_shows_the_three_options():
    text = repr(_layer(read_gate="headwise", gate_activation="silu"))
    assert "read_gate=headwise, output_gate=none, gate_activation=silu" in text


# ---------------------------------------------------------------------------
# Shapes, no bias, and the init: SwiGLU's gate rule, a truncated normal of std
# 1/sqrt(dmodel) cut at 2 std, whose actual std is 0.8796/sqrt(dmodel).

TRUNCATED_STD = 0.8796  # std of N(0, 1) truncated to [-2, 2]
WIDE = dict(dmodel=256, q_heads=4, kv_heads=2, seq_len=10)  # d_head 64


@pytest.mark.parametrize(
    "options, name, rows, tolerance",
    [
        ({"read_gate": "elementwise"}, "read_gate_proj", 256, 0.03),
        ({"read_gate": "headwise"}, "read_gate_proj", 4, 0.08),
        ({"output_gate": "elementwise"}, "output_gate_proj", 256, 0.03),
    ],
    ids=["A_elementwise", "A_headwise", "B"],
)
def test_each_gate_has_its_shape_no_bias_and_the_swiglu_init(
    options, name, rows, tolerance
):
    torch.manual_seed(0)
    layer = AMEAttention(**WIDE, vocab_size=VOCAB, **options)
    linear = getattr(layer, name)
    assert isinstance(linear, torch.nn.Linear)
    assert linear.bias is None
    assert linear.weight.shape == (rows, WIDE["dmodel"])
    assert linear.weight.dtype == layer.v_proj.weight.dtype
    bound = 2 / math.sqrt(WIDE["dmodel"])
    assert linear.weight.abs().max() <= bound
    std = linear.weight.std().item()
    assert std == pytest.approx(
        TRUNCATED_STD / math.sqrt(WIDE["dmodel"]), rel=tolerance
    )


@pytest.mark.parametrize(
    "activation, mean, rms",
    [("sigmoid", 0.500, 0.535), ("silu", 0.166, 0.513)],
)
def test_a_gate_on_unit_rms_input_starts_where_the_doc_says(activation, mean, rms):
    """The doc's activation table: under the SwiGLU rule the pre-activation has
    std 0.88, so a sigmoid gate starts at 0.500 on average and a SiLU gate at
    0.166, the branch at about half its RMS either way."""
    torch.manual_seed(0)
    layer = AMEAttention(
        **WIDE,
        vocab_size=VOCAB,
        read_gate="elementwise",
        gate_activation=activation,
    )
    x = torch.randn(8, 64, WIDE["dmodel"], generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        phi = torch.sigmoid if activation == "sigmoid" else F.silu
        gate = phi(layer.read_gate_proj(x))
    assert gate.mean().item() == pytest.approx(mean, abs=0.01)
    assert gate.pow(2).mean().sqrt().item() == pytest.approx(rms, abs=0.01)


def test_the_gates_build_on_the_meta_device_without_drawing():
    with torch.device("meta"):
        layer = AMEAttention(
            **DIMS,
            vocab_size=VOCAB,
            read_gate="elementwise",
            output_gate="elementwise",
        )
    assert layer.read_gate_proj.weight.is_meta
    assert layer.output_gate_proj.weight.is_meta


# ---------------------------------------------------------------------------
# The math (reference backend, fp32).


@pytest.mark.parametrize(
    "read_gate, output_gate, activation",
    [("none", "none", None), *GATED],
    ids=["ungated"] + [_ids(c) for c in GATED],
)
def test_the_gated_forward_matches_a_literal_transcription_of_the_doc(
    read_gate, output_gate, activation
):
    layer = _perturb_sink(_gated(5, read_gate, output_gate, activation))
    x, token_ids = _batch()
    torch.testing.assert_close(
        layer(x, token_ids), _spec_forward(layer, x, token_ids), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("activation", GATE_ACTIVATIONS)
@pytest.mark.parametrize("output_gate", OUTPUT_GATES)
def test_headwise_a_is_elementwise_a_with_its_rows_tied_within_each_head(
    activation, output_gate
):
    headwise = _perturb_sink(_gated(4, "headwise", output_gate, activation))
    elementwise = _perturb_sink(_gated(4, "elementwise", output_gate, activation))
    with torch.no_grad():
        elementwise.read_gate_proj.weight.copy_(
            headwise.read_gate_proj.weight.repeat_interleave(headwise.dhead, dim=0)
        )
    x, token_ids = _batch()
    # Not bit for bit: the GEMM need not round d tied rows identically.
    torch.testing.assert_close(
        headwise(x, token_ids), elementwise(x, token_ids), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("activation", GATE_ACTIVATIONS)
def test_b_is_its_gate_times_the_ungated_output(activation):
    ungated = _perturb_sink(_layer(seed=6))
    gated = _perturb_sink(
        _layer(seed=6, output_gate="elementwise", gate_activation=activation)
    )
    for name, tensor in ungated.state_dict().items():
        assert torch.equal(gated.state_dict()[name], tensor), name
    x, token_ids = _batch()
    phi = torch.sigmoid if activation == "sigmoid" else F.silu
    expected = phi(F.linear(x, gated.output_gate_proj.weight)) * ungated(x, token_ids)
    torch.testing.assert_close(gated(x, token_ids), expected, rtol=1e-6, atol=1e-7)


def test_headwise_a_can_switch_a_head_off():
    """The per-head off switch. Under SiLU a zero gate row closes its head
    exactly (silu(0) = 0), and then nothing that head reads reaches the output:
    its value weights can be anything."""
    layer = _perturb_sink(_layer(seed=8, read_gate="headwise", gate_activation="silu"))
    x, token_ids = _batch()
    d = layer.dhead
    with torch.no_grad():
        layer.read_gate_proj.weight[0].zero_()
        before = layer(x, token_ids)
        layer.v_proj.weight[:d].normal_()  # head 0's value rows
        layer.v_sink[0].normal_()
        after = layer(x, token_ids)
        layer.v_proj.weight[d : 2 * d].normal_()  # head 1 is still open
        moved = layer(x, token_ids)
    assert torch.equal(before, after)
    assert not torch.allclose(after, moved)


# ---------------------------------------------------------------------------
# The score table: neither gate enters q.k.


@pytest.mark.parametrize(
    "read_gate, output_gate, activation", GATED, ids=[_ids(c) for c in GATED]
)
def test_the_gates_leave_the_scores_and_their_diagnostics_alone(
    read_gate, output_gate, activation
):
    ungated = _layer(seed=2)
    gated = _gated(2, read_gate, output_gate, activation)
    assert gated.scores_read_hidden_state is False
    _, token_ids = _batch()
    assert torch.equal(gated.score_matrix(token_ids), ungated.score_matrix(token_ids))
    assert gated.attention_diagnostics(token_ids) == ungated.attention_diagnostics(
        token_ids
    )
    # and the weights still ignore the hidden state: any x, the same scores
    x1, _ = _batch(seed=3)
    x2, _ = _batch(seed=4)
    q1, k1, v1 = gated._mixed_qkv(x1, token_ids)
    q2, k2, v2 = gated._mixed_qkv(x2, token_ids)
    _, scores1 = gated._attend_reference(q1, k1, v1, return_scores=True)
    _, scores2 = gated._attend_reference(q2, k2, v2, return_scores=True)
    assert torch.equal(scores1, scores2)


# ---------------------------------------------------------------------------
# The training path.


@pytest.mark.parametrize("activation", GATE_ACTIVATIONS)
@pytest.mark.parametrize("read_gate", ["elementwise", "headwise"])
def test_both_gate_weights_get_finite_nonzero_gradients(read_gate, activation):
    layer = _perturb_sink(_gated(1, read_gate, "elementwise", activation))
    x, token_ids = _batch()
    target = torch.randn(x.shape, generator=torch.Generator().manual_seed(5))
    (layer(x, token_ids) * target).sum().backward()
    for name, weight in _gate_weights(layer).items():
        assert weight.grad is not None, name
        assert torch.isfinite(weight.grad).all(), name
        assert weight.grad.abs().sum() > 0, name
    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    assert missing == []


# The CPU compile tests run dynamo and AOTAutograd with eager kernels
# (backend="aot_eager"). Whether a graph breaks is dynamo's decision whatever the
# backend, and fullgraph=True fails on any break. Inductor's own CPU code cannot
# be used: on torch 2.7.1 its compiled backward of the AME reference path
# corrupts the heap ("malloc(): invalid size") with no gate at all, at d_head 8
# and 64 alike (2026-11-27, on the pre-gate tree). Inductor is the GPU pass's.
CPU_COMPILE_BACKEND = "aot_eager"


@pytest.mark.parametrize("read_gate", ["elementwise", "headwise"])
def test_the_gated_layer_compiles_to_one_graph_in_train_mode(read_gate):
    """The eval-time readout sits behind `if not self.training`, which dynamo
    folds away in train mode: fullgraph=True fails on any graph break."""
    torch._dynamo.reset()
    layer = _perturb_sink(
        _layer(seed=7, read_gate=read_gate, output_gate="elementwise")
    )
    assert layer.training
    x, token_ids = _batch()
    eager = layer(x, token_ids)
    compiled = torch.compile(layer, fullgraph=True, backend=CPU_COMPILE_BACKEND)(
        x, token_ids
    )
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)
    compiled.sum().backward()
    for name, weight in _gate_weights(layer).items():
        assert weight.grad is not None and weight.grad.abs().sum() > 0, name
    assert layer.gate_statistics() == {}, "a train-mode forward recorded the gates"


@pytest.mark.parametrize("read_gate", ["elementwise", "headwise"])
def test_a_gated_state_dict_round_trips_into_a_fresh_model(read_gate):
    options = [
        f"+common.attention.ame={{read_gate: {read_gate}, output_gate: elementwise}}"
    ]
    trained = _h4_model(options, seed=1)
    fresh = _h4_model(options, seed=2)
    with torch.no_grad():
        for weight in _gate_weights(trained).values():
            weight.mul_(3.0)  # off the init: a fresh init cannot match by chance
        assert not torch.equal(fresh(_fixed_ids()), trained(_fixed_ids()))
        fresh.load_state_dict(trained.state_dict(), strict=True)
        assert torch.equal(fresh(_fixed_ids()), trained(_fixed_ids()))
    assert list(fresh.state_dict()) == list(trained.state_dict())


# ---------------------------------------------------------------------------
# Optimizer groups and the FLOP accounting.


def test_the_gate_weights_decay_and_are_not_banks():
    model = _h4_model(
        ["+common.attention.ame={read_gate: headwise, output_gate: elementwise}"]
    )
    gates = {id(p) for p in _gate_weights(model).values()}
    assert len(gates) == 2 * 4
    groups = parameter_groups(model, 0.1, lr=1e-3, bank_lr_ratio=4.0)
    decayed = [g for g in groups if g["weight_decay"] == 0.1 and "lr" not in g]
    assert len(decayed) == 1
    assert gates <= {id(p) for p in decayed[0]["params"]}
    for group in groups:
        if group is not decayed[0]:
            assert gates.isdisjoint(id(p) for p in group["params"])
    banks = [g for g in groups if "lr" in g]
    assert banks and all(g["lr"] == pytest.approx(4e-3) for g in banks)


@pytest.mark.parametrize(
    "options, per_layer",
    [
        ("{read_gate: elementwise}", 256 * 256),
        ("{read_gate: headwise}", 4 * 256),
        ("{output_gate: elementwise}", 256 * 256),
        ("{read_gate: elementwise, output_gate: elementwise}", 2 * 256 * 256),
        ("{read_gate: headwise, output_gate: elementwise}", 4 * 256 + 256 * 256),
    ],
)
def test_the_mfu_count_rises_by_exactly_the_gate_weights(options, per_layer):
    plain = _h4_model(device="meta")
    gated = _h4_model([f"+common.attention.ame={options}"], device="meta")
    added = sum(p.numel() for p in _gate_weights(gated).values())
    assert added == 4 * per_layer
    assert (
        model_flops_per_token(gated, 16) - model_flops_per_token(plain, 16) == 6 * added
    )


# ---------------------------------------------------------------------------
# Matching (the doc's Matching table): with k elementwise gates the FF gives up
# k * dmodel/3 of width, dff = 32 * round((26 - 2k) * H / 3), and AME matches the
# dense model of its H in active parameters (a bank counts the one row a token
# reads).

MATCHED_DFF = {
    0: {12: 3328, 16: 4448, 20: 5536},
    1: {12: 3072, 16: 4096, 20: 5120},
    2: {12: 2816, 16: 3744, 20: 4704},
}
# (read_gate, output_gate) -> (k, active parameters against dense, in percent)
MATCHING_TABLE = {
    ("none", "none"): (0, {12: 0.02, 16: 0.17, 20: -0.12}),
    ("headwise", "none"): (0, {12: 0.08, 16: 0.24, 20: -0.03}),
    ("elementwise", "none"): (1, {12: 0.02, 16: 0.02, 20: 0.02}),
    ("none", "elementwise"): (1, {12: 0.02, 16: 0.02, 20: 0.02}),
    ("headwise", "elementwise"): (1, {12: 0.08, 16: 0.09, 20: 0.10}),
    ("elementwise", "elementwise"): (2, {12: 0.02, 16: -0.13, 20: 0.15}),
}


def _standard(name, overrides=()):
    """A configs/standard/ config, composed from its own directory (as
    `--config-path configs/standard` would): composed as `standard/<name>` from
    configs/, Hydra nests the whole config under a `standard` key."""
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs", "standard")
    ):
        cfg = compose(config_name=name, overrides=list(overrides))
    with torch.device("meta"):
        return cfg, instantiate(cfg.model, _convert_="all")


def _active_parameters(model):
    banks = {
        id(p)
        for layer in _ame_layers(model)
        for p in (layer.q_bank.weight, layer.k_bank.weight)
    }
    return sum(p.shape[-1] if id(p) in banks else p.numel() for p in model.parameters())


@pytest.mark.parametrize("h", [12, 16, 20])
def test_matched_gated_ame_has_the_dense_models_active_parameters(h):
    _, dense = _standard(f"dense_h{h}")
    dense_active = _active_parameters(dense)
    found = {}
    for (read_gate, output_gate), (k, percent) in MATCHING_TABLE.items():
        dff = 32 * round((26 - 2 * k) * h / 3)
        assert dff == MATCHED_DFF[k][h]
        overrides = [f"common.dff={dff}"]
        for key, value in (("read_gate", read_gate), ("output_gate", output_gate)):
            if value != "none":
                overrides.append(f"++common.attention.ame.{key}={value}")
        cfg, ame = _standard(f"ame_h{h}", overrides)
        layers = _ame_layers(ame)
        assert len(layers) == h
        assert {(m.read_gate, m.output_gate) for m in layers} == {
            (read_gate, output_gate)
        }
        relative = _active_parameters(ame) / dense_active - 1
        assert abs(relative) <= 0.003, f"{read_gate}/{output_gate} at h{h}"
        found[(read_gate, output_gate)] = (round(100 * relative, 2), percent[h])
    mismatched = {key: v for key, v in found.items() if v[0] != v[1]}
    assert not mismatched, f"(measured %, the doc's %) at h{h}: {mismatched}"


def test_the_shipped_v4_config_reproduces_its_own_accounting():
    """The anchor the table reproduces: v4's comment in ame_h16.yaml, +590,336."""
    cfg, ame = _standard("ame_h16")
    _, dense = _standard("dense_h16")
    assert cfg.common.dff == MATCHED_DFF[0][16]
    assert _active_parameters(ame) - _active_parameters(dense) == 590_336


# ---------------------------------------------------------------------------
# The readout: gate statistics from eval forwards only.


def _expected_readout(layer, xs):
    """The readout computed by hand from the gate values on `xs`."""
    phi = torch.sigmoid if layer.gate_activation == "sigmoid" else F.silu
    expected = {}
    for which in ("read_gate", "output_gate"):
        if getattr(layer, which) == "none":
            continue
        weight = getattr(layer, f"{which}_proj").weight
        values = torch.cat([phi(F.linear(x, weight)).flatten(0, 1) for x in xs])
        values = values.double()
        expected[f"{which}/mean"] = values.mean().item()
        expected[f"{which}/below_0.05"] = (values < 0.05).double().mean().item()
        if layer.gate_activation == "silu":
            expected[f"{which}/below_0"] = (values < 0).double().mean().item()
        if which == "read_gate" and layer.read_gate == "headwise":
            expected[f"{which}/head_mean_min"] = values.mean(0).min().item()
    return expected


@pytest.mark.parametrize(
    "read_gate, output_gate, activation", GATED, ids=[_ids(c) for c in GATED]
)
def test_eval_forwards_record_the_gates_and_reset_clears_them(
    read_gate, output_gate, activation
):
    layer = _gated(3, read_gate, output_gate, activation).eval()
    batches = [_batch(seed=s) for s in (11, 12)]
    with torch.no_grad():
        for x, token_ids in batches:
            layer(x, token_ids)
    readout = layer.gate_statistics()
    expected = _expected_readout(layer, [x for x, _ in batches])
    assert set(readout) == set(expected)
    for key, value in expected.items():
        assert isinstance(readout[key], float), key
        assert readout[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
    layer.reset_gate_statistics()
    assert layer.gate_statistics() == {}


def test_a_train_forward_records_nothing_and_an_ungated_layer_offers_nothing():
    gated = _layer(read_gate="elementwise", output_gate="elementwise")
    x, token_ids = _batch()
    gated(x, token_ids)
    assert gated.gate_statistics() == {}
    ungated = _layer().eval()
    with torch.no_grad():
        ungated(x, token_ids)
    assert ungated.gate_statistics() == {}


def test_a_compiled_eval_forward_still_records_the_gates():
    """In eval mode the readout is a @torch.compiler.disable call: a graph
    break, so it runs eagerly between two compiled pieces and still counts."""
    torch._dynamo.reset()
    layer = _gated(4, "headwise", "elementwise", "silu").eval()
    x, token_ids = _batch()
    with torch.no_grad():
        eager_out = layer(x, token_ids)
    eager = layer.gate_statistics()
    layer.reset_gate_statistics()
    with torch.no_grad():
        compiled_out = torch.compile(layer, backend=CPU_COMPILE_BACKEND)(x, token_ids)
    torch.testing.assert_close(compiled_out, eager_out, rtol=1e-5, atol=1e-6)
    compiled = layer.gate_statistics()
    assert set(compiled) == set(eager)
    for key, value in eager.items():
        assert compiled[key] == pytest.approx(value, rel=1e-6), key


def _eval_stub(model, batches):
    """Exercise Trainer.eval on a model without building a Trainer, the idiom of
    tests/test_ame.py's _instrumentation_stub: eval reads the model, the eval
    loader, the metric logger and calculate_loss, and the step counters."""
    from src.core.trainer import LossMetrics, Trainer

    class Logger:
        def __init__(self):
            self.logged = {}
            self.step = 0

        def set_step(self, step):
            self.step = step

        def log(self, name, value):
            self.logged[name] = value

        def flush_accumulated_metrics(self):
            pass

    class Stub:
        eval = Trainer.eval
        _should_log_eval_input = Trainer._should_log_eval_input
        _reset_gate_statistics = Trainer._reset_gate_statistics
        _log_gate_statistics = Trainer._log_gate_statistics

        def calculate_loss(self, batch):
            logits = self.model(batch[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].reshape(-1))
            return LossMetrics(total_loss=loss, reported_loss=loss)

    stub = Stub()
    stub.model = model
    stub.metric_logger = Logger()
    stub.eval_dataloader = batches
    stub.n_eval_steps = len(batches)
    stub.step = 10
    stub.eval_interval = 5
    stub.device = torch.device("cpu")
    return stub


def test_the_trainers_eval_logs_each_gated_layers_readout_of_that_eval_only():
    model = _h4_model(
        [
            "+common.attention.mix={type: ame, layers: every_2nd}",
            "+common.attention.ame={read_gate: headwise, gate_activation: silu}",
        ],
        attention="gqa",
    )
    generator = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 96, (2, 17), generator=generator) for _ in range(2)]
    # A stale eval-mode forward the eval must not count.
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 96, (2, 16), generator=generator))
    model.train()

    stub = _eval_stub(model, batches)
    stub.eval()

    logged = stub.metric_logger.logged
    assert "eval/loss" in logged
    gates = {name: v for name, v in logged.items() if name != "eval/loss"}
    layers = [
        "encoder.blocks.1.attention_layer.layer",
        "encoder.blocks.3.attention_layer.layer",
    ]
    keys = [
        "read_gate/mean",
        "read_gate/below_0.05",
        "read_gate/below_0",
        "read_gate/head_mean_min",
    ]
    assert set(gates) == {f"eval/{layer}/{key}" for layer in layers for key in keys}
    assert stub.metric_logger.step == 10, "eval restores the step"

    # The same numbers from the eval batches alone.
    model.eval()
    with torch.no_grad():
        for layer in _ame_layers(model):
            layer.reset_gate_statistics()
        for batch in batches:
            model(batch[:, :-1])
    for name in layers:
        expected = model.get_submodule(name).gate_statistics()
        for key, value in expected.items():
            assert gates[f"eval/{name}/{key}"] == pytest.approx(value, rel=1e-12), key


def test_an_eval_without_gates_logs_the_loss_only():
    generator = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 96, (2, 17), generator=generator)]
    stub = _eval_stub(_h4_model(), batches)
    stub.eval()
    assert list(stub.metric_logger.logged) == ["eval/loss"]


# ---------------------------------------------------------------------------
# GPU: Stage 0 of ame_attention_gate.md's "Experiments at h12". Built at the H
# family's d_head of 64: flex's compiled kernel needs at least 16, and a GPU test
# at a shape the runs never use would prove nothing about them.

GPU_DIMS = dict(dmodel=256, q_heads=4, kv_heads=2, seq_len=64)
GATED_H4 = [
    "+common.attention.ame="
    "{read_gate: headwise, output_gate: elementwise, gate_activation: silu}"
]
BF16_EPS = 2.0**-8


def _cuda_layer(read_gate, output_gate, activation, seed=0):
    torch.manual_seed(seed)
    layer = AMEAttention(
        **GPU_DIMS,
        vocab_size=VOCAB,
        read_gate=read_gate,
        output_gate=output_gate,
        gate_activation=activation,
    ).cuda()
    assert layer.dhead == 64
    return _perturb_sink(layer)


def _cuda_batch(seed=1, batch=2):
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(
        0, VOCAB, (batch, GPU_DIMS["seq_len"]), generator=generator
    )
    x = torch.randn(batch, GPU_DIMS["seq_len"], GPU_DIMS["dmodel"], generator=generator)
    target = torch.randn(x.shape, generator=generator)
    return x.cuda(), token_ids.cuda(), target.cuda()


def _forward_backward(layer, x, token_ids, target, autocast=False, call=None):
    """The output and the gradient of <output, target> for every parameter of
    `layer` and for x. `call` runs the forward instead of the layer itself (a
    compiled wrapper); the gradients are read off the layer either way."""
    layer.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = (call or layer)(x, token_ids)
    (out.float() * target).sum().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in layer.named_parameters()}
    grads["x"] = x.grad.detach().float().clone()
    return out.detach().float(), grads


def _relative(a, b):
    """max |a - b| over max |b|: an error on the scale of the tensor."""
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "read_gate, output_gate, activation", GATED, ids=[_ids(c) for c in GATED]
)
def test_on_cuda_flex_matches_the_reference_kernel_under_every_gate(
    read_gate, output_gate, activation
):
    """fp32: the production kernel and the materialized reference compute the
    same gated layer, forward and backward, for every parameter and the input."""
    layer = _cuda_layer(read_gate, output_gate, activation)
    x, token_ids, target = _cuda_batch()
    layer.backend = "reference"
    out_ref, grads_ref = _forward_backward(layer, x, token_ids, target)
    layer.backend = "flex"
    out_flex, grads_flex = _forward_backward(layer, x, token_ids, target)
    torch.testing.assert_close(out_flex, out_ref, rtol=1e-4, atol=1e-5)
    errors = {n: _relative(grads_flex[n], grads_ref[n]) for n in grads_ref}
    assert max(errors.values()) < 1e-3, errors


@pytest.mark.gpu
@pytest.mark.parametrize(
    "read_gate, output_gate, activation", GATED, ids=[_ids(c) for c in GATED]
)
def test_on_cuda_bf16_costs_flex_what_it_costs_the_reference_under_every_gate(
    read_gate, output_gate, activation
):
    """Under bf16 autocast both kernels lose resolution; flex may not lose more
    than 4x what the reference loses against fp32, in the output or in any
    gradient (the CPU test_the_backends_still_agree_under_bf16, with gates)."""
    layer = _cuda_layer(read_gate, output_gate, activation)
    x, token_ids, target = _cuda_batch()
    layer.backend = "reference"
    out32, grads32 = _forward_backward(layer, x, token_ids, target)
    results = {}
    for backend in ("reference", "flex"):
        layer.backend = backend
        results[backend] = _forward_backward(layer, x, token_ids, target, autocast=True)
    (out_ref, grads_ref), (out_flex, grads_flex) = results["reference"], results["flex"]
    cost = max(_relative(out_ref, out32), BF16_EPS)
    assert _relative(out_flex, out32) <= 4 * cost
    for name, truth in grads32.items():
        assert torch.isfinite(grads_flex[name]).all(), name
        allowed = 4 * max(_relative(grads_ref[name], truth), BF16_EPS)
        assert _relative(grads_flex[name], truth) <= allowed, name


@pytest.mark.gpu
@pytest.mark.parametrize(
    "read_gate, output_gate, activation",
    [
        ("elementwise", "elementwise", "sigmoid"),
        ("headwise", "elementwise", "silu"),
        ("elementwise", "none", "silu"),
    ],
    ids=["Aelem_B_sigmoid", "Ahead_B_silu", "Aelem_silu"],
)
def test_on_cuda_inductor_compiles_the_gated_layer(read_gate, output_gate, activation):
    """Inductor on the production path (flex): the compiled gated layer equals
    the eager one, forward and backward. The CPU tests cannot run this: on torch
    2.7.1 inductor's CPU backward of the AME reference path corrupts the heap."""
    torch._dynamo.reset()
    layer = _cuda_layer(read_gate, output_gate, activation)
    x, token_ids, target = _cuda_batch()
    out_eager, grads_eager = _forward_backward(layer, x, token_ids, target)
    compiled = torch.compile(layer)
    out_comp, grads_comp = _forward_backward(layer, x, token_ids, target, call=compiled)
    torch.testing.assert_close(out_comp, out_eager, rtol=1e-4, atol=1e-5)
    errors = {n: _relative(grads_comp[n], grads_eager[n]) for n in grads_eager}
    assert max(errors.values()) < 1e-3, errors


def _train_graph_breaks(model, token_ids):
    """Graph breaks in one compile_blocks_full training step under bf16
    autocast, as main.py compiles the stack: block.compile() per block."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    for block in model.encoder.blocks:
        block.compile()
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(token_ids)
    logits.float().sum().backward()
    return sum(counters["graph_break"].values())


@pytest.mark.gpu
def test_on_cuda_compile_blocks_full_gains_no_graph_break_from_the_gates():
    """The training graph carries no readout: a gated stack breaks no more
    often than the ungated one. In eval the readout is a break by design, and
    it still records under the compiled blocks."""
    generator = torch.Generator().manual_seed(0)
    token_ids = torch.randint(0, 96, (2, 16), generator=generator).cuda()
    plain = _train_graph_breaks(_h4_model(device="cuda"), token_ids)
    gated_model = _h4_model(GATED_H4, device="cuda")
    gated = _train_graph_breaks(gated_model, token_ids)
    assert gated <= plain, f"{gated} graph breaks with the gates, {plain} without"

    gated_model.eval()
    for layer in _ame_layers(gated_model):
        layer.reset_gate_statistics()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gated_model(token_ids)
    for layer in _ame_layers(gated_model):
        assert set(layer.gate_statistics()) == {
            "read_gate/mean",
            "read_gate/below_0.05",
            "read_gate/below_0",
            "read_gate/head_mean_min",
            "output_gate/mean",
            "output_gate/below_0.05",
            "output_gate/below_0",
        }


@pytest.mark.gpu
def test_on_cuda_the_memory_parallel_controller_trains_with_the_gates():
    """The h12 recipe's bank path (mem_parallel_size 1), as main.py builds it on
    a bf16 stack: the banks in the controller, the gate weights in the backbone
    AdamW, and both move."""
    from src.core.memory_parallel import MemoryParallelController, ShardedBank

    model = _h4_model(GATED_H4, device="cuda")
    banks = MemoryParallelController(model, 1, compute_dtype=torch.bfloat16)
    gates = _gate_weights(model)
    ignored = {id(p) for p in banks.ignored_parameters()}
    assert len(gates) == 2 * 4 and all(id(p) not in ignored for p in gates.values())
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model, 0.1, exclude=banks.ignored_parameters(), lr=1e-3, bank_lr_ratio=4.0
        ),
        lr=1e-3,
    )
    banks.build_optimizer(lr=4e-3, weight_decay=0.0)
    stepped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in stepped for p in gates.values())

    sharded = [m for m in model.modules() if isinstance(m, ShardedBank)]
    assert len(sharded) == 2 * 4
    gates_before = {n: p.detach().clone() for n, p in gates.items()}
    banks_before = [m.weight.detach().clone() for m in sharded]
    generator = torch.Generator().manual_seed(0)
    losses = []
    model.train()
    for _ in range(3):
        banks.materialize()
        batch = torch.randint(0, 96, (4, 17), generator=generator).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].reshape(-1))
        loss.backward()
        banks.reduce()
        optimizer.step()
        optimizer.zero_grad()
        banks.step(lr=4e-3)
        banks.zero_grad()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses), losses
    for name, weight in gates.items():
        assert not torch.equal(weight, gates_before[name]), name
    for module, before in zip(sharded, banks_before):
        assert not torch.equal(module.weight, before)


FSDP2_SHARDED = [
    "src.core.model.TransformerEmbedding",
    "src.core.model.TransformerBlock",
    "src.core.model.TransformerHead",
]


def _fsdp2_worker(rank, world_size, port, folder, out_path):
    """One rank: the gated h4 stack under the production FSDP2 setup (bf16
    mixed precision), 2 steps, a dcp checkpoint through the trainer's
    TrainingState, 2 more steps; then a fresh model from another seed resumes
    from the checkpoint and takes the same 2 steps; then an eval forward."""
    import json

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from omegaconf import OmegaConf

    from src.core.checkpointing import TrainingState
    from src.core.distributed_training import setup_fsdp2_model

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        fsdp = OmegaConf.create({"modules_to_shard": FSDP2_SHARDED})

        def build(seed):
            model = setup_fsdp2_model(
                _h4_model(GATED_H4, seed=seed, device="cuda"), fsdp
            )
            optimizer = torch.optim.AdamW(parameter_groups(model, 0.1), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            return model, optimizer, scheduler

        def step(state, batch):
            model, optimizer, scheduler = state
            model.train()
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), batch[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            return loss.item()

        generator = torch.Generator().manual_seed(100 + rank)  # each rank its data
        batches = [
            torch.randint(0, 96, (2, 17), generator=generator).cuda() for _ in range(4)
        ]
        straight = build(seed=0)
        gates = _gate_weights(straight[0])
        sharded = all(hasattr(p, "full_tensor") for p in gates.values())
        before = {n: p.full_tensor().clone() for n, p in gates.items()}
        first = [step(straight, b) for b in batches[:2]]
        dcp.save({"app": TrainingState(*straight)}, checkpoint_id=folder)
        after = [step(straight, b) for b in batches[2:]]
        moved = all(
            not torch.equal(p.full_tensor(), before[n]) for n, p in gates.items()
        )

        resumed = build(seed=1)
        dcp.load({"app": TrainingState(*resumed)}, checkpoint_id=folder)
        again = [step(resumed, b) for b in batches[2:]]

        model = resumed[0]
        model.eval()
        for layer in _ame_layers(model):
            layer.reset_gate_statistics()
        with torch.no_grad():
            model(batches[0][:, :-1])
        readouts = [sorted(layer.gate_statistics()) for layer in _ame_layers(model)]

        if rank == 0:
            with open(out_path, "w") as f:
                json.dump(
                    dict(
                        first=first,
                        after=after,
                        again=again,
                        sharded=sharded,
                        moved=moved,
                        readouts=readouts,
                    ),
                    f,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
def test_on_two_gpus_fsdp2_trains_resumes_and_reads_out_the_gates(tmp_path):
    """The 4-GPU twins' path at two ranks: the gate weights are sharded DTensors
    and train; a resumed run takes the steps the uninterrupted one took; the
    eval readout works per rank."""
    import json

    import torch.multiprocessing as mp

    from conftest import free_port

    if torch.cuda.device_count() < 2:
        pytest.fail("this test needs 2 GPUs; submit the GPU suite with two")
    out = tmp_path / "fsdp2.json"
    mp.spawn(
        _fsdp2_worker,
        args=(2, free_port(), str(tmp_path / "checkpoint"), str(out)),
        nprocs=2,
        join=True,
    )
    result = json.loads(out.read_text())
    assert result["sharded"], "the gate weights are not FSDP2-sharded"
    assert result["moved"], "a gate weight did not train under FSDP2"
    assert all(math.isfinite(v) for v in result["first"] + result["after"])
    torch.testing.assert_close(
        torch.tensor(result["again"]), torch.tensor(result["after"]), rtol=1e-4, atol=0
    )
    assert len(result["readouts"]) == 4
    for keys in result["readouts"]:
        assert keys == sorted(
            [
                "read_gate/mean",
                "read_gate/below_0.05",
                "read_gate/below_0",
                "read_gate/head_mean_min",
                "output_gate/mean",
                "output_gate/below_0.05",
                "output_gate/below_0",
            ]
        )
