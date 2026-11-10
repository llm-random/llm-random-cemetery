"""attention.document_mask: attention stays inside each document of a packed row.

The packer concatenates documents, each ending in EOS, into one row
(src.core.datasets). With the mask, query i sees key j only if j <= i and both
are in the same document, and AME's KV shift does not reach across a boundary
(src.core.document_mask). Positions are not reset: RoPE and ALiBi are relative,
so a document's output must not depend on where in the row it sits.

The central check is exactly that: every document of a packed row gets the
output it gets when it is run alone -- per layer type and through a whole model.
"""

import os
import warnings

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)
from src.core.attention import build_attention
from src.core.document_mask import (
    allowed_keys,
    continues_document,
    document_ids,
    document_score_mod,
)
from src.definitions import check_point, validate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOS = 0
VOCAB = 128
SEQ = 24
DIMS = dict(dmodel=64, q_heads=4, kv_heads=2, seq_len=SEQ, vocab_size=VOCAB)

# Two rows with different boundaries, so a mask indexed by the wrong batch row
# fails. Row 0: a 5-token document, an 8-token one, a document that is nothing
# but its EOS, then 10 tokens cut by the row end. Row 1: 11, then 13 cut.
# (start, end) of each document; an EOS closes every document but the last.
SEGMENTS = [
    [(0, 5), (5, 13), (13, 14), (14, 24)],
    [(0, 11), (11, 24)],
]


def _packed_ids(seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (len(SEGMENTS), SEQ), generator=generator)
    for row, segments in enumerate(SEGMENTS):
        for _, end in segments[:-1]:
            ids[row, end - 1] = EOS
    return ids


def _x(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(len(SEGMENTS), SEQ, DIMS["dmodel"], generator=generator)


def _layer(attention, seed=0, **kwargs):
    torch.manual_seed(seed)
    return build_attention(
        0, attention=attention, n_blocks=1, eos_token_id=EOS, **DIMS, **kwargs
    )


def _call(layer, x, ids):
    return layer(x, ids) if getattr(layer, "wants_token_ids", False) else layer(x)


# ---------------------------------------------------------------------------
# The document ids


def test_a_position_belongs_to_the_document_its_eos_count_says():
    ids = torch.tensor([[5, 7, EOS, 9, EOS, EOS, 3]])
    # An EOS belongs to the document it ends; an EOS right after another is a
    # document of its own.
    assert document_ids(ids, EOS).tolist() == [[0, 0, 0, 1, 1, 2, 3]]
    assert continues_document(document_ids(ids, EOS)).tolist() == [
        [False, True, True, False, True, False, False]
    ]


def test_allowed_keys_is_causal_and_within_a_document():
    doc = document_ids(torch.tensor([[5, EOS, 9, 9]]), EOS)  # docs 0 0 1 1
    assert allowed_keys(doc)[0].int().tolist() == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ]


def test_the_score_mod_masks_across_documents_and_keeps_the_rest():
    doc = torch.tensor([[0, 0, 1]])
    inner = lambda score, b, h, q, kv: score + 1.0  # noqa: E731
    mod = document_score_mod(doc, inner)
    score = torch.tensor(2.0)
    zero, one, two = (torch.tensor(i) for i in range(3))
    assert mod(score, zero, zero, one, zero).item() == 3.0  # same doc: inner applied
    assert mod(score, zero, zero, two, one).item() == float("-inf")


# ---------------------------------------------------------------------------
# One layer: a packed row equals its documents run alone


LAYERS = {
    "mha": {"type": "mha"},
    "mqa": {"type": "mqa"},
    "gqa": {"type": "gqa"},
    "gqa_nope": {"type": "gqa", "gqa": {"rope": False}},
    # S2-PS, the AME recipe of the runs: ALiBi, KV shift, sink.
    "ame": {"type": "ame", "ame": {"shift": True}},
    "ame_no_sink": {"type": "ame", "ame": {"shift": True, "sink": False}},
}


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_each_document_gets_the_output_it_gets_alone(name):
    layer = _layer({**LAYERS[name], "document_mask": True})
    ids, x = _packed_ids(), _x()
    with torch.no_grad():
        packed = _call(layer, x, ids)
        for row, segments in enumerate(SEGMENTS):
            for start, end in segments:
                alone = _call(
                    layer, x[row : row + 1, start:end], ids[row : row + 1, start:end]
                )
                torch.testing.assert_close(
                    packed[row, start:end],
                    alone[0],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"{name}: row {row}, document {start}:{end}",
                )


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_without_the_mask_the_later_documents_see_the_earlier_ones(name):
    """The check above has teeth: unmasked, a later document's output moves."""
    layer = _layer(LAYERS[name])
    ids, x = _packed_ids(), _x()
    with torch.no_grad():
        packed = _call(layer, x, ids)
        start, end = SEGMENTS[0][1]
        alone = _call(layer, x[:1, start:end], ids[:1, start:end])
    assert not torch.allclose(packed[0, start:end], alone[0], rtol=1e-3, atol=1e-3)


def test_the_ame_shift_stops_at_a_boundary():
    """Without the reset, a document's first key and value would mix in the
    previous document's EOS. Pins alpha and beta at fully shifted (0, 1): the
    first position of every document then has a zero key and value."""
    layer = _layer({"type": "ame", "ame": {"shift": True}, "document_mask": True})
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([[0.0, 1.0]] * DIMS["q_heads"]))
        layer.beta.copy_(torch.tensor([[0.0, 1.0]] * DIMS["q_heads"]))
        ids, x = _packed_ids(), _x()
        _, k, v = layer._mixed_qkv(x, ids, layer._documents(ids))
    for row, segments in enumerate(SEGMENTS):
        for start, _ in segments:
            assert torch.equal(k[row, :, start], torch.zeros_like(k[row, :, start]))
            assert torch.equal(v[row, :, start], torch.zeros_like(v[row, :, start]))
    assert k[0, :, 1].abs().sum() > 0  # inside a document the shift still acts


# ---------------------------------------------------------------------------
# Off is the old layer, bit for bit


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_off_is_the_plain_layer_bit_for_bit(name):
    """Built the way every existing config builds it (no document_mask key, no
    eos_token_id) against the switch written out as false."""
    torch.manual_seed(0)
    before = build_attention(0, attention=LAYERS[name], n_blocks=1, **DIMS)
    off = _layer({**LAYERS[name], "document_mask": False})
    assert not off.document_mask
    assert getattr(off, "wants_token_ids", False) == (name.startswith("ame"))
    for (key, p), (_, q) in zip(
        before.named_parameters(), off.named_parameters(), strict=True
    ):
        assert torch.equal(p, q), key
    ids, x = _packed_ids(), _x()
    with torch.no_grad():
        torch.testing.assert_close(
            _call(off, x, ids), _call(before, x, ids), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# Flex (the CUDA path) against the reference (the CPU path)


def test_dense_flex_matches_the_reference():
    layer = _layer({"type": "gqa", "document_mask": True})
    generator = torch.Generator().manual_seed(3)
    q, k, v = (
        torch.randn(len(SEGMENTS), DIMS["q_heads"], SEQ, 16, generator=generator)
        for _ in range(3)
    )
    doc = document_ids(_packed_ids(), EOS)
    warnings.filterwarnings("ignore")  # uncompiled flex on CPU warns
    torch.testing.assert_close(
        layer._attend_documents_flex(q, k, v, doc),
        layer._attend_documents_reference(q, k, v, doc),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("sink", [True, False])
def test_ame_flex_matches_the_reference(sink):
    attention = {
        "type": "ame",
        "ame": {"shift": True, "sink": sink, "backend": "flex"},
        "document_mask": True,
    }
    flex = _layer(attention)
    reference = _layer(
        {**attention, "ame": {**attention["ame"], "backend": "reference"}}
    )
    reference.load_state_dict(flex.state_dict())
    with torch.no_grad():  # a sink that is not zero, so the merge is exercised
        for layer in (flex, reference):
            if sink:
                layer.k_sink.normal_(generator=torch.Generator().manual_seed(4))
                layer.v_sink.normal_(generator=torch.Generator().manual_seed(5))
    ids, x = _packed_ids(), _x()
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        torch.testing.assert_close(
            flex(x, ids), reference(x, ids), rtol=1e-4, atol=1e-5
        )


# ---------------------------------------------------------------------------
# A whole model


def _h4_model(overrides, seq_len=SEQ):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=[f"common.sequence_length={seq_len}", *overrides],
        )
    torch.manual_seed(0)
    return instantiate(cfg.model, _convert_="all")


@pytest.mark.parametrize(
    "attention",
    [
        "common.attention.type=gqa",
        "common.attention.type=ame",
        "+common.attention.mix={type: ame, layers: every_2nd}",
    ],
)
def test_a_whole_model_gives_each_document_its_own_logits(attention):
    """Through every layer, the embedding and the head: a document's logits do
    not depend on what precedes it in the row."""
    model = _h4_model([attention, "+common.attention.document_mask=true"])
    model.eval()
    ids = _packed_ids()
    with torch.no_grad():
        packed = model(ids)
        for row, segments in enumerate(SEGMENTS):
            for start, end in segments:
                alone = model(ids[row : row + 1, start:end])
                torch.testing.assert_close(
                    packed[row, start:end], alone[0], rtol=1e-4, atol=1e-4
                )


def test_the_masked_model_trains_and_every_parameter_gets_a_gradient():
    model = _h4_model(
        ["common.attention.type=ame", "+common.attention.document_mask=true"]
    )
    ids = _packed_ids()
    logits = model(ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), ids[:, 1:].flatten()
    )
    loss.backward()
    assert torch.isfinite(loss)
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, missing


# ---------------------------------------------------------------------------
# AME's diagnostics see the mask too


def test_the_diagnostics_score_only_a_rows_own_document():
    layer = _layer({"type": "ame", "ame": {"shift": True}, "document_mask": True})
    ids = _packed_ids()
    scores = layer.score_matrix(ids)  # [B, H, n, 1 + n], the sink first
    doc = document_ids(ids, EOS)
    visible = torch.isfinite(scores[:, :, :, 1:])
    expected = allowed_keys(doc).unsqueeze(1).expand_as(visible)
    assert torch.equal(visible, expected)
    assert torch.isfinite(scores[:, :, :, 0]).all()  # the sink, visible to all

    readout = layer.attention_diagnostics(ids, max_batch=2)
    assert 0.0 <= readout["attention_entropy/normalized"] <= 1.0 + 1e-6
    for value in readout.values():
        assert value == value  # no NaN


# ---------------------------------------------------------------------------
# Config


def _compose_tiny(overrides):
    warnings.filterwarnings("ignore")
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(REPO_ROOT, "configs")
    ):
        return compose(config_name="tiny_local", overrides=list(overrides))


@pytest.mark.parametrize("attention_type", ["gqa", "ame"])
def test_the_config_switch_reaches_every_layer(attention_type):
    cfg = _compose_tiny(
        [
            f"common.attention.type={attention_type}",
            "common.attention.document_mask=true",
            "common.vocab_size=512",
        ]
    )
    check_point(cfg)
    assert cfg.common.eos_token_id == EOS
    torch.manual_seed(0)
    model = instantiate(cfg.model, _convert_="all")
    layers = [b.attention_layer.layer for b in model.encoder.blocks]
    assert all(layer.document_mask and layer.eos_token_id == EOS for layer in layers)
    assert all(b.attention_layer.wants_token_ids for b in model.encoder.blocks)


def test_the_default_config_leaves_the_mask_off():
    cfg = _compose_tiny([])
    assert cfg.common.attention.document_mask is False
    model = instantiate(cfg.model, _convert_="all")
    assert not any(b.attention_layer.layer.document_mask for b in model.encoder.blocks)


def test_the_mask_without_an_eos_id_fails_before_submission():
    cfg = _compose_tiny(
        ["common.attention.document_mask=true", "common.eos_token_id=null"]
    )
    with pytest.raises(ValueError, match="eos_token_id"):
        validate(cfg)
    with pytest.raises(ValueError, match="eos_token_id"):
        build_attention(
            0, attention={"type": "gqa", "document_mask": True}, n_blocks=1, **DIMS
        )


def test_a_planned_type_refuses_the_mask():
    with pytest.raises(ValueError, match="not implemented"):
        _layer({"type": "mla", "document_mask": True})


def test_the_eos_id_is_the_tokenizers():
    """The model config's eos_token_id must be what the packer appends:
    GPT-NeoX's <|endoftext|> (configs/_trainer/llama.yaml picks the tokenizer)."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", local_files_only=True
        )
    except Exception as exc:  # not cached on this machine
        pytest.skip(f"gpt-neox-20b tokenizer not cached: {exc}")
    cfg = _compose_tiny([])
    assert cfg.trainer.train_dataloader.dataset.tokenize_fn._target_.endswith(
        "gpt_neox_tokenize_fn"
    )
    assert tokenizer.eos_token_id == cfg.common.eos_token_id
    # The packer's tokenize_fn appends exactly one EOS and adds nothing else.
    encoded = tokenizer("hello world" + tokenizer.eos_token)["input_ids"]
    assert encoded.count(tokenizer.eos_token_id) == 1 and encoded[-1] == EOS


# ---------------------------------------------------------------------------
# CUDA: the production kernel. On CPU flex runs uncompiled (the math fallback),
# so only a GPU reaches the compiled Triton kernel that runs the score_mod --
# and an indexing bug there would mask the wrong scores without failing
# anything. Checked against the CPU reference path, which the tests above tie
# to "each document alone". Production head width (64); boundaries on, beside
# and across flex's 128-wide blocks, different in each row.

GPU_SEQ = 256
GPU_DIMS = dict(dmodel=256, q_heads=4, kv_heads=2, seq_len=GPU_SEQ, vocab_size=VOCAB)
GPU_EOS_AT = [[39, 40, 169], [127, 128, 200]]


def _gpu_case(attention):
    torch.manual_seed(0)
    layer = build_attention(
        0,
        attention={**attention, "document_mask": True},
        n_blocks=1,
        eos_token_id=EOS,
        **GPU_DIMS,
    )
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(1, VOCAB, (2, GPU_SEQ), generator=generator)
    for row, positions in enumerate(GPU_EOS_AT):
        ids[row, positions] = EOS
    x = torch.randn(2, GPU_SEQ, GPU_DIMS["dmodel"], generator=generator)
    return layer, x, ids


GPU_LAYERS = [
    {"type": "gqa"},
    {"type": "ame", "ame": {"shift": True}},
    {"type": "ame", "ame": {"shift": True, "sink": False}},
]


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("attention", GPU_LAYERS)
def test_on_cuda_flex_matches_the_cpu_reference_forward_and_backward(
    attention, compiled
):
    """compiled=False: the layer eager, flex through src.core.ame's compiled
    kernel. compiled=True: the layer inside torch.compile, as
    compile_blocks_full runs it, flex lowered in the surrounding graph."""
    import copy

    layer, x, ids = _gpu_case(attention)
    reference = layer(x, ids)  # CPU: the reference path
    reference.square().mean().backward()
    expected = {n: p.grad.clone() for n, p in layer.named_parameters()}

    gpu = copy.deepcopy(layer).cuda()
    gpu.zero_grad(set_to_none=True)
    run = torch.compile(gpu) if compiled else gpu
    out = run(x.cuda(), ids.cuda())
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-4, atol=1e-4)
    out.square().mean().backward()
    for name, p in gpu.named_parameters():
        torch.testing.assert_close(
            p.grad.cpu(), expected[name], rtol=1e-3, atol=1e-5, msg=name
        )


@pytest.mark.gpu
@pytest.mark.parametrize("attention", GPU_LAYERS)
def test_on_cuda_bf16_autocast_matches_fp32_within_bf16_resolution(attention):
    """The production precision: autocast bf16, compiled."""
    layer, x, ids = _gpu_case(attention)
    layer = layer.cuda()
    x, ids = x.cuda(), ids.cuda()
    fp32 = layer(x, ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = torch.compile(layer)(x, ids).float()
    relative = ((fp32 - bf16).abs().max() / fp32.abs().max()).item()
    assert relative < 0.05, f"bf16 diverges from fp32 by {relative:.2e}"
