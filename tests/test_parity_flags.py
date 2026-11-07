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


def test_the_fp32_prep_switch_is_off_by_default_and_reaches_every_layer():
    from src.core.ame import AMEAttention

    assert AMEAttention.parity_prep_fp32 is False
    model = build_model(dtype=torch.float32)
    build_memory_parallel_controller(_cfg(), model)
    layers = [m for m in model.modules() if isinstance(m, AMEAttention)]
    assert layers and not any(m.parity_prep_fp32 for m in layers)
    model = build_model(dtype=torch.float32)
    build_memory_parallel_controller(_cfg(parity_prep_fp32=True), model)
    assert all(m.parity_prep_fp32 for m in model.modules() if isinstance(m, AMEAttention))


def test_the_fp32_prep_switch_is_a_no_op_on_fp32_banks():
    # On CPU the compute copy is fp32 already, so the cast changes nothing.
    _, grads, main = _step(_cfg())
    _, prep_grads, prep_main = _step(_cfg(parity_prep_fp32=True))
    for name in grads:
        assert torch.equal(grads[name], prep_grads[name]), name
    for a, b in zip(main, prep_main):
        assert torch.equal(a, b)


# ---- documents/notes/ame_bf16_scalars_plan.md, Step 1: the fp32 attention flags ----

import pytest  # noqa: E402

from main import apply_parity_attention_flags  # noqa: E402


def _dense_layer(seed=0):
    from src.core.attention import build_attention

    torch.manual_seed(seed)
    return build_attention(
        0, attention={"type": "gqa"}, n_blocks=1, dmodel=128, q_heads=2, kv_heads=1, seq_len=64
    )


def _ame_layer(seed=0):
    from src.core.ame import AMEAttention

    torch.manual_seed(seed)
    return AMEAttention(128, 2, 2, seq_len=64, vocab_size=VOCAB, backend="auto")


def test_the_fp32_attention_flags_are_off_by_default():
    from src.core.ame import AMEAttention
    from src.core.model import RoPEAttention

    assert AMEAttention.parity_attention_fp32 is False
    assert RoPEAttention.parity_attention_fp32 is False
    model = torch.nn.ModuleList([_ame_layer(), _dense_layer()])
    apply_parity_attention_flags(OmegaConf.create({}), model)
    assert all("parity_attention_fp32" not in vars(m) for m in model)


def test_the_fp32_attention_flags_reach_their_layers_and_refuse_the_wrong_model():
    model = torch.nn.ModuleList([_ame_layer(), _dense_layer()])
    apply_parity_attention_flags(
        OmegaConf.create({"parity_attention_fp32": True, "parity_dense_attention_fp32": True}), model
    )
    assert model[0].parity_attention_fp32 and model[1].parity_attention_fp32
    with pytest.raises(ValueError):
        apply_parity_attention_flags(
            OmegaConf.create({"parity_attention_fp32": True}), torch.nn.ModuleList([_dense_layer()])
        )


@pytest.mark.parametrize("which", ["ame", "dense"])
def test_the_fp32_attention_flags_change_nothing_on_an_fp32_cpu_model(which):
    # On CPU there is no CUDA autocast, so the flag's casts are no-ops: the
    # outputs and gradients must be bit-identical with it on and off.
    ids = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(3))
    results = []
    for flag in (False, True):
        layer = _ame_layer() if which == "ame" else _dense_layer()
        layer.parity_attention_fp32 = flag
        x = torch.randn(2, 64, 128, generator=torch.Generator().manual_seed(4), requires_grad=True)
        out = layer(x, ids) if which == "ame" else layer(x)
        out.square().sum().backward()
        grads = [p.grad.clone() for p in layer.parameters() if p.grad is not None]
        results.append((out.detach(), x.grad.clone(), grads))
    (o0, x0, g0), (o1, x1, g1) = results
    assert torch.equal(o0, o1) and torch.equal(x0, x1)
    assert len(g0) == len(g1) and all(torch.equal(a, b) for a, b in zip(g0, g1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_cuda_the_ame_fp32_path_feeds_the_kernel_fp32_and_matches_fp64():
    layer = _ame_layer().cuda()
    layer.parity_attention_fp32 = True
    seen = {}
    original = layer._attend_flex

    def spy(q, k, v):
        seen["dtypes"] = (q.dtype, k.dtype, v.dtype, torch.is_autocast_enabled("cuda"))
        return original(q, k, v)

    layer._attend_flex = spy
    ids = torch.randint(0, VOCAB, (2, 64), device="cuda")
    x = torch.randn(2, 64, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        layer(x, ids)
    assert seen["dtypes"] == (torch.float32, torch.float32, torch.float32, False), seen
    # The fp32 kernel itself: flex in fp32 against the reference path in fp64.
    del layer._attend_flex
    g = torch.Generator(device="cuda").manual_seed(5)
    q, k, v = (torch.randn(2, 2, 64, 64, device="cuda", generator=g) for _ in range(3))
    out32 = layer._attend_flex(q, k, v)
    ref = layer.double()._attend_reference(q.double(), k.double(), v.double())
    err = ((out32.double() - ref).norm() / ref.norm()).item()
    assert err < 1e-5, f"fp32 flex is {err:.2e} from fp64: not IEEE fp32 (TF32 or bf16 inside?)"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_on_cuda_the_dense_fp32_path_runs_sdpa_in_fp32_and_matches_fp64():
    import src.core.model as model_module

    layer = _dense_layer().cuda()
    layer.parity_attention_fp32 = True
    seen = {}
    original = model_module.attention_mechanism

    def spy(query, key, value, causal):
        seen["dtypes"] = (query.dtype, key.dtype, value.dtype, torch.is_autocast_enabled("cuda"))
        out = original(query=query, key=key, value=value, causal=causal)
        seen["out"] = out.dtype
        ref = original(query=query.double(), key=key.double(), value=value.double(), causal=causal)
        seen["err"] = ((out.double() - ref).norm() / ref.norm()).item()
        return out

    model_module.attention_mechanism = spy
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            layer(torch.randn(2, 64, 128, device="cuda"))
    finally:
        model_module.attention_mechanism = original
    assert seen["dtypes"] == (torch.float32, torch.float32, torch.float32, False), seen
    assert seen["out"] == torch.float32
    assert seen["err"] < 1e-5, f"fp32 SDPA is {seen['err']:.2e} from fp64: not IEEE fp32"


# ---- the same plan: flex kernel options for the fp32 kernel (AME-F's speed) ----

# The options the AME-F runs use (configs/parity/fe_ame_f*.yaml): tf32x3 dots
# (three TF32 products per fp32 one, on the tensor cores) and a 32x128 fp32
# backward tile instead of flex's 16x16, one-stage default.
AME_F_OPTIONS = {"FLOAT32_PRECISION": "tf32x3", "bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 128,
                 "bwd_BLOCK_M2": 128, "bwd_BLOCK_N2": 32, "bwd_num_warps": 8, "bwd_num_stages": 2}


def _ame_flex_layer(seed=0):
    from src.core.ame import AMEAttention

    torch.manual_seed(seed)
    return AMEAttention(128, 2, 2, seq_len=64, vocab_size=VOCAB, backend="flex")


def _flex_kwargs_seen(layer, ids, x):
    """The kwargs AMEAttention hands `_flex_attention`, on CPU (uncompiled flex)."""
    import src.core.ame as ame

    seen = {}
    original = ame._flex_attention

    def spy(q, k, v, **kwargs):
        seen.update(kwargs)
        kwargs.pop("kernel_options", None)  # CPU's eager flex has no Triton kernel to tune
        return original(q, k, v, **kwargs)

    ame._flex_attention = spy
    try:
        layer(x, ids)
    finally:
        ame._flex_attention = original
    return seen


def test_the_flex_kernel_options_are_unset_by_default_and_the_call_passes_none():
    from src.core.ame import AMEAttention

    assert AMEAttention.parity_flex_kernel_options is None
    layer = _ame_flex_layer()
    apply_parity_attention_flags(OmegaConf.create({}), torch.nn.ModuleList([layer]))
    assert "parity_flex_kernel_options" not in vars(layer)
    ids = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(3))
    seen = _flex_kwargs_seen(layer, ids, torch.randn(2, 64, 128))
    assert "kernel_options" not in seen and "score_mod" in seen, sorted(seen)


def test_the_flex_kernel_options_need_the_fp32_flag_and_reach_every_ame_layer():
    model = torch.nn.ModuleList([_ame_layer(), _ame_layer(seed=1), _dense_layer()])
    apply_parity_attention_flags(
        OmegaConf.create({"parity_attention_fp32": True, "parity_flex_kernel_options": AME_F_OPTIONS}), model
    )
    expected = {**AME_F_OPTIONS, "FLOAT32_PRECISION": "'tf32x3'"}  # a Triton literal
    assert model[0].parity_flex_kernel_options == expected
    assert model[1].parity_flex_kernel_options == expected
    assert "parity_flex_kernel_options" not in vars(model[2])
    layer = _ame_flex_layer()
    layer.parity_flex_kernel_options = expected
    ids = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(3))
    assert _flex_kwargs_seen(layer, ids, torch.randn(2, 64, 128))["kernel_options"] == expected
    with pytest.raises(ValueError):
        apply_parity_attention_flags(
            OmegaConf.create({"parity_flex_kernel_options": AME_F_OPTIONS}), torch.nn.ModuleList([_ame_layer()])
        )


@pytest.fixture
def fresh_dynamo():
    """These calls compile AME's module-level `_COMPILED_FLEX` at new dtypes, shapes
    and kernel options. Left in its cache they count toward dynamo's recompile
    limit (8), past which that function silently runs eager -- the unfused,
    materializing flex that test_ame.py's memory regression then catches (job
    36462). Start and leave the cache empty."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seq_len", [64, 2048])
def test_on_cuda_the_ame_f_kernel_options_match_fp64_forward_and_backward(seq_len, fresh_dynamo):
    from src.core.ame import AMEAttention

    torch.manual_seed(0)
    layer = AMEAttention(128, 2, 2, seq_len=seq_len, vocab_size=VOCAB, backend="auto").cuda()
    layer.parity_attention_fp32 = True
    layer.parity_flex_kernel_options = {**AME_F_OPTIONS, "FLOAT32_PRECISION": "'tf32x3'"}
    g = torch.Generator(device="cuda").manual_seed(5)
    q = F.normalize(torch.randn(2, 2, seq_len, 64, device="cuda", generator=g), dim=-1) * 10
    k = F.normalize(torch.randn(2, 2, seq_len, 64, device="cuda", generator=g), dim=-1)
    v = torch.randn(2, 2, seq_len, 64, device="cuda", generator=g)
    go = torch.randn(2, 2, seq_len, 64, device="cuda", generator=g)
    q32, k32, v32 = (t.clone().requires_grad_() for t in (q, k, v))
    out32 = layer._attend_flex(q32, k32, v32)
    grads32 = torch.autograd.grad(out32, (q32, k32, v32), go)
    ref_layer = AMEAttention(128, 2, 2, seq_len=seq_len, vocab_size=VOCAB, backend="auto").cuda().double()
    ref_layer.load_state_dict(layer.state_dict())
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    ref = ref_layer._attend_reference(q64, k64, v64)
    grads64 = torch.autograd.grad(ref, (q64, k64, v64), go.double())

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm()).item()

    errs = {"out": rel(out32, ref), **{n: rel(a, b) for n, a, b in zip(("dq", "dk", "dv"), grads32, grads64)}}
    assert all(e < 1e-5 for e in errs.values()), f"tf32x3 flex is not fp32-accurate: {errs}"
