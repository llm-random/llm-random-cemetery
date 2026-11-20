"""Discarded-block control (rebuttal E3): projected_compression.mask_discarded_source.

W' = P1^T P1 W P2 P2^T: every frozen source matrix is cut to the block the one-hot
projection init selects. On a tiny PC model, CPU, single process, this checks:

- W_c and the logits at step 0 are bit-identical to unmasked PC (the run starts like PC
  and HP);
- the projections get exactly zero gradient on every entry that reads a discarded
  channel, while unmasked PC's gradient there is non-zero (so the test can fail);
- the gradient on retained entries is bit-identical to unmasked PC's;
- under AdamW with weight decay the discarded entries stay exactly zero for several steps;
- with the source model kept as the distillation teacher (masked student copies), the
  teacher's logits are unchanged and the student matches the in-place variant exactly;
- a replaced source parameter fails loudly instead of training on the unmasked weight.
"""

import copy
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the repo's custom OmegaConf resolvers)
from src.projected_compression.initialization import apply_discarded_block_mask

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_importances(path, base_dmodel, base_dff, n_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    torch.save(
        {
            "dmodel_importances": torch.rand(base_dmodel, generator=g),
            "dff_importances": torch.rand(n_blocks, base_dff, generator=g),
        },
        path,
    )


@pytest.fixture(scope="module")
def base_model():
    """A tiny PC model built the way create_model builds it, minus FSDP and CUDA."""
    with initialize_config_dir(
        config_dir=os.path.join(REPO, "configs", "_model", "llama"), version_base=None
    ):
        cfg = compose(config_name="projected_mem_eff_tiny")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "importances.pt")
        _write_importances(
            path, cfg.common.base_dmodel, cfg.common.base_dff, cfg.common.n_blocks
        )
        torch.manual_seed(0)
        model = instantiate(
            cfg.model,
            path_to_importances=path,
            adjust_grad_norm=False,
            cpu_offload_projections=False,
            block_gpu_compute=True,
            _convert_="all",
        )
        model.target_model.embedding = None
        for p in model.source_model.parameters():
            # Random, not tiny: the default init is small enough to hide real effects.
            with torch.no_grad():
                p.normal_(0.0, 0.5)
            p.requires_grad = False
        model.projections.init_projection_weights(path)
    return model


def _batch(seed=1, bsz=2, seq=16, vocab=128256):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (bsz, seq + 1), generator=g)


def _step_grads(model, ids):
    """One PC step up to the projection gradients: build W_c, forward, backward, pass
    dL/dW_c to the projections. Returns (W_c snapshot, logits, P grads by name)."""
    for p in model.parameters():
        p.grad = None
    model.prepare_compressed_weights()
    wc = {n: p.detach().clone() for n, p in model.target_model.named_parameters()}
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1).float(), ids[:, 1:].flatten())
    loss.backward()
    model.pass_gradient_to_projections(None, None, None, shared_gradient_norms=False)
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.projections.named_parameters()
        if p.grad is not None
    }
    return wc, logits.detach(), grads


def _keep_masks(model):
    """name -> 0/1 tensor marking the P entries that read retained channels only."""
    proj = model.projections
    keep = {}
    for n, p in proj.named_parameters():
        if n.endswith("projection_in_weight"):  # (base_in, target_in): rows = base
            sel = (p.detach() != 0).any(dim=1)
            keep[n] = sel[:, None].expand_as(p).float()
        elif n.endswith("projection_out_weight"):  # (target_out, base_out): cols = base
            sel = (p.detach() != 0).any(dim=0)
            keep[n] = sel[None, :].expand_as(p).float()
        elif n == "embedding":  # (target_dmodel, base_dmodel)
            sel = (p.detach() != 0).any(dim=0)
            keep[n] = sel[None, :].expand_as(p).float()
    return keep


def _make(base_model, kind):
    """A fresh variant. Copy-mode models are keyed on parameter identity, so they are
    built, never deep-copied (see test_a_replaced_source_parameter_fails_loudly)."""
    model = copy.deepcopy(base_model)
    if kind == "in_place":
        apply_discarded_block_mask(model, keep_source_unmasked=False)
    elif kind == "copies":
        apply_discarded_block_mask(model, keep_source_unmasked=True)
    else:
        assert kind == "unmasked"
    return model


def test_start_is_bit_identical_to_pc(base_model):
    ids = _batch()
    wc_a, logits_a, _ = _step_grads(_make(base_model, "unmasked"), ids)
    for kind in ("in_place", "copies"):
        wc_b, logits_b, _ = _step_grads(_make(base_model, kind), ids)
        for n in wc_a:
            assert torch.equal(wc_a[n], wc_b[n]), f"W_c differs at init: {n}"
        assert torch.equal(logits_a, logits_b)


def test_discarded_entries_get_exactly_zero_gradient(base_model):
    ids = _batch()
    keep = _keep_masks(base_model)
    assert keep, "no projection matrices found"
    _, _, g_unmasked = _step_grads(_make(base_model, "unmasked"), ids)
    _, _, g_in_place = _step_grads(_make(base_model, "in_place"), ids)
    _, _, g_copies = _step_grads(_make(base_model, "copies"), ids)

    some_leak = False
    for n, k in keep.items():
        discarded = k == 0
        assert discarded.any(), f"{n}: nothing discarded, the tiny config is wrong"
        assert torch.all(g_in_place[n][discarded] == 0), f"{n}: gradient leaks"
        some_leak |= bool((g_unmasked[n][discarded] != 0).any())
        # Retained entries: identical to unmasked PC, bit for bit.
        assert torch.equal(g_in_place[n][~discarded], g_unmasked[n][~discarded]), n
        # Teacher-safe variant gives the same student gradients as the in-place one.
        assert torch.equal(g_copies[n], g_in_place[n]), n
    assert some_leak, "unmasked PC shows no gradient on discarded channels either"

    # Auxiliary weights (W_r) are untouched by the control.
    for n in g_unmasked:
        if "auxiliary" in n:
            assert torch.equal(g_in_place[n], g_unmasked[n]), n


@pytest.mark.parametrize("kind", ["in_place", "copies"])
def test_discarded_entries_stay_zero_under_adamw(base_model, kind):
    unmasked = _make(base_model, "unmasked")
    model = _make(base_model, kind)
    keep = _keep_masks(model)
    opts = [
        torch.optim.AdamW(m.projections.parameters(), lr=1e-2, weight_decay=0.1)
        for m in (model, unmasked)
    ]
    for step in range(3):
        for m, opt in zip((model, unmasked), opts):
            _step_grads(m, _batch(seed=10 + step))
            opt.step()
            opt.zero_grad()
    params = dict(model.projections.named_parameters())
    params_unmasked = dict(unmasked.projections.named_parameters())
    moved = False
    for n, k in keep.items():
        assert torch.all(params[n].detach()[k == 0] == 0), f"{n} left the retained block"
        moved |= bool((params_unmasked[n].detach()[k == 0] != 0).any())
    assert moved, "unmasked PC never moved off the retained block, test is vacuous"


def test_teacher_is_untouched_when_source_is_the_teacher(base_model):
    unmasked, in_place, student_copies = (
        _make(base_model, k) for k in ("unmasked", "in_place", "copies")
    )
    ids = _batch()[:, :-1]
    with torch.no_grad():
        teacher_ref = unmasked.source_model(ids)
        teacher_copies = student_copies.source_model(ids)
        teacher_in_place = in_place.source_model(ids)
    assert torch.equal(teacher_ref, teacher_copies)
    # Masking in place does change the source model, which is why it is only used when
    # the source model is not the teacher.
    assert not torch.equal(teacher_ref, teacher_in_place)


def test_a_replaced_source_parameter_fails_loudly(base_model):
    model = copy.deepcopy(_make(base_model, "copies"))
    # deepcopy rebuilt every parameter object, so the id-keyed views no longer match.
    with pytest.raises(RuntimeError, match="no masked view"):
        model.prepare_compressed_weights()
