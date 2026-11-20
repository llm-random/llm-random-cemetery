"""Rebuttal E4 helpers (scripts/rebuttal/e4_knockout.py) on a tiny PC model, CPU.

At the one-hot init no projection mass sits on excluded channels; after a perturbation
some does; the knock-out removes exactly that and leaves retained entries untouched.
"""

import importlib.util
import os
import tempfile
import types

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "e4_knockout", os.path.join(REPO, "scripts", "rebuttal", "e4_knockout.py")
)
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)


def test_knockout_removes_exactly_the_excluded_mass():
    with initialize_config_dir(
        config_dir=os.path.join(REPO, "configs", "_model", "llama"), version_base=None
    ):
        cfg = compose(config_name="projected_mem_eff_tiny")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "imp.pt")
        g = torch.Generator().manual_seed(0)
        torch.save(
            {
                "dmodel_importances": torch.rand(cfg.common.base_dmodel, generator=g),
                "dff_importances": torch.rand(
                    cfg.common.n_blocks, cfg.common.base_dff, generator=g
                ),
            },
            path,
        )
        model = instantiate(
            cfg.model,
            path_to_importances=path,
            adjust_grad_norm=False,
            cpu_offload_projections=False,
            block_gpu_compute=True,
            _convert_="all",
        )
        model.projections.init_projection_weights(path)
        fake_cfg = types.SimpleNamespace(
            projected_compression=types.SimpleNamespace(path_to_importances=path)
        )
        keep_d, keep_f = e4.keep_vectors(fake_cfg, model)

    sides = e4.projection_sides(model, keep_d, keep_f)
    # per block: q, k, v (in), o (out), ff_pre, ff_gate, ff_post (in and out) = 10
    assert len(sides) == 1 + cfg.common.n_blocks * 10 + 1
    assert all(v == 0.0 for v in e4.excluded_mass(sides).values())

    with torch.no_grad():
        for _, p, _, _ in sides:
            p.add_(0.01 * torch.randn_like(p))
    before = {n: p.detach().clone() for n, p, _, _ in sides}
    assert all(v > 0.0 for v in e4.excluded_mass(sides).values())

    with torch.no_grad():
        for _, p, kind, keep in sides:
            p.mul_(e4.keep_matrix(p, kind, keep))
    assert all(v == 0.0 for v in e4.excluded_mass(sides).values())
    for n, p, kind, keep in sides:
        k = (keep[:, None] if kind == "rows" else keep[None, :]).expand(p.shape)
        assert torch.equal(p.detach()[k], before[n][k]), n
