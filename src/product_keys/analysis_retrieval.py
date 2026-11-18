"""Does PK Attention retrieve the tokens dense attention would pick? (thesis Method 4.1 note)

For every PK attention layer and head, and every query t of a few masked pre-training sequences:
  * dense logits D[t,s] = <q_t, k_s> (after RoPE, no normalisation), over all S tokens
  * the k cells PK Attention retrieves, R_t (exact top-k of the cell score over the full grid, which is
    what the two-stage top-k returns; checked against the two-stage selection)
  * recall: is argmax_s D[t,s] in R_t, and |top-k(D[t]) ∩ R_t| / k  (random: k / S)
  * decomposition of the cell score (Eq. pka-interference, with the RMSNorm weights):
        cell[t,(i,j)] = tau * ( <q~1_t, k~1_i> + <q~2_t, k~2_j> ) = own[t,(i,j)] + interference
        own[t,s] = tau * ( <q~1_t * g_k1, k_{s,1}> / rms(K1_row(s)) + <q~2_t * g_k2, k_{s,2}> / rms(K2_col(s)) )
    i.e. the token's own (normalisation-weighted) dense logit, and the 2(m-1) terms of the other tokens
    in its row and column. Recall is also reported with `own` as the ranking.

Usage (1 GPU):
  python -m src.product_keys.analysis_retrieval --ckpt <dcp step dir> --n-seqs 16 --out out.json
Local smoke test (random weights, random tokens, short sequences):
  python -m src.product_keys.analysis_retrieval --smoke
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from src.product_keys.model import RoPEProductKeysEncoderAttentionOptimized as PKA


def rms(x, eps):
    return torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def norm_parts(norm, x):
    """RMSNorm(x) = x / rms(x) * g. Returns (1/rms(x), g); g = 1 for Identity (qk_norm=False)."""
    if isinstance(norm, torch.nn.RMSNorm):
        eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
        g = norm.weight if norm.weight is not None else torch.ones(x.shape[-1], device=x.device)
        return 1.0 / rms(x, eps), g
    return torch.ones_like(x[..., :1]), torch.ones(x.shape[-1], device=x.device)


class Stats:
    KEYS = ["n", "rec1_raw", "reck_raw", "rec1_own", "reck_own", "self_retrieved",
            "lift_own", "lift_int", "sd_own", "sd_int", "corr_cell_own", "rank_top1_raw_sum",
            "own_ret", "int_ret", "abs_own_ret", "abs_int_ret", "twostage_mismatch"]

    def __init__(self, n_layers, n_heads):
        self.d = {k: torch.zeros(n_layers, n_heads, dtype=torch.float64) for k in self.KEYS}
        self.rank_hist = torch.zeros(n_layers, n_heads, 6, dtype=torch.float64)  # rank buckets of dense top-1

    def add(self, layer, key, per_head):
        self.d[key][layer] += per_head.double().cpu()


@torch.no_grad()
def analyse_layer(mod: PKA, x, layer, stats: Stats):
    B, S, _ = x.shape
    H, d, dh, m, kk = mod.q_heads, mod.dhead, mod.dhead_half, mod.m, mod.top_k
    x = x.float()
    q = mod.rope(mod.q_proj(x).view(B, S, H, -1).transpose(1, 2))
    k = mod.rope(mod.k_proj(x).view(B, S, mod.kv_heads, -1).transpose(1, 2))
    k = k.repeat_interleave(H // mod.kv_heads, dim=1)  # same as repeat_kv
    q, k = q.float(), k.float()

    k5 = k.view(B, H, m, m, d)
    K1 = k5[..., :dh].sum(-2)  # (B,H,m,dh) row sums
    K2 = k5[..., dh:].sum(-3)  # (B,H,m,dh) column sums
    q1n = mod.q_norm1(q[..., :dh]).float()
    q2n = mod.q_norm2(q[..., dh:]).float()
    r1, g1 = norm_parts(mod.k_norm1, K1)  # r1: (B,H,m,1)
    r2, g2 = norm_parts(mod.k_norm2, K2)
    tau = mod.routing_temp.float().view(1, H, 1, 1)

    # row/column scores exactly as the module computes them
    R1 = q1n @ mod.k_norm1(K1).float().transpose(-1, -2)  # (B,H,S,m)
    C2 = q2n @ mod.k_norm2(K2).float().transpose(-1, -2)
    cell = tau * (R1[..., :, None] + C2[..., None, :]).reshape(B, H, S, S)  # cell s = i*m + j

    # own term of every token s
    row_of = torch.arange(S, device=x.device) // m
    col_of = torch.arange(S, device=x.device) % m
    own1 = ((q1n * g1) @ k[..., :dh].transpose(-1, -2)) * r1[..., 0][..., row_of][:, :, None, :]
    own2 = ((q2n * g2) @ k[..., dh:].transpose(-1, -2)) * r2[..., 0][..., col_of][:, :, None, :]
    own = tau * (own1 + own2)  # (B,H,S,S)
    # check the decomposition: summing own1 over a row gives the row score
    rec = own1.view(B, H, S, m, m).sum(-1)
    assert torch.allclose(rec, R1, atol=1e-3 * R1.abs().max().item() + 1e-4), (rec - R1).abs().max()
    inter = cell - own

    dense = q @ k.transpose(-1, -2)  # (B,H,S,S) raw <q_t,k_s>

    # retrieved set: exact top-k cells, and the module's two-stage selection for comparison
    ret = cell.topk(kk, dim=-1).indices  # (B,H,S,k)
    s1, i1 = R1.topk(kk, -1)
    s2, i2 = C2.topk(kk, -1)
    grid = (s1[..., :, None] + s2[..., None, :]).flatten(-2)
    sel = grid.topk(kk, -1).indices
    ret2 = i1.gather(-1, sel // kk) * m + i2.gather(-1, sel % kk)
    same = (ret.sort(-1).values == ret2.sort(-1).values).all(-1)
    in_ret = torch.zeros(B, H, S, S, dtype=torch.bool, device=x.device).scatter_(-1, ret2, True)

    def per_head(t):  # mean over batch and queries -> sum over queries, per head
        return t.float().sum(dim=(0, 2))

    stats.add(layer, "n", torch.full((H,), B * S, dtype=torch.float64))
    stats.add(layer, "twostage_mismatch", per_head(~same))
    for name, score in (("raw", dense), ("own", own)):
        top1 = score.argmax(-1, keepdim=True)
        stats.add(layer, f"rec1_{name}", per_head(in_ret.gather(-1, top1)[..., 0]))
        topk = score.topk(kk, -1).indices
        stats.add(layer, f"reck_{name}", per_head(in_ret.gather(-1, topk).float().mean(-1)))
        if name == "raw":
            # rank (0-based) of the dense top-1 token under the cell score
            ctop = cell.gather(-1, top1)
            rank = (cell > ctop).sum(-1)
            stats.add(layer, "rank_top1_raw_sum", per_head(rank))
            edges = torch.tensor([kk, 2 * kk, 4 * kk, 8 * kk, S // 4, S + 1], device=x.device)
            b = torch.bucketize(rank, edges, right=True)
            stats.rank_hist[layer] += F.one_hot(b, 6).sum(dim=(0, 2)).double().cpu()
    diag = torch.arange(S, device=x.device)
    stats.add(layer, "self_retrieved", per_head(in_ret[:, :, diag, diag]))

    # score decomposition
    ret_own = own.gather(-1, ret2)
    ret_int = inter.gather(-1, ret2)
    stats.add(layer, "own_ret", per_head(ret_own.mean(-1)))
    stats.add(layer, "int_ret", per_head(ret_int.mean(-1)))
    stats.add(layer, "abs_own_ret", per_head(ret_own.abs().mean(-1)))
    stats.add(layer, "abs_int_ret", per_head(ret_int.abs().mean(-1)))
    stats.add(layer, "lift_own", per_head(ret_own.mean(-1) - own.mean(-1)))
    stats.add(layer, "lift_int", per_head(ret_int.mean(-1) - inter.mean(-1)))
    stats.add(layer, "sd_own", per_head(own.std(-1)))
    stats.add(layer, "sd_int", per_head(inter.std(-1)))
    cc = cell - cell.mean(-1, keepdim=True)
    oc = own - own.mean(-1, keepdim=True)
    corr = (cc * oc).sum(-1) / (cc.norm(dim=-1) * oc.norm(dim=-1) + 1e-12)
    stats.add(layer, "corr_cell_own", per_head(corr))


def build(config_name, overrides):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    import resolver  # noqa: F401  registers ${eval:...}

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(cfg_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    def strip(c):  # grid keys (^name) are handled by the launcher, not hydra
        if OmegaConf.is_dict(c):
            for key in list(c.keys()):
                if str(key).startswith("^"):
                    del c[key]
                else:
                    strip(c[key])

    with open_dict(cfg):
        strip(cfg)
    return cfg, instantiate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="product_keys/rerun_pk_full")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--out", default="retrieval_analysis.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = []
    if a.smoke:
        overrides = ["common.sequence_length=64", "common.dmodel=128", "common.dff=256",
                     "common.n_blocks=2", "common.q_heads=4", "common.kv_heads=4",
                     "model.encoder.block_fn.attention_fn.top_k=4"]
    cfg, instantiate = build(a.config, overrides)
    model = instantiate(cfg.model, _convert_="all").float()

    if a.ckpt:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import DefaultLoadPlanner
        from src.core.checkpointing import check_checkpoint_model_keys

        check_checkpoint_model_keys(a.ckpt, model)  # raises unless the key sets are identical
        sd = model.state_dict()
        before = {k: v.clone() for k, v in sd.items()}
        dcp.load({"app": {"model": sd}}, checkpoint_id=a.ckpt,
                 planner=DefaultLoadPlanner(allow_partial_load=True))
        model.load_state_dict(sd)
        unchanged = [k for k in sd if torch.equal(before[k], sd[k]) and "rope" not in k]
        print(f"loaded {len(sd)} tensors, unchanged after load: {unchanged}")
        assert not unchanged, "some weights were not loaded"
    model.to(dev).eval()

    S = cfg.common.sequence_length
    g = torch.Generator().manual_seed(a.seed)
    if a.smoke:
        data = torch.randint(0, 50000, (a.n_seqs, S), generator=g)
    else:
        loader = instantiate(cfg.trainer.eval_dataloader, _convert_="all")
        data = next(iter(loader))[: a.n_seqs]
    assert data.shape[1] == S, data.shape
    # MLM masking as in training (MaskedLMTrainer, 20%, special tokens never masked)
    tr = cfg.trainer
    labels = data.clone()
    masked = torch.rand(data.shape, generator=g) < tr.masking_percentage
    for t in tr.unmaskable_special_tokens:
        masked &= data != t
    inp = data.clone()
    inp[masked] = tr.mask_token_id
    labels[~masked] = -100

    layers = [mod for mod in model.modules() if isinstance(mod, PKA)]
    H = layers[0].q_heads
    stats = Stats(len(layers), H)
    current = {}
    hooks = [mod.register_forward_pre_hook(
        lambda mod, args, kwargs, i=i: analyse_layer(mod, args[0], i, stats), with_kwargs=True)
        for i, mod in enumerate(layers)]

    losses = []
    with torch.no_grad():
        for b in range(a.n_seqs):
            x = inp[b: b + 1].to(dev)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                          labels[b].to(dev), ignore_index=-100).item())
            print(f"seq {b}: mlm loss {losses[-1]:.4f}", flush=True)
    for h in hooks:
        h.remove()

    n = stats.d["n"]
    res = {"mlm_loss_mean": sum(losses) / len(losses), "n_seqs": a.n_seqs, "S": S,
           "top_k": layers[0].top_k, "ckpt": a.ckpt, "random_reck": layers[0].top_k / S}
    for key, v in stats.d.items():
        if key == "n":
            continue
        res[key] = (v / n).tolist()
    res["rank_hist"] = (stats.rank_hist / n[..., None]).tolist()
    res["rank_hist_edges"] = ["<k", "<2k", "<4k", "<8k", "<S/4", "rest"]
    res["routing_temp"] = [mod.routing_temp.view(-1).tolist() for mod in layers]
    res["attn_temp"] = [mod.attn_temp.view(-1).tolist() for mod in layers]
    with open(a.out, "w") as f:
        json.dump(res, f)

    print(f"mean MLM loss {res['mlm_loss_mean']:.4f}  (random recall@k {res['random_reck']:.4f})")
    print("layer  rec1_raw reck_raw rec1_own reck_own  self  own_share_of_lift  sd_int/sd_own corr  mismatch")
    for li in range(len(layers)):
        def m_(k):
            return (stats.d[k][li] / n[li]).mean().item()
        lo, lin = m_("lift_own"), m_("lift_int")
        print(f"{li:5d}  {m_('rec1_raw'):.3f}    {m_('reck_raw'):.3f}    {m_('rec1_own'):.3f}    "
              f"{m_('reck_own'):.3f}   {m_('self_retrieved'):.3f}  {lo / (lo + lin + 1e-12):.3f}"
              f"              {m_('sd_int') / m_('sd_own'):.2f}        {m_('corr_cell_own'):.3f} "
              f"{m_('twostage_mismatch'):.4f}")


if __name__ == "__main__":
    main()
