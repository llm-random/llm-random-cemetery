# Task: SliceGPT baseline for the PC paper

Branch: `mas/nips` · Started: 2026-10-06 · Status: **planning (nothing implemented yet)**

## Goal

Add SliceGPT (Ashkboos et al., ICLR 2024) as a new baseline next to HP/HPD/PC/PCD/SVD:
compress Llama 1B (Llama-3.2-1B) and 8B (Llama-3.1-8B) by **50 / 70 / 90 %**, then run recovery
training (RFT) in our framework with the same data and token budgets as HP, and compare on the same metric.

**Decision (2026-10-06, Maciej): the main baseline is the HYBRID "SliceGPT + FFN pruning"**
(SliceGPT on the residual width d, HP-style channel pruning on dff). It's preferable for our comparison.
Optionally also run faithful SliceGPT (residual only) at 1–2 points so reviewers can't say we
weakened their method.

Sources (local):
- Paper: `slicegpt/slice_gpt_paper.pdf` (arXiv 2401.15024v2)
- Official repo: `slicegpt/TransformerCompression` (microsoft, HEAD `183cd92`), MIT license

## How SliceGPT works (paper)

- **Computational invariance (Thm 1):** in an RMSNorm-connected transformer, `RMSNorm(XQ)Qᵀ = RMSNorm(X)`
  for orthogonal Q, so the residual stream can be rotated by Q if the weights are rotated to match:
  `W_embd·Q`, `Qᵀ·W_in`, `W_out·Q`, `Qᵀ·W_head`.
- **LayerNorm → RMSNorm:** fold the norm scale `diag(α)` into the following linear; models that are
  already RMSNorm (Llama) only need the weight folded.
- **Per-layer Q_ℓ:** one global Q works poorly, so each block gets its own Q from PCA of the calibration
  signal (`C_ℓ = Σ XᵀX`, eigenvectors sorted by eigenvalue, fp64). Different Q per block requires an
  extra linear map **`shortcut_Q = Q_{ℓ-1}ᵀQ_ℓ`** in each residual connection (2 per layer, d×d after slicing).
- **Slicing:** keep the top-d principal directions. This deletes columns of the embedding, q/k/v and
  gate/up inputs, and rows of o_proj/down_proj outputs; the lm_head is sliced optionally.
- **Only the residual width d shrinks.** FFN hidden dim (dff) and attention inner dims (heads × head_dim)
  stay full.
- Paper only goes up to 30 % slicing. Llama-2-7B one-shot at 30 %: avg zero-shot 69.0 → 51.5. RFT uses LoRA
  (r=32, α=10). Calibration ablation: ≥128 samples, longer sequences are better; main runs use 1024 × 2048.
  Fp32 PCA is noticeably worse than fp64 (App. A.2).
- Conclusion explicitly lists combining with structural pruning (LLM-Pruner) as future work, which
  motivates our hybrid.

## Why dff can't be sliced by rotation (supports the hybrid)

The invariance holds only for the residual stream (through RMSNorm). dff sits behind elementwise
ops (`silu(gate(x)) * up(x)`), and rotations don't commute with elementwise functions. Attention inner
dims sit behind RoPE/softmax, same problem. Only **permutations / channel selection** commute with
elementwise ops, so dff can only be reduced by pruning, i.e. exactly HP's dff branch.

## Code map (official repo, `src/slicegpt/`)

| File | What it does |
|---|---|
| `rotate.py` | `rotate_and_slice_sequential` (main loop), `pca_calc` (fp64 cov + 1 % mean-diag damping, `eigh`), `rotate_*`/`slice_*` per weight, `random_orthogonal_upper_left` |
| `layernorm_fusion.py` | `fuse_ln_linear` (fold norm weight into next linear), `fuse_modules`, `replace_layers` |
| `modules.py` | `RMSN` — weightless RMSNorm, divides by `mean_dim = hidden_size` (ORIGINAL D) |
| `adapters/llama_adapter.py` | `CompressedLlamaDecoderLayer` (residual `x @ shortcut_Q + f(norm(x))`), module mapping |
| `slicing_scheduler.py` | `ConstSlicingScheduler` (same d everywhere), per-layer schedulers (worse on Llama-2 per App. A.4) |
| `model_utils.py` | `get_layer0_inputs` (Catcher), `get_signals` (hook on 2nd norm input + layer output) |
| `data_utils.py` | wikitext2/ptb/c4/alpaca calibration loaders (packed to max_seqlen) |
| `experiments/run_slicegpt.py` | CLI: sparsity → `d = int((1-s)·D)` rounded down to `round_interval` |

Main loop per layer: rotate+slice attn inputs with Q_prev → re-run layer to get pre-MLP-norm signal →
PCA → Q_mid → set `attn_shortcut_Q`, rotate+slice o_proj and MLP inputs → re-run → PCA on layer
output → Q_next → set `mlp_shortcut_Q`, rotate+slice down_proj. Activations are kept on CPU; one layer
on GPU at a time.

## Gotchas found in their code (must handle in our port)

1. **Can't run on our models as-is.** Pinned `transformers==4.41` can't load Llama-3.1/3.2 (`llama3` rope
   type needs ≥4.43). Our pixi env has **4.55** (torch 2.7.1), where their `CompressedLlamaDecoderLayer.forward`
   breaks (attn now returns 2 values, rotary passed as `position_embeddings`). The adapter name check only
   accepts `meta-llama/Llama-2*` / `meta-llama/Meta-Llama-3*`.
2. **RMSN divides by the original D**, not the sliced d. Our `src.core.model.RMSNorm` divides by
   `normalized_shape` = d. Port must set norm weight = **√(D/d)** after slicing (exact equivalence up to eps),
   otherwise activations get rescaled (×3.4 at d=178).
3. **Embedding mean-centering is unconditional** in `fuse_modules` (meant for LayerNorm models). For RMSNorm
   Llama it changes the function. Measure loss after fusion (their `--eval-fused-model` exists for this)
   and probably skip it.
4. **Defaults vs paper:** CLI default `final_orientation="random"` (random rotation inside the kept subspace;
   function-equivalent but matters for Adam RFT / fp16), paper describes PCA. `round_interval=8`.
   `do_slice_head=False` by default, and model fp16 + fp64 PCA. Default calibration is 128 × 2048 wikitext2,
   but the paper uses 1024 × 2048.
5. **The head must be sliced** (`do_slice_head=True`): an unsliced V×D head alone is 267M (1B) / 540M (8B),
   more than the 90 % budget (189M for 1B).
6. **Llama-3.2-1B has tied embeddings.** SliceGPT unties them (different Q at input and output). nano
   already trains untied (`copy_llama_model_weights_from_HF` loads lm_head as a separate param), so it's
   consistent with HP/PC.

## Our framework (relevant facts)

- HP pipeline = `checkpoint.load.type: huggingface` → `copy_llama_model_weights_from_HF` → `apply_functions`
  (`src.projected_compression.pruning.prune`, runs on each rank before FSDP) → standard trainer.
- nano `TransformerBlock` = `Residual(norm, attention)` + `Residual(norm, LlamaFeedForward)`; Llama FF
  modules: `ff_pre_act` (up), `gate`, `ff_post_act` (down). Attention: `q_proj/k_proj/v_proj/o_proj`.
  The config has separate `dmodel`, `datt`, `dff`, so a sliced model is `dmodel=d`, `datt=original`.
- HP dims (from `configs/pc_project_mas/llama_{1B,8B}_prune.yaml`):
  1B 50/70/90 % = d/dff 1344/5376, 960/3840, 448/1728; 8B = 3072/9216, 2160/6480, 1056/3168.
- Precedent: the SVD baseline in the paper = SVD-LLM compression + **full-parameter** recovery training
  (instead of LoRA). Do the same for SliceGPT.

## Parameter matching (computed 2026-10-06, untied emb+head as in nano, shortcut_Q counted)

**Hybrid (MAIN): same dff as HP, d chosen to match HP total params**

| | 50 % | 70 % | 90 % |
|---|---|---|---|
| 1B HP d / dff | 1344 / 5376 (802M) | 960 / 3840 (502M) | 448 / 1728 (189M) |
| 1B hybrid d / dff | **1256** / 5376 (800M) | **912** / 3840 (503M) | **432** / 1728 (188M) |
| 8B HP d / dff | 3072 / 9216 (4513M) | 2160 / 6480 (2606M) | 1056 / 3168 (938M) |
| 8B hybrid d / dff | **2744** / 9216 (4513M) | **1960** / 6480 (2610M) | **984** / 3168 (936M) |

The hybrid differs from HP only in how the residual stream is compressed: PCA rotation + slicing
(+ shortcut_Q) instead of channel selection. That's a very clean comparison.

**Faithful SliceGPT (optional): dff full, d matched on total params**

| | 50 % | 70 % | 90 % |
|---|---|---|---|
| 1B d (dff 8192) | 1048 (−49 % of D) | 664 (−68 %) | 256 (−88 %) |
| 8B d (dff 14336) | 2152 (−48 %) | 1272 (−69 %) | 472 (−89 %) |

- shortcut_Q overhead: up to 35M (1B 50 %) / 296M (8B 50 %).
- Params ≠ FLOPs: faithful SliceGPT spends less on embeddings → ~1.7× HP's non-embedding params at 1B 90 %.
  Matching non-embedding params instead would give 1B d=904/520/152, 8B d=2032/1152/384.
- Param counting convention for "50/70/90 %" in the paper still to confirm. HP models keep 0.535/0.335/0.126
  of the 1B params counted untied (0.509/0.306/0.106 counted tied).

## Plan (proposed, not started)

**Approach:** port SliceGPT into nano (copy their algorithm, keep MIT headers) rather than running their HF
code and converting. Porting to transformers 4.55 is needed anyway; the nano forward pass is simpler
(packed sequences, no masks/kwargs); and it mirrors how HP plugs in.

### Phase 1 — port (1 GPU)
- `src/projected_compression/slicegpt/`:
  - `rotate.py` — copy `pca_calc`, `rotate_*`/`slice_*`; nano version of `rotate_and_slice_sequential`
    (layer-0 input = `model.embedding(x)`; pre-MLP-norm signal via forward hook on `ff_layer.norm`).
  - `fusion.py` — `fuse_ln_linear` into q/k/v, gate/ff_pre_act, head.linear; norm weights → √(D/d) after slicing.
  - `nano_adapter.py` — module mapping (embedding / q,k,v / o / ff_pre_act,gate / ff_post_act /
    attention_layer.norm, ff_layer.norm, head.norm / head.linear).
  - **Hybrid step:** after slicing d, prune dff with our existing activation importances (same dff indices as
    HP; residual rotation doesn't change FFN hidden activations, so the importances stay valid). Reuse `pruning.py`'s dff branch.
- Model: `ShortcutResidual` (`x @ shortcut_Q + layer(norm(x))`) + `_model/llama_3_1_sliced.yaml`, `llama_3_8_sliced.yaml`.
- Offline slicing script/config: HF Llama → nano (fp32), calibrate on **FineWeb-Edu 1024 × 2048**, slice (+ prune dff),
  save `slicegpt_hyb_{1b,8b}_{50,70,90}.pt` + slicing JSON to `/storage_nvme_4/nano/j321m/pc/`.
- Loader: `checkpoint.load.type: sliced_nano` branch in `main.py` (like `finalized_pc`).

### Phase 2 — validation (gates everything)
1. Invariance: fuse + rotate with d=D (no slicing) → FineWeb loss unchanged (fp32, ~1e-4); also check the
   embedding-centering effect.
2. Reference cross-check: their unmodified repo in a separate env (transformers 4.41) on `meta-llama/Meta-Llama-3-8B`
   (supported), 25 % slicing, wikitext2, vs our port on the same model (rope scaling off) → WikiText2 ppl within ~1 %.
3. One-shot (0-token) loss for every sliced model (sanity: monotone in compression; compare with HP one-shot).

### Phase 3 — recovery training (mirror HP/HPD exactly)
- `configs/pc_project_mas/llama_{1B,8B}_slicegpt_hyb.yaml` (trainer `llama`, like HP) and
  `..._slicegpt_hyb_distillation.yaml` (distillation trainer, like HPD).
- Same as HP: FineWeb-Edu, seed 123, seq 1024 × batch 512 (accum 8), `^n_steps: [2048,4096,8192,12288]`, wd 0.1, clip 1.0.
- LR sweep first (2^-10..2^-13) at 1BT per ratio (HP's best LR differed per ratio).
- Metric: final `steps/100/train/loss`; wandb `ideas_cv/pc_v2`; run on entropy (pipeline validated by the PCD repro).
- Optional faithful-SliceGPT points: 1B 50 % and 90 %, 2BT.

### Phase 4 — downstream eval
The sliced model isn't a standard HF Llama (shortcut_Q). Either export to their format (needs their adapter ported to
4.55) or write an lm-eval wrapper for nano models. Decide after Phase 3.

### Compute (rough)
Slicing: minutes (1B), ~1 h (8B, paper: 44 min for 7B). RFT dominates: 1B at 2B tokens ≈ 4 h on 4×H100
(from the PCD repro), so the full 1B grid (2 variants × 3 ratios × 4 budgets ≈ 80B tokens) ≈ 150 node-hours; 8B ≈ 5–8×.
Suggested order: 1B all ratios @ 2BT → rest of 1B budgets → 8B.

## Open decisions
- [x] Hybrid as main baseline (Maciej, 2026-10-06)
- [ ] Shortcut_Q trainable during RFT? (proposed: yes, full-param like SVD baseline)
- [ ] Orientation: `random` (repo default) vs `pca` (paper)
- [ ] Calibration: FineWeb-Edu 1024 × 2048 (proposed) vs wikitext2
- [ ] Param-counting convention for 50/70/90 % (untied as in nano, or tied)
- [ ] Run faithful SliceGPT at a few points too?
- [ ] Scope / order given compute
- [ ] Paper label, e.g. "SliceGPT + FFN pruning", citing their conclusion

## Log
- 2026-10-06: read paper + official code; findings, param matching and plan above. Maciej chose the hybrid. Nothing implemented.
