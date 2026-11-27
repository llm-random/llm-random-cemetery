# AME attention gates: a read gate and an output gate

**Status: implemented on `ame_gate` at `7d9af3f3` (2026-11-27), CPU-tested, GPU tests
pending.** CPU gate: 672 passed, 0 skipped, 0 failed (baseline: 580 passed, 0 skipped, 0
failed). Slow CPU tests: 9 passed, 0 failed (baseline: 7 passed). `black --check` clean.
What was built, and where it departs from the plan: [As built](#as-built-on-ame_gate). The
GPU tests are a later, separate pass ([listed below](#later-the-gpu-pass)). Branch `ame_gate`
(worktree `~/code/nano-ame-gate`, cut from `ame_main`), not merged into `ame_main`. Designed
2026-11-27 (Maciej).

Two gates in the style of the feed-forward's SwiGLU gate, `φ(W_g h) ⊙ (…)`, put on AME's
attention. Each works alone, and both together are a valid config:

- **A, the read gate:** multiplies the heads' aggregated values (their read `r_i`) just before
  the output projection `W_O`.
- **B, the output gate:** multiplies the attention module's output just before the residual add.

What AME is: [ame_implementation.md](ame_implementation.md). The standard models these gates
would extend: [standard_model_improvements.md](standard_model_improvements.md). The earlier
*bank* gates acted on Q and K and are a different mechanism
([archive/ame_bank_gates_plan.md](archive/ame_bank_gates_plan.md)).

## Decisions (Maciej, 2026-11-27)

| Knob | Decision |
|---|---|
| Gates | A and B, each switched on or off on its own |
| Gate input | `h_i = RMSNorm(x_i)`: the normalized residual stream the layer already reads (`W_V`'s input), at the query position `i` |
| Activation `φ` | a switch: `sigmoid` (Qwen's gated attention) or `silu` (exactly SwiGLU's gate). Default proposed: `sigmoid`, the better of the two at A's position ([Evidence](#evidence-qwens-gated-attention)) |
| Granularity | A: `elementwise` (default; one value per head channel) or `headwise` (one per head). B: `elementwise` (one per residual channel) |
| Parameters | matched to dense, as AME v4 is: every elementwise gate gives `dmodel/3` of FF width back ([Matching](#matching)) |
| Score table | untouched: neither gate enters `q·k` ([The score table survives](#the-score-table-survives)) |

Config keys, in `common.attention.ame` (`_AME_OPTIONS` in `src/core/attention.py`; built on
`ame_gate`):

```yaml
common:
  attention:
    type: ame
    ame:
      read_gate: none           # A: none | elementwise | headwise
      output_gate: none         # B: none | elementwise
      gate_activation: null     # sigmoid | silu; null = sigmoid when a gate is on
  dff: ...  # matched: 32 * round((26 - 2k) * H / 3), k = number of elementwise gates
```

## The AME layer today

Per layer: position `i` with token `t_i`; heads `h = 1…H`; `d = 64` (`d_head`); `d_att = H·d`,
which equals `dmodel` in the H family; `n(·)` is the ℓ2 norm; `∅` is the learned sink.

```
h_i    = RMSNorm(x_i)
q_i^h  = τ_h · n(E_Q^h[t_i])                                  a gather by token id
k_j^h  = α_h1 · n(E_K^h[t_j]) + α_h2 · n(E_K^h[t_{j−1}])       KV shift on keys
v_j^h  = β_h1 · W_V^h h_j     + β_h2 · W_V^h h_{j−1}           ... and on values
s_ij^h = q_i^h · k_j^h − m_h · (i − j)   for j ≤ i             ALiBi distance bias
s_i∅^h = q_i^h · k_∅^h                                         the sink, always visible
a_i·^h = softmax of s_i·^h over the sink and every j ≤ i
r_i^h  = a_i∅^h v_∅^h + Σ_{j≤i} a_ij^h v_j^h    ∈ ℝ^d          head h's read
r_i    = [r_i^1; …; r_i^H]                      ∈ ℝ^d_att
o_i    = W_O r_i = Σ_h W_O^h r_i^h              ∈ ℝ^dmodel     W_O^h: head h's d columns of W_O
x_i′   = x_i + o_i
```

Every weight `a_ij^h` is a function of `(t_i, t_j, t_{j−1}, i − j)`: where a head reads, and
how much, is decided by ids and distance alone. Context reaches the output only through the
values, i.e. through *what* is read at those positions. Nothing that depends on the context
decides whether a head's read is used.

## The two gates

```
φ ∈ {σ, SiLU},  SiLU(z) = z·σ(z);  no bias;  W_A and W_B are separate matrices

A, elementwise:  g_i^A = φ(W_A h_i) ∈ ℝ^d_att      W_A ∈ ℝ^(d_att × dmodel)
                 o_i   = W_O (g_i^A ⊙ r_i)  =  Σ_h W_O^h (g_i^A,h ⊙ r_i^h)
A, headwise:     g_i^A = φ(W_A h_i) ∈ ℝ^H          W_A ∈ ℝ^(H × dmodel)
                 o_i   = Σ_h g_i,h^A · W_O^h r_i^h
B:               g_i^B = φ(W_B h_i) ∈ ℝ^dmodel     W_B ∈ ℝ^(dmodel × dmodel)
                 x_i′  = x_i + g_i^B ⊙ o_i
A and B:         x_i′  = x_i + g_i^B ⊙ W_O (g_i^A ⊙ r_i)
```

In the H family `d_att = dmodel`, so the elementwise A and B are matrices of the same shape and
cost. They differ only in where they multiply:

- **A acts inside each head, before `W_O` mixes the heads.** Head h's value-to-output map, the
  fixed linear `W_O^h W_V^h`, becomes `W_O^h · diag(g_i^A,h) · W_V^h`, a map that depends on
  the query's context. Headwise, it is one context-dependent scalar per head: a head can turn
  its read off for this query.
- **B acts after the heads are summed.** Residual channel `c` receives `Σ_h (W_O^h r_i^h)_c`
  from every head, and `g_i,c^B` scales all of them alike. B can attenuate (sigmoid) or reshape
  (SiLU) the attention's write per channel, but it cannot silence one head without silencing
  every head's share of the same channels.
- **Neither changes `a_ij^h`.** Where AME reads stays a matter of ids and distance; the gates
  decide, from the context, how much of the read is used.

## Schemas

`(Σ)` is the attention-weighted sum over `j` (sink included), `(×)` an elementwise product and
`(+)` the residual add. `x` feeds both RMSNorm and the residual path.

Today:
```
  ids ──► scores (ids + distance only) ──► softmax ───┐
                                                      │ a_ij
  x ─┬─► RMSNorm ─► h ──► W_V, β shift ─► v_j ──────►(Σ)  r_i = Σ_j a_ij v_j
     │                                                │ r_i
     │                                               W_O
     │                                                │ o_i
     └──────────────────────────────────────────────►(+) ─► x_i′
```

A, the read gate:
```
  ids ──► scores (ids + distance only) ──► softmax ───┐
                                                      │ a_ij
  x ─┬─► RMSNorm ─► h ─┬─► W_V, β shift ─► v_j ─────►(Σ)  r_i = Σ_j a_ij v_j
     │                 │                              │ r_i
     │                 └─► W_A ─► φ ─► g_i^A ───────►(×)  A: read gate
     │                                                │
     │                                               W_O
     │                                                │ o_i
     └──────────────────────────────────────────────►(+) ─► x_i′
```

B, the output gate:
```
  ids ──► scores (ids + distance only) ──► softmax ───┐
                                                      │ a_ij
  x ─┬─► RMSNorm ─► h ─┬─► W_V, β shift ─► v_j ─────►(Σ)  r_i = Σ_j a_ij v_j
     │                 │                              │ r_i
     │                 │                             W_O
     │                 │                              │ o_i
     │                 └─► W_B ─► φ ─► g_i^B ───────►(×)  B: output gate
     │                                                │
     └──────────────────────────────────────────────►(+) ─► x_i′
```

A and B:
```
  ids ──► scores (ids + distance only) ──► softmax ───┐
                                                      │ a_ij
  x ─┬─► RMSNorm ─► h ─┬─► W_V, β shift ─► v_j ─────►(Σ)  r_i = Σ_j a_ij v_j
     │                 │                              │ r_i
     │                 ├─► W_A ─► φ ─► g_i^A ───────►(×)  A: read gate
     │                 │                              │
     │                 │                             W_O
     │                 │                              │ o_i
     │                 └─► W_B ─► φ ─► g_i^B ───────►(×)  B: output gate
     │                                                │
     └──────────────────────────────────────────────►(+) ─► x_i′
```

## The score table survives

AME's rule: a change must keep the scores a function of token ids and distance, so that the
`vocab × vocab` score table can be precomputed for decode. The rejected ST bank gate broke it by
reading `h` inside `q·k`.

- Neither gate touches `s_ij^h`, so the table and the decode step of
  [ame-model.md](../concepts/ame-model.md) (the score row read from the table, the top-k value
  sum, the bigram selection) are unchanged.
- At decode a gate reads only the current token's `h_i`: one extra matvec per gate and layer,
  nothing cached, the value cache unchanged.
- The kernels are untouched: A multiplies the kernel's output, B the output projection's.
- `scores_read_hidden_state` stays false, so `score_matrix` and `attention_diagnostics` still
  describe the scores exactly. They no longer describe what reaches the residual (a sharp head
  can be gated off), hence the gate readouts below.

## Pros and cons

### A, elementwise (A's default)

Pros
- The best-measured placement. Qwen's G1, the same position, took PPL 6.026 → 5.761 and MMLU
  58.79 → 60.82 on a 15B MoE, cut the attention on the first token from 46.7% to 4.8%, and let a
  dense 1.7B train at a learning rate where the ungated model diverged
  ([Evidence](#evidence-qwens-gated-attention)).
- The first place in AME where context controls a head's read. Dense attention's scores already
  read the content; AME's never do. Hypothesis, untested: the gate matters more for AME than for
  dense.
- A per-head off switch: a head can return ≈ 0 for a query without parking its mass on a sink.
  In Qwen's analysis the head-specific, query-dependent gate is what removed the attention sink;
  a gate shared across heads did not.
- Breaks nothing AME relies on: the score table, the kernels, FSDP2 and memory parallelism (an
  ordinary `Linear`), the FLOP accounting (a matmul weight, charged 6·P like `W_V`).

Cons
- `dmodel²` weights and `2·dmodel²` FLOPs per token and layer, +6.7% of an AME layer's matmul
  weights. Matched, the FF gives up `dmodel/3` of width for it: the gate shows as a gain only if
  it beats that much FF.
- The evidence is from dense attention; nothing measures a gate on id-addressed scores.
- It overlaps AME's learned sink (`k_∅`, `v_∅`), which exists so a head has somewhere to put its
  mass; one of the two may end up dead weight. Needs a sink-off ablation.
- AME's attention-side matvecs per decoded token go from `2·dmodel²` (`W_V`, `W_O`) to
  `3·dmodel²`. Matched, the narrower FF pays it back, so the total per token is unchanged.
- The attention diagnostics (entropy, top-k mass) stop telling the whole story: they need the
  gate readout next to them.

### A, headwise

Pros
- Most of A's gain for 1/64 of its weights: Qwen's headwise G1 reached PPL 5.792 against 5.761
  elementwise (6.026 ungated), with +1.6M parameters against +201M.
- `H·dmodel = dmodel²/64` per layer: v4's `dff` stays (the difference is inside v4's own
  rounding), and the decode cost is an `H × dmodel` matvec, negligible.
- One number per head and token: the per-head off switch in its plainest, directly readable
  form.

Cons
- It scales a head's read as a whole and cannot reshape it per channel (the elementwise gate's
  further 0.031 of PPL in Qwen's table).
- The same overlap with the sink as elementwise A.

### B, elementwise

Pros
- It does not touch the attention's internals (heads, sink, kernel), so it carries over to any
  attention type unchanged.
- A context-dependent scale per residual channel on the attention's write: an input-dependent
  LayerScale.
- With SiLU, `x + SiLU(W_B h) ⊙ W_O r` is a GLU whose up branch is the attention read:
  unbounded, it can amplify as well as attenuate. Qwen tested SiLU only at A's position, so this
  form is unmeasured.

Cons
- The weakest evidence: Qwen's G5, the same position, gave PPL 6.017 against 6.026 for +100M
  parameters, i.e. nothing. Their reading: it adds no nonlinearity between `W_V` and `W_O`.
- No per-head off switch (every head writes into every channel), so there is no reason to expect
  A's effect on the sink.
- The same cost as elementwise A (`dmodel²` per layer; matched, `dmodel/3` of FF width) for the
  weaker expected effect.

### A and B together

Pros
- Measures what B adds once A has its per-head control, on the same parameter budget.

Cons
- Two gates in series: with sigmoid, the attention branch starts at about 0.535² ≈ 0.29 of its
  RMS. Matched with elementwise A, the FF gives up `2·dmodel/3`.
- Likely redundant, if Qwen's G5 result carries over to AME.

### The activation switch: sigmoid or SiLU

| | `sigmoid` (proposed default) | `silu` |
|---|---|---|
| Range | (0, 1): attenuates only | ≥ −0.278 and unbounded above: can amplify a channel and flip its sign |
| At A's position (Qwen, Table 1) | PPL 5.761 | 5.822 (6.026 ungated) |
| At init (SwiGLU's rule, no bias: pre-activation std 0.88) | mean 0.500, std 0.189; the branch at 0.535 of its RMS | mean 0.166, std 0.486, half the values negative; the branch at 0.513 of its RMS, mostly as a random per-channel factor |
| Off | only as the pre-activation goes to −∞ (below 0.05 needs z < −2.94) | exactly 0 at z = 0, the centre of its init |
| Same form as | Qwen's gated attention; the B2 bank gate | SwiGLU's gate (our FF); the rejected ST bank gate |

The init figures were computed for the SwiGLU rule on unit-RMS inputs at `dmodel` 768, 1024 and
1280; they do not depend on `dmodel`.

## Evidence: Qwen's gated attention

Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free*, NeurIPS 2025, [arXiv 2505.06708](https://arxiv.org/abs/2505.06708). A 15B
MoE with 2.54B active parameters, 400B tokens; Table 1 unless noted. Their gate reads the
pre-normalized hidden state and has no bias (Eq. 5); its init is not stated.

| Gate | PPL | Added params | Here |
|---|---|---|---|
| none | 6.026 | — | AME today |
| G1, after the attention output, elementwise, sigmoid | **5.761** | +201M | A elementwise, sigmoid |
| G1, headwise, sigmoid | 5.792 | +1.6M | A headwise, sigmoid |
| G1, elementwise, SiLU | 5.822 | +201M | A elementwise, SiLU |
| G5, after the output projection, sigmoid | 6.017 | +100M | B, sigmoid |
| G1, input-independent (a learned vector; Table 4) | 5.917 | — | — |
| G2, on the value projection, elementwise / headwise | 5.820 / 5.808 | +25M / +0.2M | — |

With the G1 gate the share of attention on the first token fell from 46.7% to 4.8% (Fig. 2) and
the mean per-layer largest activation from 1053 to 94 (Table 4); a dense 1.7B at 1T tokens
trained at learning rate 8e-3, where the ungated model diverged (Table 2). The paper reports no
seeds or error bars.

## Matching

Decided: matched to dense in active parameters, as AME v4 is. An elementwise gate costs
`dmodel²` per layer, which is `3·dmodel·(dmodel/3)`: exactly `dmodel/3` of SwiGLU width. So the
matched FF is

```
dff = 32 · round((26 − 2k) · H / 3)      k = number of elementwise gates (A elementwise, B)
k = 0:  AME v4, dense's dff + dmodel/3
k = 1:  dense's dff, 256·H, exactly
k = 2:  dense's dff − dmodel/3
```

A headwise gate adds `H·dmodel = dmodel²/64` per layer, less than v4's own rounding of `dff` to
a multiple of 32, so it leaves `dff` alone.

| Config | k | `dff` h12 / h16 / h20 | Active params vs dense, h12 / h16 / h20 |
|---|---|---|---|
| AME v4, no gate | 0 | 3328 / 4448 / 5536 | +0.02% / +0.17% / −0.12% |
| A headwise | 0 | 3328 / 4448 / 5536 | +0.08% / +0.24% / −0.03% |
| A elementwise | 1 | 3072 / 4096 / 5120 | +0.02% / +0.02% / +0.02% |
| B | 1 | 3072 / 4096 / 5120 | +0.02% / +0.02% / +0.02% |
| A headwise + B | 1 | 3072 / 4096 / 5120 | +0.08% / +0.09% / +0.10% |
| A elementwise + B | 2 | 2816 / 3744 / 4704 | +0.02% / −0.13% / +0.15% |

The accounting is v4's (its comment in `configs/standard/ame_h16.yaml`, which this reproduces:
+590,336 at h16): per block the matmul weights, one row of each bank, τ, α, β, the sinks and the
norms, plus the embedding and the head, against dense GQA with its own τ, α, β. Per layer it
reduces to `(2 + k)·dmodel² + 3·dmodel·dff` (plus `H·dmodel` for a headwise A) against dense's
`3·dmodel² + 3·dmodel·256H`.

- With one elementwise gate, AME at dense's `dff` has AME v4's parameters, so the comparison
  against v4 asks: are `dmodel²` weights worth more as a gate than as `dmodel/3` of FF width?
- Matched configs change the FF's shape, so the FF initializes differently from v4 at the same
  seed (as v4's did from v3); every other part keeps its init (per-component seeds,
  `src/core/init_rng.py`).

## Open points (proposed defaults)

1. **Default activation:** `sigmoid`, on Qwen's G1 result; `silu` is the literal SwiGLU form.
2. **One activation switch** for both gates; split it per gate only if an ablation needs that.
3. **Init:** SwiGLU's rule, no bias, so a sigmoid gate starts at 0.5 and the branch at about
   half its RMS. Alternative, sigmoid only: B2's open init (`W = 0`, bias 3, gate 0.953), so
   step 0 is the ungated model.
4. **Sink:** on in the first runs, as in the standard AME; A with the sink off as the follow-up.
5. **Dense control:** Qwen measured these gains on dense attention, so an AME-against-dense
   comparison needs the same gates on the dense layer, as the KV shift and QK-norm already are.
   AME against AME with a gate does not.
6. **Learning rate:** start at the standard `n`; Qwen's larger-LR result suggests also trying
   `n − 1` if a gate helps (the convention tunes `n ± 1`).
7. **Activation memory** is unmeasured: each elementwise gate keeps about one more
   `B × n × dmodel` tensor per layer for backward, and the matched FF a little less. h16 is the
   tight case, as for v4.

## As built on `ame_gate`

Built 2026-11-27 from the [implementation plan](#implementation-plan) below. Three commits on
`ame_gate`; nothing pushed, nothing merged into `ame_main`.

| Commit | What |
|---|---|
| `9299e5aa` | `src/core/ame.py`, `src/core/attention.py`: the options, the parameters, the forward, the readout on the layer |
| `b97c79d2` | `src/core/trainer.py`: `Trainer.eval` resets the readout and logs it |
| `7d9af3f3` | the tests, listed below |

**The layer** (`AMEAttention`):
- Options `read_gate` (`none` | `elementwise` | `headwise`), `output_gate` (`none` |
  `elementwise`) and `gate_activation` (unset | `sigmoid` | `silu`), registered in
  `_AME_OPTIONS`. Unset means `sigmoid` once a gate is on. Each of these is a `ValueError`:
  `gate_activation` set while both gates are `none`, any unknown value (the message names the
  valid set), and `output_gate: headwise` (B has no heads to gate).
- `read_gate_proj` (`dmodel → H·d` elementwise, `dmodel → H` headwise) and `output_gate_proj`
  (`dmodel → dmodel`) are bias-free `nn.Linear`s. They are built last in `__init__`, after the
  bank gates, so every existing entry keeps its place in the state dict. They take SwiGLU's
  init from `private_generator("read_gate")` and `private_generator("output_gate")`, through
  one helper, `_private_linear`. ST's gate now uses that helper too, and its weights are as
  before, bit for bit (checked at seeds 0 and 11).
- The forward:
  1. The kernel's output, reshaped to `[B, n, H·d]`, is multiplied by
     `φ(read_gate_proj(x))`. Headwise, each head's `d` channels are multiplied by its one
     value.
  2. Then `o_proj`.
  3. Then the product with `φ(output_gate_proj(x))`.

  `φ` runs in the linear's output dtype and is cast to the dtype of the tensor it multiplies.
  Unchanged: the kernels, `_mixed_qkv`, `score_matrix`, `attention_diagnostics` and
  `scores_read_hidden_state`.
- The gate weights decay and count as matmul weights in the MFU accounting. `extra_repr`
  shows the three options.

**The readout** (plan step 7's design; it hit no blocker):
- An eval forward (`self.training` false) passes each gate's values to `_record_gate`, a
  `@torch.compiler.disable` method. It adds them to running sums, kept on the module as
  tensors, so a batch costs no host sync.
- In train mode the branch is a Python constant, so the training graph has neither the branch
  nor a graph break (the CPU `fullgraph=True` test below). In an eval forward under compile it
  is a graph break by design, as Residual's logging is.
- `Trainer.eval` calls `reset_gate_statistics()` on every module before its batches. After
  `eval/loss` it logs `gate_statistics()` as `eval/<layer>/<gate>/<statistic>`:

  | Statistic | When | Meaning |
  |---|---|---|
  | `mean` | always | the mean gate value |
  | `below_0.05` | always | the share of values below 0.05 |
  | `below_0` | `silu` | the share of values below zero |
  | `head_mean_min` | headwise A | the lowest per-head mean |

- The hook is duck-typed, so a stack without gates logs nothing. A readout that raises is
  warned about and skipped.
- Rank-local: on several GPUs each rank's statistics cover its own eval batches, and W&B shows
  rank 0's.
- `final_eval`'s forwards are eval forwards, so they add to the sums, but `final_eval` logs
  nothing from them.

**Baseline** (plan step 2), on the tree at `06ddbbc0` with `data/` and `data_eval/` linked in:
- CPU gate: 580 passed, 0 skipped, 0 failed (22 deselected, 697 s).
- Slow CPU tests: 7 passed (268 s).
- The h4 AME fingerprint at seed 1234: state dict `ab0e1527…` (63 entries), forward
  `bbd75920…`.

**Tests** at `7d9af3f3`, in the worktree with `data/` and `data_eval/` linked in:
- CPU gate: 672 passed, 0 skipped, 0 failed (24 deselected, 396 s).
- Slow CPU tests: 9 passed, 0 failed (184 s).
- `black --check`: clean.

The new tests:
- `tests/test_ame_gates.py` (83):
  - no regression: the fingerprint, and the default equal to an explicit `none`;
  - validation;
  - shapes and init per mode, including the init figures of the activation table;
  - the math against a literal transcription of [the two gates](#the-two-gates), for every
    combination under both activations;
  - headwise A as elementwise A with tied rows; B as its gate times the ungated output; a head
    closed under SiLU;
  - the score table; gradients; the CPU compile; the state-dict round trip;
  - the optimizer groups and the MFU count;
  - [Matching](#matching) from `configs/standard`: the test reproduces the table to 0.01% and
    v4's +590,336;
  - the readout, on the layer and through `Trainer.eval`.
- `tests/test_attention_registry.py` (+3): the options through `build_attention`, and into
  every AME layer of a uniform and of a mixed stack.
- `tests/test_init_rng.py` (+6 `SWITCHES`): each of A elementwise, A headwise, B, A+B, B beside
  A, and `silu` against `sigmoid` moves no shared parameter and leaves the global stream as it
  was.
- `tests/test_training_smoke.py` (+2, slow): A elementwise + B, and A headwise + `silu`,
  through `main.py` with evals at steps 5 and 10. Each run has finite losses, writes its
  checkpoint and logs the readout (4 layers × 2 evals per statistic).

**Departures from the plan, and where it was wrong:**
- The CPU compile test uses `backend="aot_eager"`. That is dynamo with `fullgraph=True`, which
  fails on any graph break, plus AOTAutograd's backward, with eager kernels. Inductor cannot be
  used on the CPU: on torch 2.7.1 its compiled backward of the AME reference path corrupts the
  heap (`malloc(): invalid size`, then an abort or a segfault).
  - This happens with no gate at all, at `d_head` 8 and 64. The compiled forward alone runs
    fine.
  - It is a failure of the baseline that no baseline test exercised. Per the plan's rule it is
    recorded here, not fixed. Inductor is left to the GPU pass.
- The plan says "as `tests/test_ame.py` does", but that file's one `fullgraph=True` compile is
  a GPU test, so the CPU compile test is new.
- The `build_attention` and mix tests are in `tests/test_attention_registry.py`, which the plan
  names, not in `tests/test_ame_gates.py`.
- The forward half of the fingerprint test compares the digest bit for bit only on a CPU with
  the recording's capability (AVX2), because the last bits of a CPU matmul depend on the
  kernels torch dispatches to. Elsewhere it checks the logits' sums to 1e-6. The state-dict
  digest is checked exactly everywhere.
- A `configs/standard/` config has to be composed from its own directory, as
  `--config-path configs/standard` does. Composed as `standard/<name>` from `configs/`, Hydra
  nests the whole config under a `standard` key.

## Implementation plan

Executed on 2026-11-27; the result is in [As built](#as-built-on-ame_gate).

**Goal:** branch `ame_gate` holds both gates as this file specifies (the
[decisions](#decisions-maciej-2026-11-27) and the proposed defaults of
[open points](#open-points-proposed-defaults) 1–3) and passes every CPU test. Points 4–7 are
for the runs, not for this build. The GPU tests are a later, separate pass
([listed below](#later-the-gpu-pass)). This section is written for the agent that executes it.

**Done when:**
- `ame_gate` holds the build in logical commits.
- In the worktree: the CPU gate has 0 failed and 0 errors, with no more skips than the
  baseline (step 2); the slow CPU tests have 0 failed; `black --check` is clean. The final
  report quotes the pytest summary lines and black's output.
- This file's status says: implemented on `ame_gate` at `<sha>`, CPU-tested (with counts), GPU
  tests pending.
- Nothing is pushed; `ame_main` gained only doc commits; no GPU test or cluster job ran.

**Rules:**
- Never change the session's working directory. Reach `~/code/nano-ame-gate` by absolute path
  and `git -C`. Run pytest and black in a subshell:
  `(cd ~/code/nano-ame-gate && ~/code/nano/.pixi/envs/default/bin/python -m pytest …)`.
- Code goes on `ame_gate`. This file is edited in `~/code/nano` and committed on `ame_main`,
  staging only your own hunks (other sessions edit there), then cherry-picked onto `ame_gate`.
- Do not merge `ame_gate` into `ame_main`. Push nothing.
- No GPU test, no cluster job.
- The decisions are final. If the code contradicts this plan or a decision is missing, ask
  Maciej instead of guessing.
- A failure already in the baseline is not yours: record it here, don't fix it.

### Steps

1. **Set up.** Symlink the gitignored `~/code/nano/data` and `~/code/nano/data_eval` into the
   worktree (they stay untracked) so the slow smoke tests run. Read this file in full, then
   `src/core/ame.py` (`AMEAttention.__init__`, `_init_gate`'s `stem` branch, `forward`,
   `instrumentation`), `src/core/attention.py` (`_AME_OPTIONS`, `_build_ame`),
   `src/core/init_rng.py`, `src/core/trainer.py` (`eval`, `_log_layer_instrumentation`),
   `tests/test_init_rng.py` and the AME tests in `tests/test_training_smoke.py`.
2. **Baseline,** on the untouched tree: the CPU gate from `.github/workflows/tests.yml`
   (`pytest -m "not gpu and not slow and not network"`) and the slow CPU tests
   (`-m "slow and not gpu and not network"`). Record passed, skipped and failed here. Save a
   fingerprint of the h4 AME model at seed 1234 (every parameter, and a forward on fixed ids)
   for the no-regression test (step 8).
3. **Options** in `common.attention.ame`, added to `_AME_OPTIONS` and to `AMEAttention`'s
   keyword-only arguments:
   - `read_gate`: `none` (default) | `elementwise` | `headwise`
   - `output_gate`: `none` (default) | `elementwise`
   - `gate_activation`: unset (default) | `sigmoid` | `silu`. Unset means `sigmoid` when a gate
     is on. Setting it while both gates are `none` is a `ValueError`, as `confidence_init` is
     without `bank_gate: confidence`.
   - An unknown value is a `ValueError` that names the valid set.
4. **Parameters,** built last in `__init__` (after `_init_gate`), so every existing parameter
   keeps its place in the state dict:
   - `read_gate_proj`: `Linear(dmodel → d_att)` elementwise, `Linear(dmodel → q_heads)` headwise.
   - `output_gate_proj`: `Linear(dmodel → dmodel)`.
   - No bias. Build them with `_rng_neutral_linear` and initialize them with SwiGLU's rule,
     `llm_random_weight_init(dmodel, init_scale)` (a truncated normal, std `1/√dmodel`), drawn
     on the CPU from `private_generator("read_gate")` and `private_generator("output_gate")`,
     exactly as the `stem` gate is (no draw on the meta device). No other parameter's init may
     move.
5. **Forward,** with `x` the layer's normalized input:
   `out = attend(q, k, v)`, reshaped to `d_att` → A: elementwise `out ⊙ φ(read_gate_proj(x))`,
   headwise each head's `dhead` channels times its scalar → `o = o_proj(out)` →
   B: `o ⊙ φ(output_gate_proj(x))`. `φ` runs in the compute dtype (as the FF's SiLU) and is cast
   to `out`'s dtype. Unchanged: the kernels (flex and reference), `_mixed_qkv`, the scores,
   `score_matrix`, `attention_diagnostics`, `scores_read_hidden_state`.
6. **Optimizer and accounting.** The gate weights decay like every `Linear` weight: they are not
   in `no_weight_decay_parameters`, not banks (`bank_lr_ratio` does not apply) and not in
   `non_matmul_parameters`, so `src/core/mfu.py` counts them as matmul weights. `extra_repr`
   shows the three options.
7. **Readouts that leave the training graph untouched.** This replaces the design's "on the
   heavy-metric interval":
   - Gate statistics are collected only in eval forwards (`self.training` is `False`), so dynamo
     specializes the train graph without a branch or a graph break. A
     `@torch.compiler.disable` helper accumulates them over the eval batches.
   - They are reset at the start of each `Trainer.eval()` and logged at its end, per AME layer
     and gate: the mean gate, the share below 0.05, for `silu` the share below 0, and for
     headwise A the lowest per-head mean.
   - The hook is duck-typed like `instrumentation()`, so a stack without gates logs nothing and
     pays nothing.
   - The statistics are rank-local on multi-GPU; record that here.
   - If this design hits a blocker, choose another that keeps the training path unchanged and
     record why.
8. **Tests,** in a new `tests/test_ame_gates.py` unless noted:
   - **No regression:** without gate options, the h4 fingerprint matches step 2's bit for bit;
     the default layer equals one with explicit `none` (state-dict keys, parameters, output).
   - **Validation:** every bad value, `output_gate: headwise` and `gate_activation` without a
     gate raise a `ValueError` naming the valid set. The three keys pass `build_attention`
     (`tests/test_attention_registry.py`) and reach every AME layer, including AME layers in a
     `mix`.
   - **Shapes, no bias and init,** per mode: weight std ≈ `0.88/√dmodel` (the truncation),
     `|w| ≤ 2/√dmodel`; a sigmoid gate on unit-RMS input averages ≈ 0.50 at init.
   - **Math** (reference backend, fp32): the output equals a manual computation from the layer's
     own parameters for A elementwise, A headwise, B and A+B, under both activations. Headwise A
     equals elementwise A with `W_A`'s rows tied within each head. B equals `φ(W_B x) ⊙` the
     ungated output with the same parameters.
   - **Score table:** `score_matrix` and `attention_diagnostics` are identical with gates on and
     off; `scores_read_hidden_state` stays `False`.
   - **Training path:** gradients are finite and nonzero for both gate weights. The CPU
     fullgraph compile of the layer (as `tests/test_ame.py` does) works with gates on, in train
     mode. A gated `state_dict` round-trips into a fresh model with identical outputs.
   - **Optimizer and accounting:** the gate weights land in `parameter_groups`' decay group, not
     the bank group; the MFU matmul-parameter count rises by exactly the gate weights.
   - **Init isolation** (`tests/test_init_rng.py`'s `SWITCHES`): read gate elementwise, read
     gate headwise, output gate, and both move only `read_gate_proj`/`output_gate_proj`, and
     the global stream after build is unchanged; `gate_activation` `silu` against `sigmoid`
     (with a gate on) moves no parameter.
   - **Matching:** compose `configs/standard/{ame,dense}_h{12,16,20}.yaml` on the meta device,
     AME with gates and `dff = 32·round((26 − 2k)·H/3)`. Active parameters (a bank counts one
     row per layer) are within ±0.3% of dense, reproducing the [matching](#matching) table.
   - **End to end** (`tests/test_training_smoke.py`, slow): AME with A elementwise + B, and with
     A headwise + `silu`, trains through `main.py` on the CPU: finite losses, a checkpoint
     written, the gate readouts in the log (set the eval interval so an eval runs inside the
     run).
9. **Gate.** Rerun the CPU gate and the slow CPU tests as in step 2, plus
   `~/code/nano/.pixi/envs/default/bin/black --check --extend-exclude='\.ipynb$' .`, and
   compare with the baseline.
10. **Update this file:** the status at the top (implemented on `ame_gate` at `<sha>`,
    CPU-tested with counts, GPU tests pending), what was built, the readout mechanism, and
    anything that departs from this plan. Commit it on `ame_main`, then cherry-pick it onto
    `ame_gate`. End with a report: what was built, test counts before and after, the commits,
    what is left open.

### Later: the GPU pass

Not part of this plan; it runs once the build is in.

- flex against reference with gates: forward and backward, bf16 autocast.
- The compiled training path: no new graph breaks.
- Inductor on the gated layer, forward and backward. The CPU tests run dynamo and AOTAutograd
  only ([As built](#as-built-on-ame_gate)).
- The eval path under `compile_blocks_full`: the readout's graph break, the recompiles it costs
  against the recompile limit, and the eval time.
- FSDP2 and the memory-parallel controller with gates, including the rank-local readout;
  checkpoint resume.
- Step time, MFU and peak memory at h16 for no gate, A, and A+B, matched
  ([open point](#open-points-proposed-defaults) 7).
