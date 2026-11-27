# AME attention gates: a read gate and an output gate

**Status: proposed, 2026-11-27 (Maciej). Not implemented: this doc fixes the design before any
code.** Code goes on branch `ame_gate` (worktree `~/code/nano-ame-gate`, cut from `ame_main` at
`412fc774`).

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

Proposed config keys (the names are proposals; nothing reads them yet):

```yaml
common:
  attention:
    type: ame
    ame:
      read_gate: none           # A: none | elementwise | headwise
      output_gate: none         # B: none | elementwise
      gate_activation: sigmoid  # sigmoid | silu
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

## For the implementation (later; nothing is built)

- **Where:** both in `AMEAttention.forward`, which already has `h` (its input, normalized by
  `Residual`): A between the kernel's output, reshaped to `d_att`, and `o_proj`; B on
  `o_proj`'s output. The kernels and `Residual` do not change.
- **Parameters:** `Linear(dmodel → d_att or H)` for A and `Linear(dmodel → dmodel)` for B, no
  bias, built last in `__init__` so every existing parameter keeps its place in the state dict.
- **Init:** SwiGLU's rule (`llm_random_weight_init(dmodel, init_scale)`: a truncated normal,
  std `1/√dmodel`), drawn from a private generator (`private_generator`, tags `read_gate` and
  `output_gate`), as the ST gate is, so a gate switch moves no other weight's init.
- **Optimizer:** weight decay as for every `Linear` weight; no bias, so nothing joins
  `no_weight_decay_parameters`; not a bank, so `bank_lr_ratio` does not apply.
- **Validation:** an unknown value fails at build with the offender named, and
  `gate_activation` set while both gates are `none` is an error, as `confidence_init` is without
  `bank_gate: confidence`.
- **Readouts,** on the heavy-metric interval with the other AME readouts: per layer and head,
  the mean gate and the share of gate values below 0.05 (Qwen's sparsity readout); for SiLU also
  the share below 0.
- **Tests:** gates off is bit-identical to today's layer (state dict and output); each gate
  switch moves only its own parameters' init (`tests/test_init_rng.py`'s `SWITCHES`);
  `score_matrix` is unchanged with the gates on; headwise A equals elementwise A with `W_A`'s
  rows tied within each head; B equals scaling the module's output.

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
