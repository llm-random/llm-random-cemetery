"""Document masking for packed rows: `common.attention.document_mask`.

The packer (src.core.datasets.AbstractDataset.sample_packer) fills each row with
documents back to back, each ending in the tokenizer's EOS, and cuts the row at
sequence_length. Without a mask, attention is causal over the whole row, so a
token also attends to every earlier document in its row. With
`attention.document_mask: true`, it attends only within its own document:

- **Document ids come from the EOS positions.** A position's document is the
  number of EOS tokens strictly before it. So an EOS belongs to the document it
  ends, and the token after it starts the next one. The first document in a row
  is usually the tail of one the packer cut into at a random offset.
- **The mask.** Query i sees key j iff j <= i and both are in the same document.
  Every query sees at least itself, so no row is fully masked.
- **The KV shift stops at a boundary.** The shifted term (`k_{j-1}`, `v_{j-1}`)
  is zero at the first position of every document, as it is at position 0 of a
  row (AME's `_shift1`). Otherwise the first token of a document would mix in
  the previous document's EOS key and value.

Positions are not reset per document, and neither encoding needs it: RoPE's
score depends on q_pos - k_pos only, and ALiBi's bias on i - j only, so within
one document both are the same at any offset. The tests check exactly that: a
packed row gives each document the output it gets alone.

The loss is unchanged: every position is trained, including the EOS that
predicts the next document's first token from the document it ends.
"""

import torch


def document_ids(token_ids, eos_token_id):
    """`[B, n]` int32: each position's document index within its row.

    Counts the EOS tokens strictly before each position, so an EOS belongs to
    the document it ends and the position after it starts a new one.

    A custom op, so that inside a compiled graph the ids land in a buffer with a
    fixed layout. The flex kernel's score_mod indexes them (`doc[b, q_idx]`),
    and torch 2.7.1's inductor cannot lower an index into a buffer whose layout
    is still flexible ("convert FlexibleLayout to FixedLayout first"). With
    gradients on, the forward keeps the ids for the backward, which fixes their
    layout anyway; under no_grad (eval, final_ce) nothing else did, and both h12
    runs died at their first eval. Inductor does not fuse a custom op: it
    allocates the output from the op's fake (contiguous int32) and calls it,
    which fixes the layout in every compile mode, with or without gradients.
    """
    return _document_ids(token_ids, int(eos_token_id))


@torch.library.custom_op("nano::document_ids", mutates_args=())
def _document_ids(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    is_eos = (token_ids == eos_token_id).to(torch.int32)
    return torch.cumsum(is_eos, dim=1, dtype=torch.int32) - is_eos


@_document_ids.register_fake
def _document_ids_fake(token_ids, eos_token_id):
    return torch.empty(token_ids.shape, dtype=torch.int32, device=token_ids.device)


def continues_document(doc):
    """`[B, n]` bool: position j is in the same document as position j - 1.

    False at position 0 and at the first position of every document: exactly the
    positions where a KV shift must not reach back.
    """
    first = torch.zeros_like(doc[:, :1], dtype=torch.bool)
    return torch.cat([first, doc[:, 1:] == doc[:, :-1]], dim=1)


def allowed_keys(doc):
    """`[B, n, n]` bool: query i may attend key j (causal, same document).

    For the paths that materialize the scores. The flex paths use
    `document_score_mod` instead, which never builds this.
    """
    seq_len = doc.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=doc.device).tril()
    return causal & (doc[:, :, None] == doc[:, None, :])


def document_score_mod(doc, score_mod=None):
    """A flex `score_mod` that sends every cross-document score to -inf, applied
    after `score_mod` when one is given (AME's ALiBi bias).

    A score_mod rather than a mask_mod on purpose. A mask_mod would need a block
    mask built per batch from `doc`, either inside the compiled block, where
    tracing `create_block_mask` breaks the graph, or outside it, where the
    BlockMask becomes a graph input (src.core.ame `_block_mask` records why both
    were rejected for AME). This keeps the cached causal block mask, so the only
    new input is `doc`, computed in the graph from the token ids it already has.
    The cost: cross-document blocks are computed and masked, not skipped.
    """

    def mod(score, b, h, q_idx, kv_idx):
        if score_mod is not None:
            score = score_mod(score, b, h, q_idx, kv_idx)
        return torch.where(doc[b, q_idx] == doc[b, kv_idx], score, -float("inf"))

    return mod
