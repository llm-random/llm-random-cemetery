"""Hand-managed, column-sharded Q/K banks -- Tiers 2 and 3 of the AME plan.

AME replaces `W_Q`/`W_K` with two `nn.Embedding` banks indexed by token id. At
h20 that is 2.576B of the model's 3.163B parameters, and FSDP2 treats them like
any other weight: it all-gathers every bank on the way into each micro-step's
forward and again on the way into its backward, materialises a dense gradient
for each, reduce-scatters it, and hands the whole thing to AdamW. The forward
only ever reads `micro_batch * seq_len` of the 50304 rows, so almost all of that
traffic is for rows nothing touched.

This module takes the banks off FSDP2 entirely and runs them by hand:

  O1  `fully_shard(..., ignored_params=...)` keeps FSDP2 out. Each bank keeps a
      compute parameter in the autocast dtype, a private fp32 master, and a
      persistent fp32 `main_grad` the backward scatters into with `index_add_`.
      No dense per-micro-step gradient is ever allocated, and `main_grad` is not
      zeroed between micro-steps, so accumulation is free.
  O2  One `all_reduce(SUM)` of `main_grad` per optimizer step, then
      `div_(world_size)` -- FSDP2 reduces gradients as a MEAN over ranks, and
      the banks must agree with the backbone about what a gradient means.
  O3  fp32 master -> compute parameter copy after each step.
  O4  The clip. See `local_grad_norm_sq`: the norm is computed JOINTLY with the
      backbone's, not separately, because a separate clip is a different
      function and the section 1.4 gate compares against a baseline that clips
      once over everything.
  O5  Three hook points, called from `src.core.trainer.Trainer.train`:
      `materialize()` at step begin, `reduce()` after the accumulation loop, and
      `step()` beside the backbone's optimizer step.
  O6  Two process groups. The *memory-parallel* group is a contiguous block of
      `mem_parallel_size` ranks that between them own one whole bank, each rank
      holding a column slice `[vocab, width / mp]`. The *memory-DP* subgroup is
      strided with size `world / mp` and holds the ranks that own the SAME
      slice; `main_grad` is all-reduced only there.
  O7  The forward all-gathers token ids inside the mp-group, gathers every
      rank's tokens from the locally owned columns, and all-to-alls the result
      so each rank ends up with its own tokens at full width. The backward
      mirrors it. Attention downstream sees exactly the tensor it saw before.
  O8  **The divisor is the GLOBAL world size, not the size of the group the
      all-reduce ran in.** The id all-gather in the forward already summed `mp`
      ranks' token contributions into each slice's local gradient, and the
      all-reduce then sums the remaining `world / mp`, so the terms total
      `mp * (world / mp) = world`. Getting this wrong scales every bank gradient
      by a constant and is invisible in the loss for hundreds of steps -- which
      is why `tests/test_memory_parallel.py` compares `mp=1` against `mp=4`
      directly.

A column slice is a HEAD subset (`width = q_heads * d_head`, and heads are
independent until `o_proj`), so this split is natural for attention even though
the all-to-all restores full width before the layer does anything with it.

`mem_parallel_size=1` is Tier 2 on its own: no collectives in the bank forward,
every rank holding a whole bank. It is mathematically identical to `mp>1` and
exists as the correctness reference. At h20 it does not FIT -- an unsharded bank
costs 46.4 GB/rank of state against 11.6 GB sharded -- so on real hardware the
useful settings are `mp == world_size`.
"""

import collections
import contextlib
import logging
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _BankGather(torch.autograd.Function):
    """ids -> `[*ids.shape, width]`, with the backward scattering into `main_grad`.

    Returns `None` as the gradient of `weight`, so `weight.grad` stays None
    forever. That is what keeps the dense per-micro-step gradient from ever
    being allocated, and it is also what makes the bank invisible to
    `Trainer.clip_gradient` (which filters on `p.grad is not None`) and to any
    optimizer that was handed the parameter by accident.
    """

    @staticmethod
    def forward(ctx, weight, ids, bank):
        mp_size = bank.mp_size
        ids_flat = ids.reshape(-1).contiguous()
        n = ids_flat.numel()

        if mp_size == 1:
            gathered_ids = ids_flat
            out = F.embedding(gathered_ids, weight).to(bank.compute_dtype)
        else:
            # O7, forward half. Every rank needs every rank's ids, because the
            # columns it owns are the only source for those columns.
            gathered_ids = torch.empty(
                mp_size * n, dtype=ids_flat.dtype, device=ids_flat.device
            )
            dist.all_gather_into_tensor(gathered_ids, ids_flat, group=bank.mp_group)
            local = F.embedding(gathered_ids, weight).to(bank.compute_dtype)
            shuffled = torch.empty_like(local)
            # Chunk i of `local` is rank i's tokens on MY columns and goes to
            # rank i; chunk j of the result is MY tokens on rank j's columns.
            dist.all_to_all_single(shuffled, local.contiguous(), group=bank.mp_group)
            # Rank j owns columns [j*w, (j+1)*w), so concatenating the received
            # chunks in rank order restores the original column order exactly.
            out = shuffled.view(mp_size, n, -1).permute(1, 0, 2).reshape(n, -1)

        ctx.bank = bank
        ctx.n = n
        ctx.save_for_backward(gathered_ids)
        return out.view(*ids.shape, out.shape[-1])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        bank = ctx.bank
        (gathered_ids,) = ctx.saved_tensors
        mp_size, n = bank.mp_size, ctx.n
        grad = grad_out.reshape(n, -1)

        if mp_size == 1:
            local = grad
        else:
            # O7, backward half: the forward's all-to-all run with the splits
            # swapped. Slice j of the width belongs to rank j, so it is sent
            # there; what comes back is every rank's gradient for MY columns,
            # row-aligned with `gathered_ids`.
            width = grad.shape[1] // mp_size
            send = (
                grad.view(n, mp_size, width)
                .permute(1, 0, 2)
                .reshape(mp_size * n, width)
                .contiguous()
            )
            local = torch.empty_like(send)
            dist.all_to_all_single(local, send, group=bank.mp_group)

        bank.accumulate(gathered_ids, local)
        return None, None, None


class ShardedBank(nn.Module):
    """A column slice of one Q/K bank, with its own fp32 master and gradient.

    Drop-in for the `nn.Embedding` it replaces: `bank(ids)` still returns
    `[batch, seq, width]` at full width, so `AMEAttention._mixed_qkv` -- the
    measured hot path -- is untouched.
    """

    # src.core.mfu charges 6*P for every 2-D parameter unless the module says
    # otherwise, and identifies lookups by `isinstance(module, nn.Embedding)`.
    # This is not one, so it must opt out by name or the MFU denominator would
    # gain 6 * 2.576B FLOPs/token the banks never spend -- and every number in
    # the plan's ledger would stop being comparable to the ones above it.
    non_matmul_parameters = ("weight",)

    def __init__(self, vocab_size, width, controller):
        super().__init__()
        mp_size = controller.mp_size
        if width % mp_size:
            raise ValueError(
                f"bank width {width} is not divisible by mem_parallel_size {mp_size}"
            )
        self.vocab_size = vocab_size
        self.width = width
        self.mp_size = mp_size
        # The master and the gradient accumulator are "fp32" in the sense that
        # matters -- higher precision than the compute parameter, and at least
        # fp32. Derived rather than hardcoded so a float64 gate really runs in
        # float64: an fp32 master under an fp64 model loses ~1e-7 per step,
        # which is six orders of magnitude above the noise the section 1.4 gate
        # is trying to distinguish a real bug from.
        self.master_dtype = (
            controller.compute_dtype
            if controller.compute_dtype.itemsize > 4
            else torch.float32
        )
        self.mp_group = controller.mp_group
        self.slice_width = width // mp_size
        self.column_offset = controller.mp_rank * self.slice_width
        # The dtype the gathered rows leave in. With a compute copy (the
        # default, every run before O38) the weight already is that dtype.
        self.compute_dtype = controller.compute_dtype
        # O38(b): without a compute copy the parameter the forward gathers from
        # IS the fp32 master, and each gather casts its rows down on the way
        # out. The rows are bit-identical -- the copy only ever held
        # `master.to(compute_dtype)` -- and the copy's memory goes: 2 bytes per
        # bank parameter, 4.8 GiB at h20 on one GPU, 1.2 GiB per rank on four.
        self.compute_copy = controller.compute_copy
        weight_dtype = (
            controller.compute_dtype if self.compute_copy else self.master_dtype
        )
        self.weight = nn.Parameter(
            torch.empty(vocab_size, self.slice_width, dtype=weight_dtype)
        )
        # Plain attributes, NOT buffers: FSDP2's MixedPrecisionPolicy casts a
        # module's buffers to the compute dtype, which would quietly turn the
        # fp32 master and the fp32 gradient accumulator into bf16 and make the
        # whole exercise pointless.
        self.master = None
        self.main_grad = None

    @torch.no_grad()
    def load_from(self, embedding):
        """Take this rank's columns out of the full, already-initialised bank.

        Every rank builds the identical model under one seed before this runs,
        so slicing by `column_offset` gives a consistent global parameter with
        no communication -- and `mp=1` and `mp=4` start from the same numbers,
        which is what makes their loss trajectories comparable.
        """
        full = embedding.weight.detach()
        columns = full[:, self.column_offset : self.column_offset + self.slice_width]
        self.weight.copy_(columns.to(self.weight.dtype))
        if self.compute_copy:
            self.master = columns.to(self.master_dtype).clone()
        else:
            self.master = self.weight.data  # one tensor, two names

    def materialize(self, device):
        """O5 hook 1: make sure the master and the accumulator sit on the
        parameter's device, allocating the accumulator on first use."""
        if not self.compute_copy and (
            self.master is None or self.master.data_ptr() != self.weight.data_ptr()
        ):
            # The master is the weight. Rebind only when stale: the bank
            # optimizer holds this very tensor object, and `step()` hands it its
            # gradient through `master.grad` -- a fresh `.data` view each step
            # would leave the optimizer's own object without one, and the banks
            # would silently never move.
            self.master = self.weight.data
        if self.master is None:
            self.master = self.weight.detach().to(self.master_dtype).clone()
        if self.master.device != device:
            self.master = self.master.to(device)
        if self.main_grad is None:
            self.main_grad = torch.zeros_like(self.master)
        elif self.main_grad.device != device:
            self.main_grad = self.main_grad.to(device)

    def accumulate(self, ids, grad):
        """Scatter one micro-step's gradient into the persistent fp32 buffer.

        `index_add_` rather than a dense `embedding_dense_backward` + add: the
        dense form allocates `[vocab, slice_width]` per micro-step per bank and
        then adds it to a buffer of the same shape, which is the allocation this
        tier exists to delete.
        """
        self.main_grad.index_add_(0, ids, grad.to(self.main_grad.dtype))

    # Eager, always. Dynamo cannot trace this: it is a custom autograd.Function
    # that stashes a live module on `ctx` and calls three process-group
    # collectives, and a silent partial trace here would be a wrong gradient
    # rather than a slow one. The cost is a graph break at the top of each
    # attention layer, which leaves the expensive part -- the mix, the kernel
    # and the feed-forward -- inside one compiled region. If compile_blocks_full
    # is ever measured as worse with this in it, that is the number to compare,
    # not a reason to let dynamo in.
    @torch.compiler.disable
    def forward(self, ids):
        return _BankGather.apply(self.weight, ids, self)

    # ---- checkpointing -------------------------------------------------
    #
    # The bank is deliberately absent from the module's state_dict. At
    # mem_parallel_size > 1 every rank's `weight` has the SAME fully-qualified
    # name and a DIFFERENT quarter of the columns, and it is a plain tensor, not
    # a DTensor -- so a distributed save would write one rank's slice under that
    # name and silently drop the rest. The controller writes a rank-local
    # sidecar instead (`MemoryParallelController.save`), which also carries the
    # fp32 master and the bank optimizer's moments; the compute parameter is
    # derived from the master on load and is not worth storing twice.

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Loud, not silent: a checkpoint written WITHOUT memory-parallel banks
        # carries `<prefix>weight`, and loading it here would otherwise look
        # like a success while leaving the banks at their initial values.
        for key in list(state_dict):
            if key.startswith(prefix):
                unexpected_keys.append(key)

    def extra_repr(self):
        return (
            f"{self.vocab_size}, {self.width}, mem_parallel={self.mp_size}, "
            f"columns=[{self.column_offset}, {self.column_offset + self.slice_width})"
        )


class FusedBankPair(nn.Module):
    """One gather, one all-to-all and one permute for a layer's q and k banks.

    `AMEAttention` looks both banks up on the SAME token ids, back to back, and
    uses the two results identically. Under column sharding each lookup costs a
    local gather, an all-to-all and a transposing copy, and all three are
    **launch-bound** -- the gather measured 90.8 ms for 168 MB of traffic, 5.6x
    its bandwidth bound, over 328 launches. Doing the pair in one call halves
    every one of those counts while moving exactly the same bytes.

    It owns no parameters. The two `ShardedBank` modules remain the owners of
    the weights, masters, gradients and optimizer state; this only batches the
    data path, so nothing about sharding, reduction or checkpointing changes.
    """

    def __init__(self, q_bank, k_bank):
        super().__init__()
        if q_bank.mp_size != k_bank.mp_size or q_bank.width != k_bank.width:
            raise ValueError("a fused pair needs two identically shaped banks")
        # Plain attributes, not submodules: the banks are already registered on
        # the attention layer under their own names, and registering them twice
        # would duplicate them in `model.parameters()`.
        object.__setattr__(self, "q", q_bank)
        object.__setattr__(self, "k", k_bank)

    def forward(self, ids):
        return _FusedBankGather.apply(self.q.weight, self.k.weight, ids, self)


class _FusedBankGather(torch.autograd.Function):
    """`_BankGather` for two banks at once. The two weights are concatenated
    along the column axis, so one collective carries both."""

    @staticmethod
    def forward(ctx, q_weight, k_weight, ids, pair):
        bank = pair.q
        mp_size = bank.mp_size
        ids_flat = ids.reshape(-1).contiguous()
        n = ids_flat.numel()
        width = q_weight.shape[1]

        if mp_size == 1:
            gathered_ids = ids_flat
            q = F.embedding(gathered_ids, q_weight).to(bank.compute_dtype)
            k = F.embedding(gathered_ids, k_weight).to(bank.compute_dtype)
            out_q, out_k = q, k
        else:
            gathered_ids = torch.empty(
                mp_size * n, dtype=ids_flat.dtype, device=ids_flat.device
            )
            dist.all_gather_into_tensor(gathered_ids, ids_flat, group=bank.mp_group)
            # One gather of the pair: [mp*n, 2*width/mp], q columns then k.
            local = torch.cat(
                (
                    F.embedding(gathered_ids, q_weight).to(bank.compute_dtype),
                    F.embedding(gathered_ids, k_weight).to(bank.compute_dtype),
                ),
                dim=1,
            )
            shuffled = torch.empty_like(local)
            dist.all_to_all_single(shuffled, local.contiguous(), group=bank.mp_group)
            # One permute restores full width for both halves at once.
            full = shuffled.view(mp_size, n, 2, width).permute(1, 2, 0, 3)
            full = full.reshape(n, 2, mp_size * width)
            out_q, out_k = full[:, 0], full[:, 1]

        ctx.pair = pair
        ctx.n = n
        ctx.width = width
        ctx.save_for_backward(gathered_ids)
        tail = out_q.shape[-1]
        return out_q.reshape(*ids.shape, tail), out_k.reshape(*ids.shape, tail)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k):
        pair = ctx.pair
        (gathered_ids,) = ctx.saved_tensors
        mp_size, n, width = pair.q.mp_size, ctx.n, ctx.width

        if mp_size == 1:
            pair.q.accumulate(gathered_ids, grad_q.reshape(n, -1))
            pair.k.accumulate(gathered_ids, grad_k.reshape(n, -1))
            return None, None, None, None

        # Mirror of the forward: stack the pair, one all-to-all, then split.
        stacked = torch.stack(
            (grad_q.reshape(n, mp_size, width), grad_k.reshape(n, mp_size, width)),
            dim=1,
        )  # [n, 2, mp, width]
        send = stacked.permute(2, 0, 1, 3).reshape(mp_size * n, 2 * width).contiguous()
        local = torch.empty_like(send)
        dist.all_to_all_single(local, send, group=pair.q.mp_group)
        pair.q.accumulate(gathered_ids, local[:, :width])
        pair.k.accumulate(gathered_ids, local[:, width:])
        return None, None, None, None


def _expand_rows(rows, inverse, mp):
    """O47 v2: `[mp * distinct, 2w]` -> `[mp * n, 2w]`, row `inverse[i]` of each
    rank's chunk for token i."""
    distinct = rows.shape[0] // mp
    return rows.view(mp, distinct, -1)[:, inverse].reshape(mp * inverse.numel(), -1)


def _reduce_rows(grads, inverse, mp, distinct):
    """O47 v2, backward of `_expand_rows`: token gradients summed per distinct
    id in fp32, returned in the gradients' dtype."""
    n = inverse.numel()
    width = grads.shape[1]
    # fp32 or better: `.float()` would silently downcast a float64 gate.
    acc = torch.promote_types(grads.dtype, torch.float32)
    out = torch.zeros(mp, distinct, width, dtype=acc, device=grads.device)
    out.index_add_(1, inverse, grads.view(mp, n, width).to(acc))
    return out.view(mp * distinct, width).to(grads.dtype)


def _gather_pair(ids, q_weight, k_weight, dtype):
    """Every rank's tokens on my columns, q then k: `[mp*n, 2*w]`."""
    return torch.cat(
        (F.embedding(ids, q_weight), F.embedding(ids, k_weight)), dim=1
    ).to(dtype)


def _scatter_pair(q_grad, k_grad, ids, local, width):
    """`main_grad` += this micro-step's rows, for both banks of the pair."""
    q_grad.index_add_(0, ids, local[:, :width].to(q_grad.dtype))
    k_grad.index_add_(0, ids, local[:, width:].to(k_grad.dtype))


_COMPILED_BANK_KERNELS = {}


def _bank_kernel(fn, device):
    """`fn` compiled, on CUDA; `fn` itself elsewhere.

    O39's first version ran these eagerly on the bank stream and they were the
    slowest kernels in the step (job 33527, h20, 4 GPUs, per step): the
    gathers 169.8 ms as `indexSelectLargeIndex`, the scatter 158.6 ms as
    `indexFuncLargeIndex` plus 86.0 ms of fp32-conversion copies -- against
    ~15 and 36.9 ms for the same work inductor generated when the bank path sat
    inside the compiled block (job 33502). Slow kernels on a side stream are
    not free: they took memory bandwidth and SMs from the compute stream, whose
    GEMMs ran 178.6 ms slower. Not compiled on CPU, where inductor's C++
    backend miscompiles exactly this kind of scatter (§1.18 of the report).
    """
    if device.type != "cuda":
        return fn
    compiled = _COMPILED_BANK_KERNELS.get(fn)
    if compiled is None:
        compiled = _COMPILED_BANK_KERNELS[fn] = torch.compile(fn, dynamic=False)
    return compiled


class _ExchangedPair(torch.autograd.Function):
    """O39: one layer's q and k rows for this rank's tokens, exchanged ahead of
    the layer.

    The forward is `_FusedBankGather`'s without the reassembly: gather every
    rank's tokens from the locally owned columns of both banks into one buffer,
    one all-to-all, and return the receive buffer as it lands -- rank-major,
    rows `[j*n, (j+1)*n)` holding this rank's n tokens on rank j's columns,
    q's `w` columns then k's. The outputs are the two column halves of that
    buffer, as views. The consumer reassembles each through a view inside its
    compiled region (`AMEAttention`, `_rank_major_to_heads`), where the
    transpose fuses into the normalize that reads it instead of being a copy of
    its own; measured before O39 (job 33502), that copy was
    `triton_poi_fused_stack_10`, 135.6 ms per step at h20 on 4 GPUs.

    It is called on the bank stream (`BankPrefetcher`), and autograd runs a
    Function's backward on the stream its forward ran on, so the gradient
    exchange and the scatter into `main_grad` run there too: off the stream the
    rest of the backward is computing on.
    """

    @staticmethod
    def forward(ctx, q_weight, k_weight, gathered_ids, pair, inverse=None):
        bank = pair.q
        gather = _bank_kernel(_gather_pair, gathered_ids.device)
        # Cast for O38(b), a no-op otherwise.
        local = gather(gathered_ids, q_weight, k_weight, bank.compute_dtype)
        out = torch.empty_like(local)
        dist.all_to_all_single(out, local, group=bank.mp_group)
        width = q_weight.shape[1]
        ctx.pair = pair
        ctx.width = width
        ctx.has_inverse = inverse is not None
        if inverse is not None:
            # O47 v2: `out` holds one row per DISTINCT id per rank; expand to
            # tokens here, on the bank stream, so the compiled block sees the
            # token-level layout it always had.
            mp = pair.q.mp_size
            out = _bank_kernel(_expand_rows, out.device)(out, inverse, mp)
            ctx.save_for_backward(gathered_ids, inverse)
        else:
            ctx.save_for_backward(gathered_ids)
        return out[:, :width], out[:, width:]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k):
        gathered_ids, *rest = ctx.saved_tensors
        inverse = rest[0] if ctx.has_inverse else None
        pair, width = ctx.pair, ctx.width
        grads = []
        for grad in (grad_q, grad_k):
            if grad is None:
                grad = torch.zeros(
                    gathered_ids.numel(),
                    width,
                    dtype=pair.q.compute_dtype,
                    device=gathered_ids.device,
                )
            elif grad.is_cuda:
                # Produced on the compute stream, read here on the bank stream.
                # Without this the compute stream may reuse the block as soon
                # as autograd drops its reference, while the copy below is
                # still reading it.
                grad.record_stream(torch.cuda.current_stream(grad.device))
            grads.append(grad)
        send = torch.cat(grads, dim=1)
        if inverse is not None:
            # O47 v2: sum each distinct id's token gradients, in fp32, before
            # the exchange -- not inside the compiled block, where the same
            # reduction became bf16 atomics (job 34394, 34.55%).
            send = _bank_kernel(_reduce_rows, send.device)(
                send, inverse, pair.q.mp_size, gathered_ids.numel() // pair.q.mp_size
            )
        local = torch.empty_like(send)
        # Mirror of the forward: chunk j (my tokens, rank j's columns) goes back
        # to rank j; chunk i received is rank i's tokens on MY columns, which is
        # row-aligned with `gathered_ids`.
        dist.all_to_all_single(local, send, group=pair.q.mp_group)
        scatter = _bank_kernel(_scatter_pair, gathered_ids.device)
        scatter(pair.q.main_grad, pair.k.main_grad, gathered_ids, local, width)
        return None, None, None, None, None


def _distinct(ids, cap_fraction):
    """`(unique, inverse)` for a 1-D id tensor, with `unique` padded to a fixed
    `ceil(cap_fraction * n)` slots and no host synchronisation.

    Sort, mark first occurrences, cumulative-sum them into slots, then scatter:
    every position writes its id into its slot, and all writers of a slot write
    the same id. Unused slots hold 0, a real row that no token's inverse points
    at, so it is gathered and exchanged but never read, and its gradient is
    zero. `inverse[i]` is position i's slot: `unique[inverse] == ids`.
    """
    n = ids.numel()
    cap = max(1, math.ceil(cap_fraction * n))
    sorted_ids, order = torch.sort(ids)
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    slot = torch.cumsum(first, 0) - 1
    torch._assert_async(
        slot[-1] < cap,
        "bank dedup: more distinct ids in the micro-batch than "
        "trainer.bank_dedup_cap reserves; raise the cap",
    )
    slot = slot.clamp(max=cap - 1)
    unique = torch.zeros(cap, dtype=ids.dtype, device=ids.device)
    unique.scatter_(0, slot, sorted_ids)
    inverse = torch.empty_like(ids)
    inverse.scatter_(0, order, slot)
    return unique, inverse


@contextlib.contextmanager
def _on_stream(stream, wait=False):
    """Run the body on `stream` (None: the current stream, CPU or no CUDA).
    `wait` first makes it wait for everything already queued on the current
    stream."""
    if stream is None:
        yield
        return
    if wait:
        stream.wait_stream(torch.cuda.current_stream(stream.device))
    with torch.cuda.stream(stream):
        yield


class BankPrefetcher:
    """O39: the bank exchange, taken off the critical path.

    AME's queries and keys are gathers by token id, so a layer's q and k rows
    depend on nothing the layers below it compute -- only on the ids, which are
    known before the first layer runs. Inside each compiled block the exchange
    sat on the critical path all the same: the block waited for its own
    all-to-all, and the profile of the h20/4-GPU best (job 33502) shows that
    all-to-all 196.6 ms per step and exposed for all of it, no compute kernel
    running beside it.

    This issues each layer's gather and all-to-all on a separate CUDA stream,
    `depth` layers ahead of the layer that consumes them; the encoder takes a
    layer's rows just before calling the block and the compute stream waits
    only on that layer's event. The backward exchange and scatter run on the
    same stream (see `_ExchangedPair`), so the compute stream never waits for
    them; `finish()` joins the two before `main_grad` is read.

    One stream, deliberately: the forward exchanges and the backward ones go to
    one process group, and NCCL needs every rank to issue a group's
    collectives in the same order. A single stream keeps that order the CPU's,
    which is the same on every rank.
    """

    def __init__(self, controller, encoder, depth, cpu_lead=None, dedup_cap=None):
        if depth < 1:
            raise ValueError(f"bank prefetch depth must be >= 1, got {depth}")
        self.controller = controller
        self.depth = int(depth)
        # O44: at most this many blocks between the CPU and the compute stream
        # (None: unbounded). Each `take` records an event on the compute stream
        # and the CPU waits for the one `cpu_lead` blocks back. Without a bound,
        # once the blocks compiled whole (O41, O42) the CPU ran arbitrarily far
        # ahead, and the cross-stream buffers this class hands over cannot be
        # recycled until the GPU catches up.
        self.cpu_lead = int(cpu_lead) if cpu_lead else None
        self._lead_events = collections.deque()
        # O47: exchange each micro-batch's DISTINCT ids only. A micro-batch of
        # 16 x 2048 FineWeb tokens holds ~22% distinct ids (job 34335), so the
        # all-to-all, the owner-side gathers and the scatters shrink ~4.5x.
        # `dedup_cap` is the fraction of tokens reserved for distinct ids --
        # fixed so that no size ever has to come back to the host; more
        # distinct ids than that fails a device-side assert instead of
        # corrupting rows. None: off.
        if dedup_cap is not None and not 0 < dedup_cap <= 1:
            raise ValueError(f"bank dedup cap must be in (0, 1], got {dedup_cap}")
        self.dedup_cap = dedup_cap
        self._inverse = None
        # One slot per block, in the order the encoder calls them: the block's
        # fused q/k pair, or None for a block whose attention has no banks.
        self.slots = []
        for block in encoder.blocks:
            layer = getattr(getattr(block, "attention_layer", None), "layer", None)
            pair = getattr(layer, "qk_bank", None)
            self.slots.append(pair if isinstance(pair, FusedBankPair) else None)
        if not any(self.slots):
            raise ValueError("bank prefetch found no fused q/k pair in any block")
        self.stream = None
        self._pending = {}
        self._gathered = None

    def _stream_for(self, device):
        if device.type != "cuda":
            return None
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        return self.stream

    def begin(self, token_ids):
        """Start one forward: all-gather this micro-batch's ids over the
        mp-group once (not once per layer, S6) and queue the first `depth`
        layers' exchanges."""
        stream = self._stream_for(token_ids.device)
        self._pending.clear()
        self._lead_events.clear()
        group = self.controller.mp_group
        mp_size = self.controller.mp_size
        # wait=True: the ids, and the compute weights the last optimizer step
        # wrote, were produced on the compute stream. Later layers' exchanges
        # do NOT wait on it again -- that would queue them behind the compute
        # they are meant to overlap.
        with _on_stream(stream, wait=True):
            ids_flat = token_ids.reshape(-1).contiguous()
            send, inverse = ids_flat, None
            if self.dedup_cap:
                send, inverse = _distinct(ids_flat, self.dedup_cap)
            gathered = torch.empty(
                mp_size * send.numel(), dtype=send.dtype, device=send.device
            )
            dist.all_gather_into_tensor(gathered, send, group=group)
        if stream is not None:
            ids_flat.record_stream(stream)
        self._gathered = gathered
        self._inverse = inverse  # used on the bank stream only (O47 v2)
        for index in range(len(self.slots)):
            if len(self._pending) >= self.depth:
                break
            self._issue(index)

    def _issue(self, index):
        pair = self.slots[index]
        if pair is None or index in self._pending:
            return
        stream = self.stream if self._gathered.is_cuda else None
        with _on_stream(stream):
            q, k = _ExchangedPair.apply(
                pair.q.weight, pair.k.weight, self._gathered, pair, self._inverse
            )
            event = None
            if stream is not None:
                event = torch.cuda.Event()
                event.record(stream)
        self._pending[index] = (q, k, event)

    def take(self, index):
        """Block `index`'s rows, `(q_rows, k_rows)`, or None if the block has
        no banks. Queues the exchange `depth` blocks ahead first, so the bank
        stream always has the next layers in flight."""
        ahead = 0
        for later in range(index + 1, len(self.slots)):
            if self.slots[later] is not None:
                ahead += 1
                if ahead > self.depth:
                    break
                self._issue(later)
        if self.cpu_lead and self.stream is not None:
            mark = torch.cuda.Event()
            mark.record(torch.cuda.current_stream(self.stream.device))
            self._lead_events.append(mark)
            if len(self._lead_events) > self.cpu_lead:
                self._lead_events.popleft().synchronize()
        if self.slots[index] is None:
            return None
        if index not in self._pending:
            self._issue(index)
        q, k, event = self._pending.pop(index)
        if event is not None:
            current = torch.cuda.current_stream(q.device)
            current.wait_event(event)
            # Allocated on the bank stream, consumed (and saved for backward)
            # on the compute stream.
            q.record_stream(current)
            k.record_stream(current)
        return q, k

    def finish(self):
        """Make the compute stream wait for every bank-stream backward still
        in flight. Called before anything reads `main_grad`."""
        if self.stream is not None:
            torch.cuda.current_stream(self.stream.device).wait_stream(self.stream)


def _find_banks(model):
    """Every (module, attribute) pair holding an id-indexed bank.

    Duck-typed on the attribute names rather than on `isinstance(AMEAttention)`
    so a variant layer that keeps the same contract is picked up too, and so
    this module does not import the attention layer it rewrites.
    """
    found = []
    for module in model.modules():
        for name in ("q_bank", "k_bank"):
            bank = getattr(module, name, None)
            if isinstance(bank, nn.Embedding):
                found.append((module, name, bank))
    return found


class MemoryParallelController:
    """Owns every Q/K bank: its sharding, its gradient, and its optimizer.

    Built in `main.py` *before* `fully_shard`, so FSDP2 sees `ShardedBank`
    parameters it has been told to ignore rather than `nn.Embedding` weights it
    would shard a second, incompatible way.
    """

    def __init__(
        self,
        model,
        mem_parallel_size,
        compute_dtype=torch.bfloat16,
        fuse_qk=True,
        prefetch_depth=None,
        compute_copy=True,
        nccl_max_ctas=None,
        default_group=False,
        prefetch_cpu_lead=None,
        dedup_cap=None,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.compute_dtype = compute_dtype
        # O38(b), see ShardedBank. Only meaningful when the compute dtype is
        # narrower than the master's; otherwise there is no copy to drop.
        self.compute_copy = bool(compute_copy) or compute_dtype.itemsize >= 4

        mp_size = int(mem_parallel_size)
        if mp_size < 1:
            raise ValueError(f"mem_parallel_size must be >= 1, got {mp_size}")
        if self.world_size % mp_size:
            raise ValueError(
                f"mem_parallel_size {mp_size} does not divide world_size "
                f"{self.world_size}"
            )
        self.mp_size = mp_size
        self.mp_rank = self.rank % mp_size
        self.dp_size = self.world_size // mp_size
        # O39: with the exchange on a side stream its NCCL kernels run beside
        # the compute stream's and take SMs from it. Capping the mp-group's
        # CTAs trades exchange bandwidth (now off the critical path) for
        # compute. None keeps NCCL's own choice, as every earlier run did.
        self.nccl_max_ctas = nccl_max_ctas
        # O43: at mem_parallel_size == world_size the mp-group is every rank,
        # i.e. the same ranks as the default group FSDP2's 1-D mesh uses
        # (torch/distributed/device_mesh.py). On that group the bank exchange
        # shares FSDP2's NCCL communicator and stream, so every collective runs
        # in the CPU's issue order on every rank. On its own group, with O39
        # issuing it from a side stream and nothing keeping the CPU in step with
        # the GPU, the two communicators' kernels reached the GPUs in different
        # orders on different ranks and spun waiting for each other (job 33639:
        # forward all-gather 1985 ms busy, all-to-all 1997 ms, 1.8 s/step idle).
        self.default_group = bool(default_group) and mp_size == self.world_size
        self.mp_group, self.dp_group = self._build_groups()

        self.banks = []
        for module, name, embedding in _find_banks(model):
            bank = ShardedBank(embedding.num_embeddings, embedding.embedding_dim, self)
            bank.to(embedding.weight.device)
            bank.load_from(embedding)
            setattr(module, name, bank)
            self.banks.append(bank)
        if not self.banks:
            raise ValueError(
                "mem_parallel_size was set but the model has no q_bank/k_bank "
                "embeddings -- this controller only knows how to run AME banks"
            )

        # D2: batch each layer's q and k into one gather/all-to-all/permute.
        # Purely a data-path change -- the ShardedBank modules above stay the
        # owners of every parameter, gradient and optimizer slot.
        self.fused_pairs = []
        if fuse_qk:
            for module in model.modules():
                q = getattr(module, "q_bank", None)
                k = getattr(module, "k_bank", None)
                if isinstance(q, ShardedBank) and isinstance(k, ShardedBank):
                    if not hasattr(module, "qk_bank"):
                        raise ValueError(
                            f"{type(module).__name__} holds sharded banks but has "
                            "no `qk_bank` hook; fuse_qk=False runs the unfused path"
                        )
                    pair = FusedBankPair(q, k)
                    module.qk_bank = pair
                    self.fused_pairs.append(pair)

        # O39: the exchange issued ahead of each layer on its own stream. Only
        # where there is an exchange to move -- at mem_parallel_size 1 the
        # gather is local, already fused into the compiled block, and taking it
        # out would only add a materialised copy.
        self.prefetcher = None
        if prefetch_depth and mp_size > 1:
            if not self.fused_pairs:
                raise ValueError("bank prefetch needs fuse_qk=True")
            encoders = [
                m
                for m in model.modules()
                if hasattr(m, "bank_prefetcher") and hasattr(m, "blocks")
            ]
            if len(encoders) != 1:
                raise ValueError(
                    f"bank prefetch needs exactly one encoder exposing "
                    f"`bank_prefetcher`, found {len(encoders)}"
                )
            self.prefetcher = BankPrefetcher(
                self,
                encoders[0],
                prefetch_depth,
                cpu_lead=prefetch_cpu_lead,
                dedup_cap=dedup_cap,
            )
            encoders[0].bank_prefetcher = self.prefetcher
        elif prefetch_depth:
            logger.info(
                "[memory-parallel] bank prefetch requested at mem_parallel_size 1; "
                "there is no exchange to move, so it stays off"
            )

        self.optimizer = None
        total = sum(b.weight.numel() for b in self.banks)
        logger.info(
            f"[memory-parallel] {len(self.banks)} banks "
            f"({len(self.fused_pairs)} fused q/k pairs), "
            f"mem_parallel_size={mp_size}, "
            f"mp_rank={self.mp_rank}, dp_size={self.dp_size}, "
            f"{total / 1e9:.3f}B bank parameters per rank, "
            f"prefetch depth {self.prefetcher.depth if self.prefetcher else 0}"
        )
        model._memory_parallel = self

    # ---- O6: the two process groups ----------------------------------------

    def _build_groups(self):
        """Contiguous mp-groups, strided dp-groups. Both are collective calls:
        every rank walks the whole loop, keeping only the group it is in."""
        if not dist.is_initialized() or self.world_size == 1:
            return None, None

        if self.default_group:
            return None, None  # None is the default group in every collective
        mp_group = None
        options = None
        if self.nccl_max_ctas and dist.get_backend() == "nccl":
            options = dist.ProcessGroupNCCL.Options()
            options.config.max_ctas = int(self.nccl_max_ctas)
        if self.mp_size > 1:
            for start in range(0, self.world_size, self.mp_size):
                ranks = list(range(start, start + self.mp_size))
                group = dist.new_group(ranks, pg_options=options)
                if self.rank in ranks:
                    mp_group = group

        dp_group = None
        if 1 < self.dp_size < self.world_size:
            for offset in range(self.mp_size):
                ranks = list(range(offset, self.world_size, self.mp_size))
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    dp_group = group
        # dp_size == world_size (mp == 1) leaves dp_group None, which means the
        # default group -- every rank holds the same slice and all of them must
        # take part. dp_size == 1 means nothing to reduce; `reduce()` checks.
        return mp_group, dp_group

    # ---- O1: what FSDP2 and the backbone optimizer must not touch -----------

    def ignored_parameters(self):
        return {bank.weight for bank in self.banks}

    def build_optimizer(self, lr, weight_decay=0.0, **kwargs):
        """A dedicated AdamW over the fp32 masters.

        `weight_decay` defaults to 0.0 because `AMEAttention` already names both
        banks in `no_weight_decay_parameters` -- the backbone optimizer gives
        them a wd=0 group, and this must reproduce that, not quietly reinstate
        decay on 2.576B parameters.
        """
        for bank in self.banks:
            bank.materialize(bank.weight.device)
        masters = [bank.master for bank in self.banks]
        self.optimizer = torch.optim.AdamW(
            masters, lr=lr, weight_decay=weight_decay, **kwargs
        )
        return self.optimizer

    # ---- O5: the three hook points -----------------------------------------

    def materialize(self):
        for bank in self.banks:
            bank.materialize(bank.weight.device)

    @torch.no_grad()
    def reduce(self):
        """O2 + O8: one all-reduce per optimizer step, divided by WORLD size."""
        if self.prefetcher is not None:
            # O39: the last micro-step's bank backward may still be running on
            # the bank stream.
            self.prefetcher.finish()
        if self.dp_size > 1:
            grads = [bank.main_grad for bank in self.banks]
            for grad in grads:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dp_group)
        if self.world_size > 1:
            # O8. Not dp_size: the forward's id all-gather already summed the
            # mp-group's contributions into this slice.
            for bank in self.banks:
                bank.main_grad.div_(self.world_size)

    @torch.no_grad()
    def local_grad_norm_sq(self, device=None):
        """O4: this rank's contribution to the JOINT (backbone + bank) norm.

        Only the first mp-block of ranks contributes. Every slice is replicated
        across its dp-subgroup after `reduce()`, so letting all of them report
        would count each slice `dp_size` times and inflate the global norm by
        `sqrt(dp_size)` -- which is exactly the kind of silent constant factor
        the grad-norm trace exists to catch.
        """
        reference = self.banks[0].main_grad
        device = device or reference.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        if self.rank < self.mp_size:
            for bank in self.banks:
                # vector_norm, not `.float().pow(2).sum()`: the latter
                # materialises a full-size fp32 copy of every bank gradient --
                # 10.3 GB at mem_parallel_size 1, 2.6 GB at 4 -- to compute one
                # scalar. The reduction reads the buffer and allocates nothing.
                norm = torch.linalg.vector_norm(bank.main_grad, 2.0)
                total = total + norm.float().to(device) ** 2
        return total

    @torch.no_grad()
    def clip_(self, max_norm, total_norm):
        """Apply the same coefficient `torch.nn.utils.clip_grads_with_norm_`
        applies to the backbone, computed from the same joint `total_norm`."""
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for bank in self.banks:
            bank.main_grad.mul_(clip_coef_clamped.to(bank.main_grad.device))

    @torch.no_grad()
    def step(self, lr):
        """O3: step the masters, then refresh the compute parameters."""
        if self.optimizer is None:
            raise RuntimeError("build_optimizer() was never called")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        for bank in self.banks:
            bank.master.grad = bank.main_grad
        self.optimizer.step()
        for bank in self.banks:
            bank.master.grad = None
            if bank.compute_copy:
                bank.weight.copy_(bank.master.to(bank.weight.dtype))

    @torch.no_grad()
    def zero_grad(self):
        for bank in self.banks:
            bank.main_grad.zero_()

    # ---- bookkeeping --------------------------------------------------------

    def log_metrics(self, metric_logger, lr):
        """O5 hook 3: the bank LR is not the backbone's by construction (it is
        copied each step), so it is logged as its own series rather than assumed."""
        metric_logger.log("train/bank_lr", lr)

    SIDECAR = "memory_parallel_rank{rank}.pt"

    def state_dict(self):
        return {
            "masters": [bank.master for bank in self.banks],
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "mem_parallel_size": self.mp_size,
            "mp_rank": self.mp_rank,
            "world_size": self.world_size,
        }

    def save(self, folder):
        """One file per rank beside the model checkpoint.

        Rank-local rather than gathered, for the reason pkmf reached the same
        way round (O30): gathering 2.576B parameters to rank 0 is minutes of
        wall time and a 10 GB write, and the only thing it buys is a layout that
        this loader would have to re-split anyway.
        """
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, self.SIDECAR.format(rank=self.rank))
        torch.save(self.state_dict(), path)
        logger.info(f"[memory-parallel] wrote bank state to {path}")

    def load(self, folder):
        path = os.path.join(folder, self.SIDECAR.format(rank=self.rank))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no memory-parallel bank state at {path}. The model has a bank "
                "controller, so the checkpoint must carry one sidecar per rank; "
                "a checkpoint written without mem_parallel_size cannot be "
                "resumed into one."
            )
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        logger.info(f"[memory-parallel] loaded bank state from {path}")

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("world_size", self.world_size) != self.world_size:
            raise ValueError(
                "bank checkpoint was written at world_size "
                f"{state['world_size']}, this run uses {self.world_size}; the "
                "sidecars are per rank and do not resplit"
            )
        if state["mem_parallel_size"] != self.mp_size:
            raise ValueError(
                "bank checkpoint was written at mem_parallel_size "
                f"{state['mem_parallel_size']}, this run uses {self.mp_size}; "
                "resharding bank state across mem_parallel_size is not implemented"
            )
        for bank, master in zip(self.banks, state["masters"]):
            bank.materialize(bank.weight.device)
            bank.master.copy_(master.to(bank.master.device))
            if bank.compute_copy:
                bank.weight.copy_(bank.master.to(bank.weight.dtype))
        if state["optimizer"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])


def get_controller(model):
    """The controller a model was given, or None.

    Survives `fully_shard`, which returns the module object it was handed, and
    `DistributedDataParallel`, which does not -- DDP returns a wrapper and
    `nn.Module.__getattr__` does not forward unknown attributes to `.module`, so
    a plain getattr on a DDP-wrapped model silently reports "no controller" and
    the banks would never be stepped.
    """
    found = getattr(model, "_memory_parallel", None)
    if found is None:
        inner = getattr(model, "module", None)
        if inner is not None:
            found = getattr(inner, "_memory_parallel", None)
    return found
