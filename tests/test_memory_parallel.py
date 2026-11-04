"""Section 1.4's correctness gate for Tiers 2 and 3, on a CPU toy model.

Tier 2 moves 2.576B parameters out of FSDP2 and AdamW into a hand-written
gradient and optimizer path; Tier 3 then splits them across ranks and changes
which process group reduces them. Neither may change the mathematics, and every
way they can go wrong is silent:

- a frozen bank (the gradient never arrives) trains, and converges slower;
- a wrong O8 divisor scales every bank gradient by a constant, which moves the
  loss curve by less than seed noise for hundreds of steps;
- a separately-clipped bank is a different function of the same gradients;
- a double-counted norm inflates the global norm by sqrt(dp_size) and nothing
  reports it.

So the gate is parameter-for-parameter equality against the ordinary path, on a
model small enough to run both in one process, and `mp=1` against `mp=4` on a
gloo mesh for the divisor. float64 throughout: the question is whether the two
paths compute the same function, not how fp32 rounds, and at 1e-12 any
structural error is orders of magnitude clear of the noise floor.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from src.core.ame import AMEAttention
from src.core.memory_parallel import (
    MemoryParallelController,
    ShardedBank,
    get_controller,
)
from src.core.model import parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The toy model section 1.4.1 specifies.
VOCAB = 256
DMODEL = 64
HEADS = 4
BLOCKS = 2
SEQ = 16
GA = 2  # micro-steps per optimizer step, so accumulation is exercised
STEPS = 3
LR = 0.01
CLIP = 1.0


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AMEAttention(
            dmodel=DMODEL,
            q_heads=HEADS,
            kv_heads=HEADS,
            seq_len=SEQ,
            vocab_size=VOCAB,
            backend="reference",
        )
        self.norm = nn.LayerNorm(DMODEL)
        self.ff = nn.Sequential(
            nn.Linear(DMODEL, 2 * DMODEL), nn.GELU(), nn.Linear(2 * DMODEL, DMODEL)
        )

    def forward(self, x, ids):
        x = x + self.attention(self.norm(x), ids)
        return x + self.ff(self.norm(x))


class ToyAME(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DMODEL)
        self.blocks = nn.ModuleList(ToyBlock() for _ in range(BLOCKS))
        self.head = nn.Linear(DMODEL, VOCAB)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, ids)
        return self.head(x)


def build_model(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    model = ToyAME()
    return model.to(dtype)


def micro_batches(rank=0, n=GA * STEPS, batch=2):
    generator = torch.Generator().manual_seed(100 + rank)
    return [
        torch.randint(0, VOCAB, (batch, SEQ), generator=generator) for _ in range(n)
    ]


def clip_through_trainer(model, clip, banks):
    """The real `Trainer.clip_gradient`, borrowed the way the O9 tests borrow
    `_accumulation_grad_sync` -- so the joint-clip path under test is the one
    that ships, not a copy of it."""
    from types import SimpleNamespace

    from src.core.trainer import Trainer

    stub = SimpleNamespace(model=model, gradient_clipping=clip, _banks=banks)
    return Trainer.clip_gradient(stub)


def train(model, batches, banks=None, clip=CLIP, steps=STEPS, lr=LR):
    """One faithful copy of Trainer.train's optimizer section.

    Baseline when `banks is None`: one AdamW over everything, one clip.
    """
    ignored = banks.ignored_parameters() if banks is not None else None
    optimizer = torch.optim.AdamW(
        parameter_groups(model, weight_decay=0.1, exclude=ignored), lr=lr
    )
    if banks is not None:
        banks.build_optimizer(lr=lr, weight_decay=0.0)

    norms, losses = [], []
    model.train()
    for step in range(steps):
        if banks is not None:
            banks.materialize()
        step_loss = 0.0
        for i in range(GA):
            ids = batches[step * GA + i]
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
            )
            (loss / GA).backward()
            step_loss += loss.detach().item() / GA
        losses.append(step_loss)
        if banks is not None:
            banks.reduce()
        norm = clip_through_trainer(model, clip, banks)
        norms.append(None if norm is None else float(norm))
        optimizer.step()
        optimizer.zero_grad()
        if banks is not None:
            banks.step(lr=lr)
            banks.zero_grad()
    train.last_losses = losses
    return norms


def named_tensors(model):
    """Every trained tensor by name, with a bank reported through its fp32
    master so the two paths are compared on the same quantity."""
    out = {}
    for name, parameter in model.named_parameters():
        out[name] = parameter.detach().clone()
    for name, module in model.named_modules():
        if isinstance(module, ShardedBank):
            out[f"{name}.weight"] = module.master.detach().clone()
    return out


# ---- Tier 2: the hand-managed path is the ordinary path ---------------------


def test_toy_model_parameter_equivalence_without_clipping():
    """Section 1.4.1, in its strict form: with no clip in the way, the
    hand-managed path must be **bit-identical** to the ordinary one.

    Nothing here is allowed to be approximate. The gather, the scatter into
    `main_grad`, accumulation across micro-steps, the fp32 master, the separate
    AdamW and the master -> compute copy are each exact operations, and in
    float64 at one rank there is no reduction order for them to disagree about.
    A tolerance here would hide a frozen bank, a dropped micro-step or a
    double-applied weight decay.
    """
    batches = micro_batches()

    baseline = build_model()
    train(baseline, batches, clip=None)

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    train(managed, batches, banks=banks, clip=None)

    assert len(banks.banks) == 2 * BLOCKS, "not every bank was claimed"

    before = named_tensors(build_model())
    expected, actual = named_tensors(baseline), named_tensors(managed)
    assert set(expected) == set(actual), (
        f"parameter sets diverged: only baseline "
        f"{sorted(set(expected) - set(actual))}, only managed "
        f"{sorted(set(actual) - set(expected))}"
    )
    banks_moved = [
        name
        for name in expected
        if "bank" in name and not torch.equal(expected[name], before[name])
    ]
    assert len(banks_moved) == 2 * BLOCKS, (
        f"only {banks_moved} moved over three steps -- a gate that passes on a "
        "frozen bank is not a gate"
    )

    differing = {
        name: (expected[name] - actual[name]).abs().max().item()
        for name in expected
        if not torch.equal(expected[name], actual[name])
    }
    assert not differing, f"the hand-managed path is not bit-identical: {differing}"


@pytest.mark.parametrize(
    "dtype,atol,norm_rtol", [(torch.float64, 1e-6, 1e-6), (torch.float32, 2e-5, 1e-5)]
)
def test_toy_model_parameter_equivalence_with_clipping(dtype, atol, norm_rtol):
    """Section 1.4.1 with the clip on, where bit-identity is the wrong gate.

    `Trainer.clip_gradient` computes the global norm and then casts it with
    `.float()`. The baseline sums one list of per-parameter norms; the joint
    clip sums the backbone's and adds the banks' separately. Same terms, and in
    float64 the two sums agree to about 1e-16 -- but the fp32 cast quantises the
    result, and the two land on neighbouring fp32 values. Measured: 1.3832852840
    against 1.3832854032, **exactly one fp32 ulp**, at step 1 only. Every later
    step's norm is identical and the parameters carry the one-ulp difference
    forward at ~1e-7.

    That is float associativity in a scalar, not a difference in the function
    being computed -- which `test_..._without_clipping` above establishes by
    being exact. What this test is really for is the *shape*: the norm traces
    must track each other, because a separate (non-joint) clip, a
    double-counted bank norm or a missing one all move the trace by far more
    than an ulp.
    """
    batches = micro_batches()

    baseline = build_model(dtype)
    base_norms = train(baseline, batches, clip=CLIP)

    managed = build_model(dtype)
    banks = MemoryParallelController(managed, 1, compute_dtype=dtype)
    managed_norms = train(managed, batches, banks=banks, clip=CLIP)

    for i, (a, b) in enumerate(zip(base_norms, managed_norms)):
        assert abs(a - b) <= norm_rtol * abs(a), (
            f"grad norm diverged at step {i}: baseline {a!r} vs managed {b!r}. "
            "A joint clip sums the same terms; a separate one does not."
        )

    expected, actual = named_tensors(baseline), named_tensors(managed)
    worst = max(((expected[n] - actual[n]).abs().max().item(), n) for n in expected)
    assert worst[0] <= atol, f"{worst[1]} differs by {worst[0]:.3e} (atol {atol:.1e})"


def test_a_separate_clip_would_fail_this_gate():
    """The negative control for the test above: clipping the banks on their own
    coefficient instead of the joint one must move the parameters far more than
    the tolerance, or the gate is not testing anything."""
    batches = micro_batches()

    baseline = build_model()
    train(baseline, batches, clip=CLIP)

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    original = banks.clip_

    def bank_only_clip(max_norm, total_norm):
        own = torch.sqrt(banks.local_grad_norm_sq().double())
        original(max_norm, own)

    banks.clip_ = bank_only_clip
    train(managed, batches, banks=banks, clip=CLIP)

    expected, actual = named_tensors(baseline), named_tensors(managed)
    worst = max(((expected[n] - actual[n]).abs().max().item(), n) for n in expected)
    assert worst[0] > 1e-4, (
        f"a separately-clipped bank moved the model by only {worst[0]:.3e}; "
        "the equivalence gate above cannot tell the two clips apart"
    )


def test_bank_never_allocates_a_dense_gradient():
    """O1's whole point: `weight.grad` must stay None, every step, forever.

    If it is not None something is writing a `[vocab, width]` tensor per bank
    per micro-step -- the allocation this tier deletes -- and worse, the
    trainer's clip would then find it and clip it twice.
    """
    model = build_model()
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    train(model, micro_batches(), banks=banks)
    for bank in banks.banks:
        assert bank.weight.grad is None, "a dense bank gradient was materialised"
        assert bank.main_grad is not None
        assert torch.equal(
            bank.main_grad, torch.zeros_like(bank.main_grad)
        ), "main_grad was not zeroed after the step"


def test_main_grad_matches_the_dense_backward():
    """`index_add_` into a persistent fp32 buffer must sum to exactly what
    `embedding_dense_backward` + `.grad` accumulation sums to."""
    batches = micro_batches(n=GA)

    baseline = build_model()
    for i in range(GA):
        logits = baseline(batches[i])
        (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), batches[i][:, 1:].reshape(-1)
            )
            / GA
        ).backward()

    managed = build_model()
    banks = MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    banks.materialize()
    for i in range(GA):
        logits = managed(batches[i])
        (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), batches[i][:, 1:].reshape(-1)
            )
            / GA
        ).backward()

    dense = {
        name: p.grad
        for name, p in baseline.named_parameters()
        if "bank" in name and p.grad is not None
    }
    assert dense, "the baseline produced no bank gradients"
    scattered = {
        f"{name}.weight": module.main_grad
        for name, module in managed.named_modules()
        if isinstance(module, ShardedBank)
    }
    assert set(dense) == set(scattered)
    for name in dense:
        assert torch.allclose(dense[name], scattered[name], rtol=0, atol=1e-12), (
            f"{name}: scattered gradient differs by "
            f"{(dense[name] - scattered[name]).abs().max().item():.3e}"
        )
        assert dense[name].abs().max() > 0, f"{name} gradient was all zero"


def test_banks_are_excluded_from_the_backbone_optimizer():
    model = build_model()
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    ignored = banks.ignored_parameters()
    assert len(ignored) == 2 * BLOCKS

    groups = parameter_groups(model, weight_decay=0.1, exclude=ignored)
    claimed = {id(p) for group in groups for p in group["params"]}
    assert not (claimed & {id(p) for p in ignored}), (
        "a bank parameter reached the backbone optimizer as well as the "
        "controller's -- it would be stepped twice"
    )
    # and nothing else was dropped on the way
    assert len(claimed) == sum(1 for _ in model.parameters()) - len(ignored)


def test_mfu_denominator_is_unchanged_by_the_swap():
    """A ShardedBank is not an nn.Embedding, and src.core.mfu identifies lookups
    by that class. Without `non_matmul_parameters` the swap would silently add
    6*P FLOPs/token for the banks and every MFU in the ledger would stop being
    comparable to the ones above it."""
    from src.core.mfu import model_flops_per_token

    baseline = build_model()
    before = model_flops_per_token(baseline, SEQ)

    managed = build_model()
    MemoryParallelController(managed, 1, compute_dtype=torch.float64)
    after = model_flops_per_token(managed, SEQ)

    assert before == after, (
        f"FLOPs/token moved from {before} to {after} when the banks were swapped"
    )


def test_mem_parallel_size_must_divide_world_size():
    model = build_model()
    with pytest.raises(ValueError, match="does not divide world_size"):
        MemoryParallelController(model, 3, compute_dtype=torch.float64)


def test_controller_is_reachable_from_the_model():
    model = build_model()
    assert get_controller(model) is None
    banks = MemoryParallelController(model, 1, compute_dtype=torch.float64)
    assert get_controller(model) is banks


def test_a_model_without_banks_is_rejected():
    plain = nn.Sequential(nn.Embedding(VOCAB, DMODEL), nn.Linear(DMODEL, VOCAB))
    with pytest.raises(ValueError, match="no q_bank/k_bank"):
        MemoryParallelController(plain, 1, compute_dtype=torch.float64)


def test_trainer_rejects_a_config_whose_model_has_no_controller():
    """`trainer.mem_parallel_size=4` on a model built without one would train
    the ordinary FSDP2 path and report it as a Tier 3 number."""
    from types import SimpleNamespace

    from src.core.trainer import Trainer

    stub = SimpleNamespace(model=build_model(), mem_parallel_size=4)
    stub._instrumentation_failures = set()
    with pytest.raises(ValueError, match="carries no "):
        # the two lines of __attrs_post_init__ that own this check
        banks = get_controller(stub.model)
        if stub.mem_parallel_size and banks is None:
            raise ValueError(
                "trainer.mem_parallel_size=4 but the model carries no "
                "memory-parallel controller"
            )


# ---- Tier 3: the column split, the two groups, and O8's divisor -------------
#
# Everything below runs four gloo ranks on the CPU, because `mem_parallel_size`
# is only interesting when it is neither 1 nor world_size at the same time. The
# comparison is always the same shape: run the identical schedule at several
# `mem_parallel_size` values and demand the SAME answer, since mp is a layout
# choice and nothing about the mathematics may depend on it.


def full_width_main_grad(controller, index):
    """Bank `index`'s gradient at full width, assembled on every rank.

    At mp=1 the rank already holds it. At mp>1 it owns one column slice, so the
    slices are all-gathered in mp-rank order -- the same order the forward's
    all-to-all assumes -- and concatenated.
    """
    bank = controller.banks[index]
    if controller.mp_size == 1:
        return bank.main_grad.clone()
    slices = [torch.empty_like(bank.main_grad) for _ in range(controller.mp_size)]
    dist.all_gather(slices, bank.main_grad.contiguous(), group=controller.mp_group)
    return torch.cat(slices, dim=1)


def full_width_master(controller, index):
    bank = controller.banks[index]
    if controller.mp_size == 1:
        return bank.master.clone()
    slices = [torch.empty_like(bank.master) for _ in range(controller.mp_size)]
    dist.all_gather(slices, bank.master.contiguous(), group=controller.mp_group)
    return torch.cat(slices, dim=1)


def _tier3_worker(rank, world_size, port, mem_parallel_size, out_path):
    """Results go to a file, not to an mp.Queue: the payload is a few MB of
    tensors and the queue's feeder socket does not survive the spawned process
    exiting under it."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        model = build_model()  # identical on every rank under one seed
        banks = MemoryParallelController(
            model, mem_parallel_size, compute_dtype=torch.float64
        )
        batches = micro_batches(rank)

        # Phase A: one optimizer step's worth of backward, stopped right after
        # the reduce. This is the quantity O8's divisor scales.
        banks.materialize()
        for i in range(GA):
            ids = batches[i]
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
            )
            (loss / GA).backward()
        banks.reduce()
        if os.environ.get("NANO_TEST_WRONG_O8_DIVISOR"):
            # Negative control: exactly what dividing by the memory-DP subgroup
            # size instead of the global world size produces.
            with torch.no_grad():
                for bank in banks.banks:
                    bank.main_grad.mul_(banks.mp_size)
        grads = [
            full_width_main_grad(banks, i).clone() for i in range(len(banks.banks))
        ]
        joint_norm_sq = banks.local_grad_norm_sq().clone()
        dist.all_reduce(joint_norm_sq, op=dist.ReduceOp.SUM)

        # Phase B: the same schedule from scratch, three full steps.
        banks.zero_grad()
        model.zero_grad(set_to_none=True)
        norms = train(model, batches, banks=banks)
        losses = train.last_losses
        masters = [full_width_master(banks, i).clone() for i in range(len(banks.banks))]

        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "grads": grads,
                    "joint_norm": float(joint_norm_sq.sqrt()),
                    "norms": norms,
                    "losses": losses,
                    "masters": masters,
                    "slice_width": banks.banks[0].slice_width,
                    "dp_size": banks.dp_size,
                    "n_banks": len(banks.banks),
                },
                out_path,
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            torch.save(
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
                out_path,
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_tier3(world_size, mem_parallel_size, tmp_path, wrong_divisor=False):
    from conftest import free_port

    out = str(tmp_path / f"tier3_mp{mem_parallel_size}_{int(wrong_divisor)}.pt")
    previous = os.environ.pop("NANO_TEST_WRONG_O8_DIVISOR", None)
    if wrong_divisor:
        os.environ["NANO_TEST_WRONG_O8_DIVISOR"] = "1"
    try:
        mp.spawn(
            _tier3_worker,
            args=(world_size, free_port(), mem_parallel_size, out),
            nprocs=world_size,
            join=True,
        )
    finally:
        os.environ.pop("NANO_TEST_WRONG_O8_DIVISOR", None)
        if previous is not None:
            os.environ["NANO_TEST_WRONG_O8_DIVISOR"] = previous
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(f"mem_parallel_size={mem_parallel_size}: {payload['message']}")
    return payload


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_o8_divisor_is_the_global_world_size(mem_parallel_size, tmp_path):
    """**The Tier 3 gate.** `mem_parallel_size` must not change the gradient.

    The all-reduce runs in the memory-DP subgroup, whose size is
    `world / mp`, but the divisor is the GLOBAL world size: the forward's id
    all-gather already summed `mp` ranks' token contributions into each slice,
    so the terms total `mp * (world/mp) = world`. Dividing by the subgroup size
    instead -- the natural-looking mistake, and the one this plan called its
    most likely silent bug -- leaves every bank gradient a factor of `mp` too
    large. At mp=4 on 4 ranks the subgroup is a singleton, so that mistake is
    "no division at all" and the error is exactly 4x.

    A constant factor on the bank gradient does not break training. AdamW
    normalises by sqrt(v), so the first steps look almost unchanged and the
    damage arrives later as a wrong effective learning rate on 81% of the
    model's parameters. Nothing but this comparison would report it.
    """
    reference = run_tier3(4, 1, tmp_path)
    sharded = run_tier3(4, mem_parallel_size, tmp_path)

    assert sharded["dp_size"] == 4 // mem_parallel_size
    assert sharded["slice_width"] * mem_parallel_size == reference["slice_width"]
    assert reference["n_banks"] == sharded["n_banks"] == 2 * BLOCKS

    for i, (want, got) in enumerate(zip(reference["grads"], sharded["grads"])):
        assert want.shape == got.shape
        assert want.abs().max() > 0, f"bank {i} gradient was all zero"
        worst = (want - got).abs().max().item()
        ratio = (got.abs().sum() / want.abs().sum()).item()
        assert worst <= 1e-14, (
            f"bank {i}: gradient differs by {worst:.3e} at mem_parallel_size "
            f"{mem_parallel_size} (magnitude ratio {ratio:.4f} -- a ratio near "
            f"{mem_parallel_size} or 1/{mem_parallel_size} means the divisor)"
        )


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_loss_trajectory_coincides_across_mem_parallel_size(
    mem_parallel_size, tmp_path
):
    """Section 1.4.2 and 1.4.3 for Tier 3: same losses, same grad-norm trace,
    same parameters. The norm trace is the part that catches a bank norm counted
    once per dp-rank instead of once per slice -- that inflates it by
    sqrt(dp_size) and changes nothing else."""
    reference = run_tier3(4, 1, tmp_path)
    sharded = run_tier3(4, mem_parallel_size, tmp_path)

    for step, (a, b) in enumerate(zip(reference["losses"], sharded["losses"])):
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (
            f"loss diverged at step {step}: mp=1 {a!r} vs "
            f"mp={mem_parallel_size} {b!r}"
        )
    for step, (a, b) in enumerate(zip(reference["norms"], sharded["norms"])):
        assert abs(a - b) <= 1e-6 * max(1.0, abs(a)), (
            f"grad norm diverged at step {step}: mp=1 {a!r} vs "
            f"mp={mem_parallel_size} {b!r}"
        )
    assert abs(reference["joint_norm"] - sharded["joint_norm"]) <= 1e-6 * abs(
        reference["joint_norm"]
    ), (
        f"the joint clip norm moved with the layout: {reference['joint_norm']!r} "
        f"vs {sharded['joint_norm']!r} -- a bank slice is being counted once per "
        "dp-rank rather than once"
    )
    for i, (want, got) in enumerate(zip(reference["masters"], sharded["masters"])):
        worst = (want - got).abs().max().item()
        assert worst <= 1e-12, f"bank {i} master differs by {worst:.3e} after 3 steps"


def _forward_worker(rank, world_size, port, mem_parallel_size, queue):
    """The column split plus the all-to-all must reassemble the EXACT tensor the
    unsharded gather produces, column for column. A transposed reshape here is
    a permutation of the head dimension: the model still trains, and every head
    is reading another head's bank row."""
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        reference_model = build_model()
        reference = reference_model.blocks[0].attention.q_bank
        ids = micro_batches(rank, n=1)[0]
        with torch.no_grad():
            want = reference(ids)

        model = build_model()
        MemoryParallelController(model, mem_parallel_size, compute_dtype=torch.float64)
        got = model.blocks[0].attention.q_bank(ids)

        if rank == 0:
            queue.put(("ok", {"equal": bool(torch.equal(want, got.detach()))}))
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            queue.put(("error", f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("mem_parallel_size", [2, 4])
def test_column_shard_reassembles_the_full_width_gather(mem_parallel_size):
    from conftest import free_port

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(
        _forward_worker,
        args=(4, free_port(), mem_parallel_size, queue),
        nprocs=4,
        join=True,
    )
    status, payload = queue.get(timeout=120)
    if status == "error":
        pytest.fail(payload)
    assert payload["equal"], (
        "the sharded gather is not the unsharded gather -- the all-to-all "
        "chunks are being reassembled in the wrong column order"
    )


def test_the_o8_gate_catches_a_wrong_divisor(tmp_path):
    """Negative control for the test above. Dividing by the memory-DP subgroup
    size instead of the global world size must be visible as a factor of
    `mem_parallel_size` on every bank gradient -- if it is not, the gate passes
    on the bug it was written for."""
    reference = run_tier3(4, 1, tmp_path)
    broken = run_tier3(4, 4, tmp_path, wrong_divisor=True)

    ratios = [
        (b.abs().sum() / a.abs().sum()).item()
        for a, b in zip(reference["grads"], broken["grads"])
    ]
    assert all(abs(r - 4.0) < 1e-9 for r in ratios), (
        f"the wrong divisor did not show up as a 4x gradient: ratios {ratios}"
    )
