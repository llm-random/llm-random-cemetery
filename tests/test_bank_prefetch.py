"""O39: the bank exchange issued ahead of each layer must change nothing.

`BankPrefetcher` moves each layer's gather and all-to-all out of the compiled
block and onto a separate CUDA stream, and hands the block the all-to-all's
receive buffer as it lands (rank-major); the block reassembles it through a
view. Every way that can go wrong is silent:

- a block handed another block's rows trains, on the wrong banks;
- a rank-major buffer reassembled in the wrong order swaps heads between
  ranks' column slices, and the model still trains;
- a missing stream wait reads rows before the exchange has written them;
- a gradient freed while the bank stream still reads it is a race.

The CPU tests run the real AME stack (src.core.model, h4) on a gloo mesh in
float64 and require BIT-identity with the fused path (`_FusedBankGather`),
which computes the same values in the same order. The CUDA test runs the
production shape -- NCCL, compiled blocks, the bank stream -- against the eager
fused path.
"""

import os

import pytest
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import resolver  # noqa: F401  (registers the eval resolver the configs use)
from src.core.memory_parallel import MemoryParallelController
from src.core.model import parameter_groups

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 96
STEPS = 2
GA = 2
LR = 0.01


def _h4_ame(seq_len, device="cpu"):
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(REPO_ROOT, "configs", "_model", "h_balanced"),
    ):
        cfg = compose(
            config_name="h4",
            overrides=[
                f"common.sequence_length={seq_len}",
                f"common.vocab_size={VOCAB}",
                "common.attention.type=ame",
            ],
        )
    with torch.device(device):
        return instantiate(cfg.model, _convert_="all")


def _batches(rank, seq_len, device):
    generator = torch.Generator().manual_seed(100 + rank)
    return [
        torch.randint(0, VOCAB, (2, seq_len), generator=generator).to(device)
        for _ in range(STEPS * GA)
    ]


def _full_width(banks, tensor):
    if banks.mp_size == 1:
        return tensor.detach().cpu().clone()
    slices = [torch.empty_like(tensor) for _ in range(banks.mp_size)]
    dist.all_gather(slices, tensor.contiguous(), group=banks.mp_group)
    return torch.cat(slices, dim=1).cpu()


def _train(model, banks, batches):
    """Trainer.train's optimizer section, as in tests/test_memory_parallel.py."""
    optimizer = torch.optim.AdamW(
        parameter_groups(model, 0.1, exclude=banks.ignored_parameters()), lr=LR
    )
    banks.build_optimizer(lr=LR, weight_decay=0.0)
    losses = []
    model.train()
    for step in range(STEPS):
        banks.materialize()
        for i in range(GA):
            ids = batches[step * GA + i]
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1)
            )
            (loss / GA).backward()
            losses.append(loss.item())
        banks.reduce()
        optimizer.step()
        optimizer.zero_grad()
        banks.step(lr=LR)
        banks.zero_grad()
    return losses


def _cpu_worker(rank, world_size, port, mp_size, out_path):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        batches = _batches(rank, 16, "cpu")
        results = {}
        for tag, depth in (("fused", None), ("prefetch", 2)):
            torch.manual_seed(0)
            model = _h4_ame(16).double()
            banks = MemoryParallelController(
                model, mp_size, compute_dtype=torch.float64, prefetch_depth=depth
            )
            assert (banks.prefetcher is not None) == bool(depth)
            initial = [_full_width(banks, b.master) for b in banks.banks]
            losses = _train(model, banks, batches)
            results[tag] = {
                "initial": initial,
                "losses": losses,
                "masters": [_full_width(banks, b.master) for b in banks.banks],
                "params": {
                    n: p.detach().clone() for n, p in model.named_parameters()
                },
            }
        a, b = results["fused"], results["prefetch"]
        worst_master = max(
            (x - y).abs().max().item() for x, y in zip(a["masters"], b["masters"])
        )
        worst_param = max(
            (a["params"][n] - b["params"][n]).abs().max().item() for n in a["params"]
        )
        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "losses_equal": a["losses"] == b["losses"],
                    "worst_master": worst_master,
                    "worst_param": worst_param,
                    "bank_moved": max(
                        (x - y).abs().max().item()
                        for x, y in zip(b["masters"], b["initial"])
                    ),
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


@pytest.mark.parametrize("mp_size", [2, 4])
def test_prefetch_is_bit_identical_to_the_fused_path(mp_size, tmp_path):
    from conftest import spawn_on_free_port

    out = str(tmp_path / f"prefetch_mp{mp_size}.pt")
    spawn_on_free_port(
        _cpu_worker, lambda port: (mp_size, port, mp_size, out), mp_size
    )
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])
    assert payload["bank_moved"] > 0, "the banks never moved; the test is vacuous"
    assert payload["losses_equal"], "the prefetch path changed the loss trajectory"
    assert payload["worst_master"] == 0.0, (
        f"bank masters differ by {payload['worst_master']:.3e}: the rank-major "
        "rows are being reassembled, or routed to a layer, wrongly"
    )
    assert payload["worst_param"] == 0.0, (
        f"backbone parameters differ by {payload['worst_param']:.3e}"
    )


def test_the_encoder_runs_unchanged_without_a_prefetcher():
    """No controller, no prefetch: the encoder takes the original path."""
    model = _h4_ame(16)
    assert model.encoder.bank_prefetcher is None
    ids = torch.randint(0, VOCAB, (2, 16))
    assert torch.isfinite(model(ids)).all()


def test_prefetch_stays_off_at_mem_parallel_size_one():
    """At mp=1 there is no exchange to move; the fused gather stays inside the
    compiled block."""
    model = _h4_ame(16).double()
    banks = MemoryParallelController(
        model, 1, compute_dtype=torch.float64, prefetch_depth=2
    )
    assert banks.prefetcher is None
    assert model.encoder.bank_prefetcher is None


# ---- CUDA: NCCL, compiled blocks, the bank stream ---------------------------

RTOL = 1e-4


def _cuda_worker(rank, world_size, port, out_path):
    try:
        from conftest import isolate_compile_caches

        isolate_compile_caches(rank)
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device("cuda", rank)
        seq = 128  # flex's block size, as the real runs align to
        batches = _batches(rank, seq, device)

        results = {}
        for tag, depth, compiled in (("eager_fused", None, False), ("compiled_prefetch", 2, True)):
            torch._dynamo.reset()
            torch.manual_seed(0)
            model = _h4_ame(seq, device="cuda").to(device)
            banks = MemoryParallelController(
                model, world_size, compute_dtype=torch.float32, prefetch_depth=depth
            )
            if compiled:
                for block in model.encoder.blocks:
                    block.compile()
            losses = _train(model, banks, batches)
            results[tag] = {
                "losses": losses,
                "masters": [_full_width(banks, b.master) for b in banks.banks],
                "params": [p.detach().cpu().clone() for p in model.parameters()],
            }
        a, b = results["eager_fused"], results["compiled_prefetch"]

        def rel(xs, ys):
            worst = 0.0
            for x, y in zip(xs, ys):
                scale = x.abs().max().clamp_min(1e-30)
                worst = max(worst, ((x - y).abs().max() / scale).item())
            return worst

        master_err = torch.tensor(rel(a["masters"], b["masters"]), device=device)
        param_err = torch.tensor(rel(a["params"], b["params"]), device=device)
        dist.all_reduce(master_err, op=dist.ReduceOp.MAX)
        dist.all_reduce(param_err, op=dist.ReduceOp.MAX)
        if rank == 0:
            torch.save(
                {
                    "status": "ok",
                    "master_err": master_err.item(),
                    "param_err": param_err.item(),
                    "loss_diff": max(abs(x - y) for x, y in zip(a["losses"], b["losses"])),
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


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs at least 2 CUDA devices",
)
def test_compiled_prefetch_matches_the_eager_fused_path_over_nccl(tmp_path):
    from conftest import spawn_on_free_port

    world_size = min(4, torch.cuda.device_count())
    out = str(tmp_path / "prefetch_cuda.pt")
    spawn_on_free_port(_cuda_worker, lambda port: (world_size, port, out), world_size)
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])
    # Two optimizer steps in fp32 through compiled kernels: the masters move by
    # LR-sized steps, so a relative error of 1e-4 is far below any structural
    # mistake (a misrouted layer or swapped heads is O(1)).
    assert payload["master_err"] < RTOL, payload
    assert payload["param_err"] < RTOL, payload
    assert payload["loss_diff"] < 1e-3, payload


# ---- O38(b): no bf16 compute copy; the gathered weight is the fp32 master ---


def _o38b_run(mp_size, compute_copy, depth=None):
    torch.manual_seed(0)
    model = _h4_ame(16).float()
    banks = MemoryParallelController(
        model,
        mp_size,
        compute_dtype=torch.bfloat16,
        prefetch_depth=depth,
        compute_copy=compute_copy,
    )
    if not compute_copy:
        assert all(b.weight.dtype == torch.float32 for b in banks.banks)
        assert all(b.master.data_ptr() == b.weight.data_ptr() for b in banks.banks)
    rank = dist.get_rank() if dist.is_initialized() else 0
    losses = _train(model, banks, _batches(rank, 16, "cpu"))
    return losses, [_full_width(banks, b.master) for b in banks.banks]


def test_o38b_is_bit_identical_in_process():
    """The copy only ever held master.to(bf16): gathering the master and
    casting gives the same rows, so the whole trajectory is identical."""
    base_losses, base_masters = _o38b_run(1, compute_copy=True)
    losses, masters = _o38b_run(1, compute_copy=False)
    assert losses == base_losses
    assert all(torch.equal(a, b) for a, b in zip(base_masters, masters))


def _o38b_worker(rank, world_size, port, out_path):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        a = _o38b_run(world_size, compute_copy=True)
        b = _o38b_run(world_size, compute_copy=False, depth=2)
        same = a[0] == b[0] and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
        if rank == 0:
            torch.save({"status": "ok", "same": same}, out_path)
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


def test_o38b_with_prefetch_is_bit_identical_over_gloo(tmp_path):
    """mp 2, the exchange path: compute copy + fused pair against fp32 master +
    prefetch. Both changes together, so one test covers the combination the
    4-GPU runs use."""
    from conftest import spawn_on_free_port

    out = str(tmp_path / "o38b.pt")
    spawn_on_free_port(_o38b_worker, lambda port: (2, port, out), 2)
    payload = torch.load(out, weights_only=False)
    if payload["status"] == "error":
        pytest.fail(payload["message"])
    assert payload["same"], "O38(b) changed the trajectory"
