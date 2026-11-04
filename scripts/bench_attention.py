"""Forward and backward cost of AME's attention kernel, measured separately.

The aggregated kernel dump cannot answer this. AME's `flex_attention` is
compiled into the whole-block graph, so no operator boundary survives for the
profiler to attribute against -- unlike the dense model, whose flash path leaves
`aten::_scaled_dot_product_flash_attention{,_backward}` rows and splits cleanly
(182.1 ms forward / 365.4 ms backward per step, job 32601).

So measure it directly, on the real layer, at the real shape.

    python scripts/bench_attention.py                 # h20 shape, one GPU

Reports milliseconds per CALL and, scaled by layers x micro-steps, the
contribution to one optimizer step of the h20 4-GPU recipe -- so the numbers
line up with the kernel tables in documents/notes/ame_optimization_plan.md.

The SDPA row is the floor, not a drop-in replacement: `is_causal=True` with no
additive bias is what the dense twin runs. AME needs an ALiBi distance bias and
a learned sink, neither of which PyTorch 2.7's flash backend accepts. The gap
between the two rows is what a bias-free kernel would be worth, which is the
thing worth knowing before anyone engineers one.
"""

import argparse
import sys

import torch

sys.path.insert(0, "/home/maciej/code/nano")


def timed(fn, *, warmup=5, iters=20):
    """ms per call, CUDA-event timed, warmed and synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16, help="micro-batch per GPU")
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--dhead", type=int, default=64)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--micro-steps", type=int, default=8, help="ga")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(dev)}  torch {torch.__version__}")
    print(f"shape: B={args.batch} H={args.heads} n={args.seq} D={args.dhead} bf16")
    per_step = args.layers * args.micro_steps
    print(f"per-step scale: {args.layers} layers x {args.micro_steps} micro-steps "
          f"= {per_step} calls\n")

    from src.core.ame import AMEAttention

    layer = AMEAttention(
        args.heads * args.dhead, args.heads, args.heads,
        vocab_size=50304, seq_len=args.seq, dhead=args.dhead,
    ).to(dev).to(torch.bfloat16)

    shape = (args.batch, args.heads, args.seq, args.dhead)

    def fresh(requires_grad=True):
        return [torch.randn(shape, device=dev, dtype=torch.bfloat16,
                            requires_grad=requires_grad) for _ in range(3)]

    rows = []

    # ---- AME's actual kernel: compiled flex_attention + score_mod + block mask
    q, k, v = fresh()
    fwd_ms = timed(lambda: layer._attend_flex(q, k, v))
    out = layer._attend_flex(q, k, v)
    grad = torch.randn_like(out)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(grad, retain_graph=True)

    bwd_ms = timed(bwd)
    rows.append(("AME flex_attention (ALiBi + sink)", fwd_ms, bwd_ms))

    # ---- the floor: SDPA flash, causal, no bias
    q2, k2, v2 = fresh()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa_fwd():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, is_causal=True)

    try:
        f2 = timed(sdpa_fwd)
        o2 = sdpa_fwd()
        g2 = torch.randn_like(o2)

        def sdpa_bwd():
            for t in (q2, k2, v2):
                t.grad = None
            o2.backward(g2, retain_graph=True)

        b2 = timed(sdpa_bwd)
        rows.append(("SDPA flash, causal, NO bias (floor)", f2, b2))
    except Exception as exc:  # noqa: BLE001
        print(f"SDPA flash unavailable: {type(exc).__name__}: {exc}")

    hdr = f"{'kernel':38s} {'fwd ms':>8} {'bwd ms':>8} {'bwd/fwd':>8} {'total':>8}"
    print(hdr); print("-" * len(hdr))
    for name, f, b in rows:
        print(f"{name:38s} {f:8.3f} {b:8.3f} {b/f:8.2f} {f+b:8.3f}")
    print()
    print(f"{'scaled to one optimizer step (ms)':38s} {'fwd':>8} {'bwd':>8} {'':>8} {'total':>8}")
    print("-" * len(hdr))
    for name, f, b in rows:
        print(f"{name:38s} {f*per_step:8.1f} {b*per_step:8.1f} {'':>8} "
              f"{(f+b)*per_step:8.1f}")
    if len(rows) == 2:
        (_, f1, b1), (_, f2_, b2_) = rows
        print()
        print(f"headroom if the bias were free: fwd {(f1-f2_)*per_step:+.1f} ms, "
              f"bwd {(b1-b2_)*per_step:+.1f} ms, "
              f"total {((f1+b1)-(f2_+b2_))*per_step:+.1f} ms per step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
