"""Time one compiled AME block, cold vs warm block-mask cache, and name its kernels.

Diagnoses O42 (AME report section 1.21): with the flex block mask built before
the block's first trace, the 4-GPU forward ran ~5x slower (job 33588). Builds
one real h20 block on one GPU, compiles it, and for each variant reports the
steady-state forward and backward time, the number of compiled graphs run per
call, and the top kernels -- whether flex's Triton template (`triton_tem_`)
runs, or the unfused math path.

    python scripts/diag_block_compile.py [--batch 4] [--logger]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    args = ap.parse_args()

    import torch
    import torch._dynamo
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import resolver  # noqa: F401

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(root, "configs", "_model", "h_balanced")
    ):
        cfg = compose(
            config_name="h20",
            overrides=[
                f"common.sequence_length={args.seq}",
                "common.vocab_size=50304",
                "common.attention.type=ame",
            ],
        )

    for variant in ("cold", "warm"):
        torch._dynamo.reset()
        torch.manual_seed(0)
        with torch.device("cuda"):
            model = instantiate(cfg.model, _convert_="all")
        block = model.encoder.blocks[0].to(torch.bfloat16)
        layer = block.attention_layer.layer
        dmodel = block.ff_layer.layer.ff_pre_act.weight.shape[1]
        x = torch.randn(args.batch, args.seq, dmodel, device="cuda",
                        dtype=torch.bfloat16, requires_grad=True)
        ids = torch.randint(0, 50304, (args.batch, args.seq), device="cuda")
        if variant == "warm":
            layer.prepare_for_shape(args.seq, x.device)
        compiled = torch.compile(block)
        for _ in range(3):
            out = compiled(x, ids)
            out.float().sum().backward()
        torch.cuda.synchronize()
        s, m, e = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            s.record()
            out = compiled(x, ids)
            m.record()
            out.float().sum().backward()
            e.record()
            torch.cuda.synchronize()
        graphs = sum(1 for ev in prof.events() if ev.name == "CompiledFunction")
        print(f"== {variant}: fwd {s.elapsed_time(m):.2f} ms, bwd {m.elapsed_time(e):.2f} ms, "
              f"CompiledFunction calls {graphs}, cache {len(layer._block_mask_cache)}, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        rows = sorted(
            (k for k in prof.key_averages() if k.device_time_total > 0 and "cuda" in str(k.device_type).lower()),
            key=lambda k: -k.device_time_total,
        )[:12]
        for k in rows:
            print(f"   {k.device_time_total / 1000:8.2f} ms  x{k.count:<4d} {k.key[:110]}")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
