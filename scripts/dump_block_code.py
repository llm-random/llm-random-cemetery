"""Dump the Triton code inductor generates for one AME (or dense) block.

The kernel profile names kernels (`triton_red_fused_div_mul_sum_7`) but not
what they compute; this compiles one real TransformerBlock at the production
shape on one GPU, runs forward and backward once, and leaves inductor's output
code where `scripts/kprof_kernels.py`-style analysis can be read against it.

    python scripts/dump_block_code.py --out /tmp/code [--attention gqa] [--batch 16]

Writes `<out>/output_code.log` (TORCH_LOGS=output_code) and prints, per
generated kernel, its name and the source ops inductor fused into it.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention", default="ame")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--h", default="h20")
    ap.add_argument("--prefetched", action="store_true",
                    help="feed the block O39's rank-major rows (mp 4 layout)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    import torch._logging
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import resolver  # noqa: F401

    handler = logging.FileHandler(os.path.join(args.out, "output_code.log"))
    torch._logging.set_logs(output_code=True)
    logging.getLogger("torch._inductor").addHandler(handler)
    for name in ("torch._inductor.codecache", "torch._inductor.graph",
                 "torch._inductor.compile_fx", "torch._inductor"):
        logging.getLogger(name).addHandler(handler)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(root, "configs", "_model", "h_balanced")
    ):
        cfg = compose(
            config_name=args.h,
            overrides=[
                f"common.sequence_length={args.seq}",
                "common.vocab_size=50304",
                f"common.attention.type={args.attention}",
            ],
        )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = instantiate(cfg.model, _convert_="all")
    block = model.encoder.blocks[0].to(torch.bfloat16)
    x = torch.randn(args.batch, args.seq, block.ff_layer.layer.ff_pre_act.weight.shape[1],
                    device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(0, 50304, (args.batch, args.seq), device="cuda")
    compiled = torch.compile(block)
    extra = ()
    if args.prefetched:
        layer = block.attention_layer.layer
        width = layer.q_heads * layer.dhead // 4
        rows = [torch.randn(4 * args.batch * args.seq, width, device="cuda",
                            dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        extra = (tuple(rows),)
    out = compiled(x, ids, *extra)
    out.float().sum().backward()
    torch.cuda.synchronize()
    print("done; code in", os.path.join(args.out, "output_code.log"))


if __name__ == "__main__":
    main()
