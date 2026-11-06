"""Distinct token ids per micro-batch on the training stream (AME lead A3a, O15).

Sizes deduplicated bank exchange: each rank would request only the distinct ids
of its micro-batch, so the exchange volume scales with distinct/tokens.

    python scripts/unique_ids.py --config-path configs/ame --config-name h20_4gpu [--batches 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default="configs/ame")
    ap.add_argument("--config-name", default="h20_4gpu")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--micro", type=int, default=16)
    args = ap.parse_args()
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    import resolver  # noqa: F401
    import src.definitions  # noqa: F401  (registers base_schema)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with initialize_config_dir(version_base=None, config_dir=os.path.join(root, args.config_path)):
        cfg = compose(config_name=args.config_name)
    loader = instantiate(cfg.trainer.train_dataloader)
    it = iter(loader)
    for b in range(args.batches):
        batch = next(it)[:, :-1]  # the model sees width - 1
        micro = batch.split(args.micro)
        per = [torch.unique(m).numel() / m.numel() for m in micro]
        four = [torch.unique(torch.cat(micro[i:i + 4])).numel() / (4 * micro[0].numel())
                for i in range(0, len(micro) - 3, 4)]
        print(f"batch {b}: {batch.shape}, distinct/tokens per micro-batch of {args.micro}: "
              f"mean {sum(per)/len(per):.3f} min {min(per):.3f} max {max(per):.3f}; "
              f"over 4 micro-batches (4 ranks' tokens): mean {sum(four)/len(four):.3f}")


if __name__ == "__main__":
    main()
