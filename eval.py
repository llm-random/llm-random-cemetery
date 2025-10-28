"""
Minimal Hydra-based evaluation script:
- Loads a checkpoint
- Converts to HuggingFace format
- Runs lm_eval on a specified task
"""

import os
import tempfile
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
from lm_eval import evaluator

from src.core.checkpointing import load_checkpoint_from_file
from src.core.conversion_to_hf import save_to_llama_3_hf


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    """
    Main evaluation function.

    Config expects:
    - checkpoint: checkpoint config with path and model_checkpoint_filename
    - model: model config (instantiable)
    - model_params: dict with dmodel, dff, n_att_heads, n_kvatt_heads, head_dim, nlayers
    - eval_task: lm_eval task name (e.g., 'wikitext', 'hellaswag')
    - eval_num_fewshot: number of few-shot examples (default: 0)
    - eval_batch_size: batch size for evaluation (default: 8)
    """

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load nano model
    print(f"\n=== Loading nano model ===")
    model = instantiate(cfg.model, _convert_="all").to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Load checkpoint (dummy optimizer/scheduler)
    print(f"Loading checkpoint from {cfg.checkpoint.path}")
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    load_checkpoint_from_file(cfg.checkpoint, model, dummy_optimizer, None)
    model.eval()

    # Convert to HuggingFace format
    print(f"\n=== Converting to HuggingFace format ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model_path = os.path.join(tmpdir, "hf_model")
        os.makedirs(hf_model_path, exist_ok=True)

        # Save nano state_dict in HF format
        save_to_llama_3_hf(
            model.state_dict(),
            hf_model_path,
            dmodel=cfg.model_params.dmodel,
            dff=cfg.model_params.dff,
            n_att_heads=cfg.model_params.n_att_heads,
            n_kvatt_heads=cfg.model_params.n_kvatt_heads,
            head_dim=cfg.model_params.head_dim,
            nlayers=cfg.model_params.nlayers,
        )
        print(f"Saved HF model to {hf_model_path}")

        # Run lm_eval
        print(f"\n=== Running lm_eval on task: {cfg.eval_task} ===")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={hf_model_path},dtype=float16",
            tasks=[cfg.eval_task],
            num_fewshot=cfg.get("eval_num_fewshot", 0),
            batch_size=cfg.get("eval_batch_size", 8),
            device=str(device),
        )

        print(f"\n=== Results ===")
        print(results)

        return results


if __name__ == "__main__":
    main()
