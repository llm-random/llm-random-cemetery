#!/bin/bash
# One 1-GPU allocation for: S2 (hybrid r50 slicing + save), S3 (strict reload into sliced config -> same loss),
# V3 ours (port, faithful mode, Meta-Llama-3-8B) and V3 theirs (official repo, pinned transformers 4.41).
# Usage (inside run_env.sh, from the code dir): bash smoke_configs/gpu1_chain.sh
set -x
OUT=$HOME/slicegpt_dev/results; mkdir -p $OUT
SMOKE=/storage_nvme_4/nano/j321m/pc/slicegpt_smoke

echo "===== S2 ====="
[ -n "$SKIP_S2" ] || python main.py --config-path=smoke_configs --config-name=s2_slice_r50.yaml > $OUT/s2.log 2>&1; echo "S2 exit $?"

echo "===== S3 ====="
python - > $OUT/s3.log 2>&1 <<'PY'
import json, os
for k, v in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}.items(): os.environ.setdefault(k, v)
import torch, resolver
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from src.projected_compression.slicegpt.run_slicing import evaluate_loss, sample_token_batches
SMOKE = "/storage_nvme_4/nano/j321m/pc/slicegpt_smoke"
meta = json.load(open(f"{SMOKE}/llama1b_slicegpt_hyb_r50_smoke.json"))
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("configs/pc_project_mas")):
    cfg = compose(config_name="llama_1B_slicegpt_hyb_entropy", overrides=["slicegpt_ratio=r50"])
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("smoke_configs")):
    scfg = compose(config_name="s2_slice_r50.yaml")
model = instantiate(cfg.model, _convert_="all")
model.load_state_dict(torch.load(f"{SMOKE}/llama1b_slicegpt_hyb_r50_smoke.pt", map_location="cpu"), strict=True)
model = model.cuda().float()
ev = scfg.slicegpt.eval
batches = sample_token_batches(scfg.trainer.train_dataloader.dataset, ev.sequence_length, ev.seed, ev.n_samples, ev.batch_size)
loss = evaluate_loss(model, batches, torch.device("cuda"))
params = sum(p.numel() for p in model.parameters())
print("S3_RESULT", json.dumps({"reloaded_loss": loss, "sliced_loss_in_job": meta["sliced_loss"], "abs_diff": abs(loss - meta["sliced_loss"]),
      "reloaded_params": params, "job_params": meta["sliced_params"]}))
PY
echo "S3 exit $?"

echo "===== V3 token dump (official env) ====="
HF_HOME=$HOME/slicegpt_dev/hf_home PYTHONPATH=$HOME/slicegpt_dev/ref_libs:$HOME/slicegpt_dev/TransformerCompression/src python scripts/slicegpt_dump_ref_tokens.py meta-llama/Meta-Llama-3-8B $OUT/ref_tokens.pt > $OUT/v3_dump.log 2>&1; echo "dump exit $?"

echo "===== V3 ours ====="
HF_HOME=$HOME/slicegpt_dev/hf_home python scripts/slicegpt_crosscheck.py --token-file $OUT/ref_tokens.pt --out $OUT/v3_ours.json > $OUT/v3_ours.log 2>&1; echo "V3 ours exit $?"

echo "===== V3 theirs ====="
( cd $HOME/slicegpt_dev/TransformerCompression/experiments && \
  HF_HOME=$HOME/slicegpt_dev/hf_home PYTHONPATH=$HOME/slicegpt_dev/ref_libs:$HOME/slicegpt_dev/TransformerCompression/src \
  python run_slicegpt.py --model meta-llama/Meta-Llama-3-8B --sparsity 0.25 --dtype fp32 --cal-dataset wikitext2 \
    --cal-nsamples 128 --cal-max-seqlen 2048 --cal-batch-size 16 --final-orientation pca --round-interval 8 \
    --ppl-eval-batch-size 4 --eval-baseline --eval-fused-model --no-wandb --device cuda:0 ) > $OUT/v3_theirs.log 2>&1; echo "V3 theirs exit $?"
