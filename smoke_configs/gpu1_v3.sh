#!/bin/bash
# V3: dump the official pipeline's exact WikiText2 tokens (their env), then run our port on them (our env).
OUT=$HOME/slicegpt_dev/results; mkdir -p $OUT
echo "===== V3 token dump (official env) ====="
HF_HOME=$HOME/slicegpt_dev/hf_home PYTHONPATH=$HOME/slicegpt_dev/ref_libs:$HOME/slicegpt_dev/TransformerCompression/src \
  python scripts/slicegpt_dump_ref_tokens.py meta-llama/Meta-Llama-3-8B $OUT/ref_tokens.pt > $OUT/v3_dump.log 2>&1; echo "dump exit $?"
echo "===== V3 ours ====="
HF_HOME=$HOME/slicegpt_dev/hf_home python scripts/slicegpt_crosscheck.py --token-file $OUT/ref_tokens.pt --out $OUT/v3_ours.json > $OUT/v3_ours.log 2>&1; echo "V3 ours exit $?"
