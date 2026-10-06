#!/bin/bash
# Activate the entropy pixi env (same as configs/_cluster/entropy.yaml) and run "$@" from the code dir.
# HF_TOKEN, if needed, is read from stdin by the caller (never put on the command line).
set -e
export PROJECT_HOME_PATH=/storage_nvme_4/nano
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
# smoke tests use only cached HF assets (tokenizers, Meta-Llama-3-8B, WikiText) -> no token / network needed
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PIXI_HOME=$PROJECT_HOME_PATH/$USER/pixi
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_DATA_HOME="$PIXI_HOME/data" XDG_CACHE_HOME="$PIXI_HOME/cache" XDG_STATE_HOME="$PIXI_HOME/state"
cd "$PIXI_HOME"; eval "$(pixi shell-hook)"; cd "$HOME/slicegpt_dev/nano-slicegpt"
"$@"
