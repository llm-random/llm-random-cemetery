#!/usr/bin/env bash
# Submit the MoE vs dense step profiles to lem: configs/jm/mfu/h{12,16,20}_sl1024_{moe,dense}_profile.yaml.
# Optional $1: a job id (e.g. update_pixi.py +wait=false) they start after; cancelled if it fails.
set -euo pipefail
cd "$(dirname "$0")/.."
after=()
if [ $# -ge 1 ]; then
  after=("++infrastructure.slurm.dependency=afterok:$1" "++infrastructure.slurm.kill_on_invalid_dep=yes")
fi
for h in 12 16 20; do
  for kind in moe dense; do
    pixi run python run_exp.py --config-path configs/jm/mfu --config-name "h${h}_sl1024_${kind}_profile" ${after[@]+"${after[@]}"}
  done
done
