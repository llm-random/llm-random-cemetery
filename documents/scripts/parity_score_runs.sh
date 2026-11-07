#!/bin/bash
# ame_bf16_scalars_plan.md Step 4: score finished full-length runs with the common
# evaluator (<arm>_ref32) and each run's own production path, chained in one job.
#
#   documents/scripts/parity_score_runs.sh COMMIT JOBNAME SHORT [SHORT...]
#
# SHORT is a run's short name (ame_b1, p1_2, ame_f3, db1, df2, pref, p0b, p1_16_1).
set -euo pipefail
COMMIT=$1; NAME=$2; shift 2
cmds=()
for short in "$@"; do
  case $short in
    ame_b*|p0b) arm=ame; ev=ame_ref32,ame_prod ;;
    p1_*) arm=ame; ev=ame_ref32,ame_std ;;
    ame_f*) arm=ame; ev=ame_ref32,ame_f3x ;;
    pref) arm=ame; ev=ame_ref32,ame_pref ;;
    db*) arm=dense; ev=dense_ref32,dense_prod ;;
    df*) arm=dense; ev=dense_ref32,dense_f ;;
    *) echo "unknown run $short" >&2; exit 1 ;;
  esac
  # The checkpoints live on h100a's NVMe, which the login node cannot see, so the glob
  # is written into the job script unexpanded and resolves on the node. Runs started
  # before the move (2026-11-07 ~02:00) save under <short>, later ones under
  # parity_full_<short>; `*<short>` matches either and nothing else.
  ckpt="/storage_nvme_1/\$USER/parity_full/*${short}/*/0/step_3814"
  cmds+=("score --arms $arm --ckpt-$arm $ckpt --evaluators $ev --tag $short")
done
joined=$(printf ' :: %s' "${cmds[@]}"); joined=${joined# :: }
set -f  # $joined is split into words below; its globs must reach the node unexpanded
"$(dirname "$0")/parity_submit.sh" "$COMMIT" "$NAME" 1 00:55:00 -- $joined
