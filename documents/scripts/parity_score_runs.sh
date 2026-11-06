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
    ame_f*) arm=ame; ev=ame_ref32,ame_f ;;
    pref) arm=ame; ev=ame_ref32,ame_pref ;;
    db*) arm=dense; ev=dense_ref32,dense_prod ;;
    df*) arm=dense; ev=dense_ref32,dense_f ;;
    *) echo "unknown run $short" >&2; exit 1 ;;
  esac
  ckpt=$(ssh -o BatchMode=yes entropy "ls -d ~/llmrandom_cemetery/parity_full_${short}_*/checkpoint/*/0/step_3814 2>/dev/null | tail -1")
  [ -n "$ckpt" ] || { echo "no final checkpoint for $short" >&2; exit 1; }
  cmds+=("score --arms $arm --ckpt-$arm $ckpt --evaluators $ev --tag $short")
done
joined=$(printf ' :: %s' "${cmds[@]}"); joined=${joined# :: }
"$(dirname "$0")/parity_submit.sh" "$COMMIT" "$NAME" 1 00:55:00 -- $joined
