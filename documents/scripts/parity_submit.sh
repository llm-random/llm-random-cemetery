#!/bin/bash
# Parity audit: run one parity_numerics.py check on entropy, from a committed
# state of parity_audit (git archive of COMMIT, unpacked under ~/parity/code).
#
#   documents/scripts/parity_submit.sh COMMIT NAME GPUS TIME -- CHECK [ARGS...] [:: CHECK [ARGS...]]
#
# Several checks separated by `::` run one after the other in the same job.
# GPUS 0 runs CPU-only on the h100 node (the only one with /storage_nvme_4).
# Prints the job id. Logs: entropy:~/parity/logs/NAME_<job>.out, results in
# entropy:~/parity/out/NAME_<job>/.
set -euo pipefail
COMMIT=$1; NAME=$2; GPUS=$3; TIME=$4; shift 4
[ "$1" = "--" ] && shift
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
FULL=$(git -C "$REPO" rev-parse "$COMMIT")
SHORT=${FULL:0:8}
if ! ssh -o BatchMode=yes entropy "test -f ~/parity/code/$SHORT/PARITY_COMMIT"; then
  git -C "$REPO" archive "$FULL" | ssh -o BatchMode=yes entropy \
    "mkdir -p ~/parity/code/$SHORT ~/parity/logs ~/parity/out && tar -x -C ~/parity/code/$SHORT && echo $FULL > ~/parity/code/$SHORT/PARITY_COMMIT"
fi
if [ "$GPUS" = "0" ]; then RES="#SBATCH --cpus-per-task=8
#SBATCH --mem=48G"; else RES="#SBATCH --gres=gpu:h100:$GPUS
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=90G"; fi
ssh -o BatchMode=yes entropy "cat > ~/parity/logs/$NAME.sbatch && sbatch --parsable ~/parity/logs/$NAME.sbatch" <<SBATCH
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodes=1
$RES
#SBATCH --time=$TIME
#SBATCH --job-name=$NAME
#SBATCH --output=/home/mstefaniak/parity/logs/${NAME}_%j.out
export PROJECT_HOME_PATH=/storage_nvme_4/nano
export HF_HOME=\$PROJECT_HOME_PATH/hf_cache
export PIXI_HOME=\$PROJECT_HOME_PATH/\$USER/pixi
export PATH="\$HOME/.pixi/bin:\$PATH"
export XDG_DATA_HOME="\$PIXI_HOME/data"
export XDG_CACHE_HOME="\$PIXI_HOME/cache"
export XDG_STATE_HOME="\$PIXI_HOME/state"
# h100a's /tmp (100 GB) filled up on 2026-11-06 ~22:40 and killed five gates
# (Errno 28 in the Triton cache); every cache goes to the node's NVMe instead.
export TMPDIR=\$PROJECT_HOME_PATH/\$USER/parity_tmp/\$SLURM_JOB_ID
mkdir -p \$TMPDIR
export TRITON_CACHE_DIR=\$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=\$TMPDIR/inductor
cd "\$PIXI_HOME" && eval "\$(pixi shell-hook)" && cd - > /dev/null
cd ~/parity/code/$SHORT
echo "parity_audit commit \$(cat PARITY_COMMIT), job \$SLURM_JOB_ID on \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$(printf '%s\n' "$*" | sed 's/ :: /\n/g' | while read -r cmd; do echo "python documents/scripts/parity_numerics.py $cmd --out ~/parity/out/${NAME}_\$SLURM_JOB_ID || echo \"CHECK FAILED: $cmd\""; done)
SBATCH
