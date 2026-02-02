#!/bin/bash -l

#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=scaling_eval
#SBATCH --mem-per-gpu=90G
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=1-00:00:00

#---------- SCRIPT ----------
ml CUDA/12.4.0
export PROJECT_HOME_PATH=/lustre/pd01/plgrid/plgllmefficont2/nano/context_scaling
export HF_HOME=$PROJECT_HOME_PATH/hf_cache
export HYDRA_FULL_ERROR=1
export PIXI_HOME=$PROJECT_HOME_PATH/pixi
export PATH="$HOME/.pixi/bin:$PATH"
export XDG_DATA_HOME="$PROJECT_HOME_PATH/data"
export XDG_CACHE_HOME="$PROJECT_HOME_PATH/cache"
export XDG_STATE_HOME="$PROJECT_HOME_PATH/state"
cd "$PIXI_HOME"
eval "$(pixi shell-hook)"
cd -
#-------- SCRIPT END --------


export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    export MASTER_PORT=$((40000 + ${SLURM_JOB_ID} % 10000))
else
    export MASTER_PORT=$((30000 + (${SLURM_JOB_ID} % 1250) * 8 + (${SLURM_ARRAY_TASK_ID} % 8)))
fi
srun torchrun --nnodes=${SLURM_NNODES}\
  --nproc-per-node=${SLURM_GPUS_ON_NODE} \
  --rdzv-id=${SLURM_JOBID} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  main.py \
    --config-path=generated_configs \
    --config-name=config_${SLURM_ARRAY_TASK_ID}.yaml \
    +checkpoint_config.slurm_array_task_id=${SLURM_ARRAY_TASK_ID}