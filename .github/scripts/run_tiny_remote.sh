#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --job-name=tiny_remote_ci
#SBATCH --time=00:10:00

# Load environment (adjust based on your entropy setup)
# Example: module load python/3.11 cuda/12.1
# Or if using pixi:
# eval "$(pixi shell-hook)"

# Run the training
python main.py --config-path=configs --config-name=tiny_remote
