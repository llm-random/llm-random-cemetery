#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from somewhere in llm-random directory
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the path to config file as the second argument (e.g. "runs/my_config.yaml")
# EXAMPLE USAGE: bash lizrd/scripts/run_exp_remotely.sh atena lizrd/configs/quick.json
set -e
regex_branch="<<'(.*)'>>"
regex_directory="##*(.*)*##"

# source venv/bin/activate
# python3 -m lizrd.support.sync_code --host $1

run_grid_remotely() {
  host=$1
  config=$2
  
  output=$(python3 -m lizrd.support.code_versioning --config $config)
  echo $output
  if [[ $output =~ $regex_branch ]]; then
    experiment_branch="${BASH_REMATCH[1]}"
  else 
    echo "Could not find experiment branch. Exiting."
    exit 1
  fi
  if [[ $output =~ $regex_directory ]]; then
    experiment_directory="${BASH_REMATCH[1]}"
  else 
    echo "Could not find experiment directory. Exiting."
    exit 1
  fi
  echo "Experiment branch: $experiment_branch"
  echo "Experiment directory: $experiment_directory"

  script="cd $experiment_directory && tmux new-session -d -s $experiment_branch bash"
  script+="; tmux send-keys -t $experiment_branch '"
  if [ -n "$NEPTUNE_API_TOKEN" ]; then
    script+="NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN "
  fi
  if [ -n "$WANDB_API_KEY" ]; then
    script+="WANDB_API_KEY=$WANDB_API_KEY"
  fi
  script+="./run_experiment.sh' C-m"
  script+="; tmux attach -t $experiment_branch"
  script+="; echo 'done'" #black magic: without it, interactive sessions like "srun" cannot be detached from without killing the session

  # ssh -t $host "$script"
  echo $script
}

for i in "${@:2}"
do
  # run your bash function
run_grid_remotely $1 $i
done

