#!/bin/bash
# INSTRUCTIONS:
# 1. needs to be called from somewhere in llm-random directory
# 2. needs to be called with the host as the first argument (as configured in ~/.ssh/config, e.g. gpu_entropy)
# 3. needs to be called with the path to config file as the second argument (e.g. "runs/my_config.yaml")
# EXAMPLE USAGE: bash lizrd/scripts/run_exp_remotely.sh atena lizrd/configs/quick.json
set -e
regex="<<'(\K.*?)(?='>>')"

# source venv/bin/activate
# python3 -m lizrd.support.sync_code --host $1

run_grid_remotely() {
  host=$1
  config=$2
  echo "Running grid search on $host with config $config"

  # echo "Extracted text: $experiment_branch"
  output=$(python3 -m lizrd.support.code_versioning --config $config)
  regex="<<'(.*)'>>"

  if [[ $output =~ $regex ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
  # experiment_branch=$(echo "$output" | sed -n "s/$regex/\1/p")
  # echo "TOJESTTEST: $experiment_branch"
  # script="cd $base_dir && tmux new-session -d -s $session_name bash"
  # script+="; tmux send-keys -t $session_name 'python3 -m lizrd.grid --config_path=$config --git_branch=$git_branch"
  # if [ -n "$NEPTUNE_API_TOKEN" ]; then
  #   script+=" --neptune_key=$NEPTUNE_API_TOKEN"
  # fi
  # if [ -n "$WANDB_API_KEY" ]; then
  #   script+=" --wandb_key=$WANDB_API_KEY"
  # fi
  # script+="' C-m"
  # script+="; tmux attach -t $session_name"
  # script+="; echo 'done'" #black magic: without it, interactive sessions like "srun" cannot be detached from without killing the session

  # ssh -t $host "$script"
}

for i in "${@:2}"
do
  # run your bash function
run_grid_remotely $1 $i
done

