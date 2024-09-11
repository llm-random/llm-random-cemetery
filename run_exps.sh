#!/bin/bash

# Folder containing the config files
CONFIG_FOLDER="configs/experiments/relative_lr/medium_local_minimum/"

# Loop through all config files in the folder
for config in "$CONFIG_FOLDER"*
do
  # Execute the python script with the current config file
  echo $config
  python submit_experiment.py --host athena --config "$config" --skip_confirmation "True"
  
  # Optionally, you can add a sleep interval to avoid overwhelming the system with too many requests at once
  # sleep 1
done
