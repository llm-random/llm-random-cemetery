#!/bin/bash

# Folder containing the config files
CONFIG_FOLDER="configs/experiments/relative_lr/medium_local_minimum/"

# Initialize a counter
counter=1

# Loop through all config files in the folder
for config in "$CONFIG_FOLDER"*
do
  # Print the loop number
  echo "Loop number: $counter"
  
  # Execute the python script with the current config file
  echo $config
  python submit_experiment.py --host writer --config "$config"
  
  # Increment the counter
  ((counter++))

  # Optionally, you can add a sleep interval to avoid overwhelming the system with too many requests at once
  # sleep 1
done
