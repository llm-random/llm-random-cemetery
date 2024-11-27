#!/bin/bash -l

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Replace placeholders in command-line arguments
args=()
for arg in "$@"; do
    arg="${arg//__HEAD_NODE_IP__/$head_node_ip}"
    arg="${arg//__RANDOM__/$RANDOM}"
    args+=( "$arg" )
done

module load ML-bundle/24.06a
source /net/storage/pr3/plgrid/plggllmeffi/datasets/make_singularity_image/venv/bin/activate
echo "Will run the following command:"
echo "${args[@]}"
echo "==============================="
"${args[@]}"