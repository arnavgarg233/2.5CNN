#!/bin/bash

# Usage: ./launch_job_on_gpu.sh <job_command> <gpu_index> <git_branch>

job_command=$1
gpu_index=$2
git_branch=$3

conda activate pytorch

# Fetch the specific branch from the 'versions' remote
git fetch versions "$git_branch"

# Checkout to the specified branch
git checkout "$git_branch"

# Set the environment variable to use a specific GPU
export CUDA_VISIBLE_DEVICES="$gpu_index"

# Launch the job
eval "$job_command"
echo "Launched job on GPU $gpu_index"

# Wait for a short duration
sleep 5

# Return to the master branch
git checkout master
