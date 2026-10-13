#!/bin/bash

# --- SLURM Job Directives ---
#SBATCH --job-name=fl_case1_scholar    # Name of your job
#SBATCH --nodes=1                      # Request one node
#SBATCH --ntasks=1                     # Run a single task (your python script)
#SBATCH --partition=gpu_scholar        # ** THE FIX: Use the new partition **
#SBATCH --cpus-per-task=8              # Request 8 CPU cores
#SBATCH --gres=gpu:1                   # Request only 1 GPU
#SBATCH --time=24:00:00                # Request 24 hours
#SBATCH --output=fl_case1_scholar_%j.out # Log file for output
#SBATCH --error=fl_case1_scholar_%j.err  # Log file for errors

# --- Your Job's Commands ---

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "Using partition: ${SLURM_JOB_PARTITION}"

# Add your locally installed packages to the PATH
export PATH="/dgxa_home/se24maid001/.local/bin:$PATH"

# Check if the GPU is visible
echo "NVIDIA-SMI output:"
nvidia-smi

# Run your Python script
# Use the full, absolute path
echo "Starting Federated Learning script..."
python3 /dgxa_home/se24maid001/Federated_learning/Base_Fed.py

echo "Script finished."
echo "Job finished on $(date)"