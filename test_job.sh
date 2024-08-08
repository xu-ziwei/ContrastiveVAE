#!/bin/bash

### slurm job options: line must begin with '#SBATCH'

#SBATCH --job-name=test_job    # job name
#SBATCH --mail-type=END,FAIL   # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ziwei.xu@sund.ku.dk # email address to receive the notification    
#SBATCH --ntasks=1             # run a single task
#SBATCH --cpus-per-task=1      # number of requested cores
#SBATCH --mem=4gb              # total requested RAM
#SBATCH --time=10:00:00        # max. running time of the job, format in D-HH:MM:SS
#SBATCH --output=test_job_%j.log  # standard output and error log, '%j' gives the job ID
#SBATCH -p gpuqueue            # specify the GPU queue
#SBATCH --gres=gpu:1           # request 1 GPU

### write your own scripts below 

# Load the conda environment
module load dangpu_libs miniconda/latest
conda activate vae_env

# Print environment information
echo "Running on node:"
hostname
echo "Using GPU(s):"
nvidia-smi
echo "Python version:"
python --version

# Simple Python test script
python -c "print('Hello from Python')"
