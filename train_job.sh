#!/bin/bash

#SBATCH --job-name=vae_training    # Job name
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ziwei.xu@sund.ku.dk # Where to send mail
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --mem=256gb                # Job memory request
#SBATCH --time=5:00:00            # Time limit hrs:min:sec
#SBATCH --output=vae_training_%j.log  # Standard output and error log
#SBATCH -p gpuqueue                # Specify the GPU queue
#SBATCH --gres=gpu:2               # Request three GPUs

# Load the conda environment
module load dangpu_libs miniconda/latest
source ~/.bashrc
conda activate vae_env

# Change to the directory with the training script
cd $HOME/ContrastiveVAE

# Run the training script
python src/training/train.py config/config.yaml
