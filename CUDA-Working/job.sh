#!/bin/bash
#SBATCH --job-name="CNN"
#SBATCH --output="cuda_V100.log"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --error="cuda_error.log"
module load CUDA
make
./CNN



