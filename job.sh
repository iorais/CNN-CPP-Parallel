#!/bin/bash
#SBATCH --job-name="CNN-cpp"
#SBATCH --output="logs/CNN.%j.%N.log"
#SBATCH --error=errors/CNN.%j.err
#SBATCH --partition=cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL

module load GCC
make
./bin/CNN