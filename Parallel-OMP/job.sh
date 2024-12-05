#!/bin/bash
#SBATCH --job-name="CNN-cpp"
#SBATCH --output="logs/CNN.%j.%N.log"
#SBATCH --error=errors/CNN.%j.err
#SBATCH --partition=cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH -t 08:00:00
#SBATCH --mail-type=ALL

#
# Run the following command to submit a job:
#
# sbatch --mail-user=<user@email.com> job.sh
#

module load GCC
make
for threads in {1,2,3,4,5,6,7,8,12,14,16,18,20,22,24}; do
  export OMP_NUM_THREADS=$threads
  bin/CNN --num_epochs 1 --preview_period 100000
done