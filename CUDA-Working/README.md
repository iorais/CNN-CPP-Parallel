### Architecture
All tests performed on an Nvidia Tesla V100 GPU
Training with 1 epoch: 85% accuracy
Total GPU Time 5.47s
Wall Clock Training Time 9.9s

### Compiling and Execution
On WAVE HPC
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --nodes=1 --pty /bin/bash

module load CUDA
To compile just navigate to root and type `make`
Executable can be run using `./CNN`

or 

sbatch job.sh



