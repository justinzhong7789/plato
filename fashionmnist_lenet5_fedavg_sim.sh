#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=fedavg_lenet5.out


CUDA_LAUNCH_BLOCKING=1 ./run -c configs/FashionMNIST/fedavg_lenet5.yml
