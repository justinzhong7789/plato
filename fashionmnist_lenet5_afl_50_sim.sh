#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=afl_lenet5_50.out


CUDA_LAUNCH_BLOCKING=1 python examples/afl/afl.py  -c examples/afl/afl_FashionMNIST_lenet5_50.yml
