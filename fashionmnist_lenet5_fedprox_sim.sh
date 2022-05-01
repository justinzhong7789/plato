#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fedprox_lenet5.out


CUDA_LAUNCH_BLOCKING=1 python examples/fedprox/fedprox.py  -c examples/fedprox/fedprox_FashionMNIST_lenet5.yml
