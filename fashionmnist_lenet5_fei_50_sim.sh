#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fei_lenet5_50.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py  -c examples/fei/configs/FashionMNIST/fei_lenet5_50.yml
