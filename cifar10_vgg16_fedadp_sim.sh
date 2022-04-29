#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fedadp_vgg16.out


CUDA_LAUNCH_BLOCKING=1 python examples/fedadp/fedadp.py  -c examples/fedadp/fedadp_CIFAR10_vgg16.yml
