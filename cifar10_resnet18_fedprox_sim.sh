#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fedprox_resnet18.out


CUDA_LAUNCH_BLOCKING=1 python examples/fedprox/fedprox.py  -c examples/fedprox/fedprox_CIFAR10_resnet18.yml
