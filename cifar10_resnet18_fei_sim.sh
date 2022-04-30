#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fei_resnet18.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py  -c examples/fei/configs/CIFAR10/fei_resnet18.yml
