#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --output=print.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py -c examples/fei/configs/CIFAR10/fei_resnet18_train.yml
