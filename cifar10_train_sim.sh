#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=print.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py -c examples/fei/configs/CIFAR10/fei_resnet18_train.yml
