#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=print.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py -c examples/fei/configs/CIFAR10/fei_resnet18.yml
