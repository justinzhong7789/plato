#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=afl_vgg16.out


CUDA_LAUNCH_BLOCKING=1 python examples/afl/afl.py  -c examples/afl/afl_CIFAR10_vgg16.yml
