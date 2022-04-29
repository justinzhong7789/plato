#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=fedavg_vgg16.out


CUDA_LAUNCH_BLOCKING=1 ./run -c configs/CIFAR10/fedavg_vgg16.yml
