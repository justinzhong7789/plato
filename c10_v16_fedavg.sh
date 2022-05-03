#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=fedavg_c10_v16.out

./run -c configs/CIFAR10/c10_v16.yml
