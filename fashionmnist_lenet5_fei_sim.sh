#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=fei_lenet5.out


CUDA_LAUNCH_BLOCKING=1 python examples/fei/fei.py  -c examples/fei/configs/FashionMNIST/fei_lenet5.yml
