#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=print.out


module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/fedatt/fedatt.py -c examples/fedatt/fedatt_CIFAR10_resnet18.yml
