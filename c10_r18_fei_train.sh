#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=train.out

python examples/fei/fei.py  -c examples/fei/configs/c10_r18_train.yml
