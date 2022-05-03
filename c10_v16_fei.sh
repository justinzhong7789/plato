#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=fei_c10_v16.out

python examples/fei/fei.py  -c examples/fei/configs/c10_v16.yml
