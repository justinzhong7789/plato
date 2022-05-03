#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=fedadp_c10_v16.out

python examples/fedadp/fedadp.py  -c examples/fedadp/c10_v16.yml
