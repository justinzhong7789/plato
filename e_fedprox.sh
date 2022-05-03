#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=fedprox_e.out

python examples/fedprox/fedprox.py -c examples/fedprox/e.yml
