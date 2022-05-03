#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=afl_f.out

python examples/afl/afl.py  -c examples/afl/f.yml
