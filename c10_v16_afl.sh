#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=afl_c10_v16.out

ython examples/afl/afl.py  -c examples/afl/c10_v16.yml
