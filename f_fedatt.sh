#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=fedatt_f.out

python examples/fedatt/fedatt.py  -c examples/fedatt/f.yml
