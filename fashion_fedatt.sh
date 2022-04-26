#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=print.out


module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/fedatt/fedatt.py -c examples/fedatt/fedatt_FashionMNIST_lenet5.yml
