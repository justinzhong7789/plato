#!/bin/bash 
#SBATCH --time=5:00:00 
#SBATCH --cpus-per-task=21 
#SBATCH --gres=gpu:1 
#SBATCH --mem=100G 
#SBATCH --output=./slurm_loggings/contrasAdap_MNIST_lenet5.out 
 
/home/sijia/envs/miniconda3/envs/INFOCOM23/bin/python ./examples/contrastive_adaptation/contrasAdap/contrasAdap.py -c ./examples/contrastive_adaptation/configs/contrasAdap_MNIST_lenet5.yml -b /data/sijia/INFOCOM23/experiments 
