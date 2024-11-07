#!/bin/bash

#SBATCH -A es_compmech
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --gres=gpumem:74512m
#SBATCH --time=24:00:00
#SBATCH --job-name="z252l256"
#SBATCH --mem-per-cpu=12288
#SBATCH --mail-type=END
#SBATCH --mail-user=jian.tang@empa.ch

module load gcc/8.2.0 python_gpu/3.11.2
module load cuda

python mgpu_ppre_t_grad.py

