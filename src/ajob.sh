#!/bin/bash

#SBATCH -A es_compmech
#SBATCH -n 1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=a100_80gb:2
#SBATCH --time=24:00:00
#SBATCH --job-name="5l256-dp5"
#SBATCH --mem-per-cpu=12288
#SBATCH --mail-type=END
#SBATCH --mail-user=jian.tang@empa.ch

module load stack/2024-06
module load gcc/12.2.0 python_cuda/3.11.6
module load cuda

python mgpu_ppre_t_grad.py

