#!/bin/bash

#SBATCH --wckey=p19505
#SBATCH --job-name=pretrain_stella_mlm_b100
#SBATCH --partition=A800
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=50g
#SBATCH --gres=gpu:a800:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=peg2_wu@163.com

module load cuda/12.4

source /share/home/u19505/apps/miniconda3/etc/profile.d/conda.sh

conda activate stella

srun sh pretrain.sh