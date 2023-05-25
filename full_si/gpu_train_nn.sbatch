#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-10:00
#SBATCH --mem=80G

module load Anaconda3/2019.10
module load cuda/11.7.1-fasrc01
source activate vaccine_abc

srun -c 1 --gres=gpu:1 python gpu_train_nn.py
source deactivate

