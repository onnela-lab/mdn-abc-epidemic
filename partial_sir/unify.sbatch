#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-2:00
#SBATCH --mem 180G


module load Anaconda3/2019.10
source activate vaccine_abc
python unify.py $1 $2
source deactivate

