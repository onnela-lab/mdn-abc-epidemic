#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem 50G


module load Anaconda3/2019.10
source activate vaccine_abc
python calculate_coverage.py $1 $2
source deactivate

