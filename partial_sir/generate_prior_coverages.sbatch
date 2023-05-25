#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-30:00
#SBATCH --mem 200G


module load Anaconda3/2019.10
source activate vaccine_abc
python prior_calculate_coverages.py $1 
source deactivate

