#!/bin/bash
#
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-08:00
#SBATCH --mem 150G


module load Anaconda3/2019.10
source activate vaccine_abc
python -u whisker_plots.py $1 $2
source deactivate

