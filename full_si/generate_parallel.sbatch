#!/bin/bash
#
#SBATCH -c 1
#SBATCH -J vaccine_abc
#SBATCH -p serial_requeue
#SBATCH -t 0-01:30
#SBATCH --mem 300M


module load Anaconda3/2019.10
source activate vaccine_abc
python simple_generate.py ${SLURM_ARRAY_TASK_ID} $1 $2
source deactivate
