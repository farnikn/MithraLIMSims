#!/bin/bash

#SBATCH -p cca
#SBATCH --job-name=SM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=farnik.nikakhtar@gmail.com
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/mnt/home/fnikakhtar/lim/HiddenValleySims-master/analysis/
#SBATCH --output=/mnt/home/fnikakhtar/lim/HiddenValleySims-master/log/savemesh_210719_%j.log


source /mnt/home/fnikakhtar/.bashrc
source activate idp

time python -u savemesh.py params_savemesh.yml
