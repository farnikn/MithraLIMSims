#!/bin/bash

#SBATCH -p cca
#SBATCH --job-name=HV
#SBATCH --mem=1000gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=farnik.nikakhtar@gmail.com
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/mnt/home/fnikakhtar/lim/HiddenValleySims-master/analysis/
#SBATCH --output=/mnt/home/fnikakhtar/lim/HiddenValleySims-master/log/hodgal_210719_%j.log


source /mnt/home/fnikakhtar/.bashrc
source activate idp

echo "Running script on a single CPU core"
time python -u hodgal.py params_hodgal.yml
