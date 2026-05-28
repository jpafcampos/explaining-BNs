#!/bin/bash
#SBATCH --job-name=threshold_test
#SBATCH --output=results/logs/threshold_test.out
#SBATCH --error=results/logs/threshold_test.err
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --mem=140G

source ~/miniforge3/bin/activate bn-medical

export PYTHONUNBUFFERED=1

python threshold_test.py