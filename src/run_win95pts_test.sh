#!/bin/bash
#SBATCH --job-name=win95pts_test
#SBATCH --output=results/logs/win95.out
#SBATCH --error=results/logs/win95.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G

source ~/miniforge3/bin/activate bn-medical

export PYTHONUNBUFFERED=1

python win95pts_test.py