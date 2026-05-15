#!/bin/bash
#SBATCH --job-name=memory_test
#SBATCH --output=results/logs/memory_test.out
#SBATCH --error=results/logs/memory_test.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=140G

source ~/miniforge3/bin/activate bn-medical

export PYTHONUNBUFFERED=1

python memory_test.py