#!/bin/bash
#SBATCH --job-name=memory_test
#SBATCH --output=memory_test.out
#SBATCH --error=memory_test.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --mem=128G

source ~/miniforge3/bin/activate bn-medical

export PYTHONUNBUFFERED=1

python memory_test.py