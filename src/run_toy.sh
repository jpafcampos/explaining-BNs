#!/bin/bash
#SBATCH --job-name=toy_experiment
#SBATCH --output=results/logs/experiment_%j.out
#SBATCH --error=results/logs/experiment_%j.err
#SBATCH --ntasks=1

source ~/miniforge3/bin/activate bn-medical

mkdir -p results/logs

python synthetic_experiments_script.py \
    --toy \
    --output results/toy_experiment_$SLURM_JOB_ID.csv