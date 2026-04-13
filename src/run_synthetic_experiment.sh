#!/bin/bash
#SBATCH --job-name=syntetic_experiment
#SBATCH --output=results/logs/experiment_%j.out
#SBATCH --error=results/logs/experiment_%j.err
#SBATCH --ntasks=1

source ~/miniforge3/bin/activate my-experiment-env

mkdir -p results/logs

python synthetic_experiments_script.py \
    --bif-dir ./generated_bif_files \
    --output results/experiment_$SLURM_JOB_ID.csv