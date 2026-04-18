#!/bin/bash
#SBATCH --job-name=sdp_ultra_parallel
#SBATCH --output=results/logs/parallel_experiment_%j.out
#SBATCH --error=results/logs/parallel_experiment_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --mem=0

source ~/miniforge3/bin/activate bn-medical
mkdir -p results/logs

export PYTHONUNBUFFERED=1

python synthetic_experiment_parallel.py \
    --bif-dir ./generated_bif_files/ \
    --output results/final_results_135.csv \
    --n-workers 135