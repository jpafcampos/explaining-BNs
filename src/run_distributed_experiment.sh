#!/bin/bash
#SBATCH --job-name=sdp__parallel_experiment
#SBATCH --output=results/logs/parallel_experiment_%j.out
#SBATCH --error=results/logs/parallel_experiment_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36

source ~/miniforge3/bin/activate bn-medical
mkdir -p /projects/b36ag/explaining-BNs/results/logs

python src/synthetic_experiment_parallel.py \
    --bif-dir ./generated_bif_files/ \
    --output results/final_results.csv \
    --n-workers 36