#!/bin/bash
#SBATCH --job-name=random_experiment
#SBATCH --output=results/logs/random_experiment_%j.out
#SBATCH --error=results/logs/random_experiment_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=140G

source ~/miniforge3/bin/activate bn-medical
mkdir -p results/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python random_patient_script.py \
    --bif-dir ./generated_bif_files/ \
    --output results/final_results_ratio75_$SLURM_JOB_ID.csv \
    --n-workers 1