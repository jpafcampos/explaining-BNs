#!/bin/bash
#SBATCH --job-name=sdp_benchmark_experiment
#SBATCH --output=results/logs/benchmark_experiment_%j.out
#SBATCH --error=results/logs/benchmark_experiment_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G

source ~/miniforge3/bin/activate bn-medical
mkdir -p results/logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python benchmark_experiments_script.py