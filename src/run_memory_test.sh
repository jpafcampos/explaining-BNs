#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=results/logs/benchmark.out
#SBATCH --error=results/logs/benchmark.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=128G

source ~/miniforge3/bin/activate bn-medical

export PYTHONUNBUFFERED=1

python benchmark_memory.py