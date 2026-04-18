srun --time=00:30:00 --ntasks=1 --cpus-per-task=1 --pty /bin/bash --login
source ~/miniforge3/bin/activate bn-medical
python benchmark_memory.py