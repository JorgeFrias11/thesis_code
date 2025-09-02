#!/bin/bash
#SBATCH --job-name=delta_test
#SBATCH --array=5              # 5 tasks, one per observable factor
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --partition=gpu

# Load necessary modules 
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtual environment
source ~/pyvenv/bin/activate

# Add your code directory to PYTHONPATH
export PYTHONPATH=/home/jfriasna/thesis_code/pythoncode:$PYTHONPATH

# Get timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

exec > logs/delta_test/delta_${SLURM_ARRAY_TASK_ID}_${timestamp}.out 2>&1

# Run the Python script
srun python -u 06_delta_test.py $SLURM_ARRAY_TASK_ID
