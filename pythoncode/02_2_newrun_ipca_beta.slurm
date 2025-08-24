#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --job-name=ipca_beta
#SBATCH --array=5   # Runs regs K = 1 to 6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=950G
#SBATCH --partition=highmem

# Load necessary modules 
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtual environment
source ~/pyvenv/bin/activate

# use path to import ipca_utils.py functions
export PYTHONPATH=/home/jfriasna/thesis_code/pythoncode:$PYTHONPATH

# Get timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Redirect output and error
exec > logs/new_reg_beta/ipca_K${SLURM_ARRAY_TASK_ID}_${timestamp}.out 2>&1
#exec 2> testlogs/reg_ipca_K${SLURM_ARRAY_TASK_ID}_${timestamp}.err

# Run your script, unbuffered -u to display output on the go
srun python -u 02_2_newrun_ipca_beta.py $SLURM_ARRAY_TASK_ID
