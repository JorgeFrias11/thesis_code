#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --job-name=ipca_reg
#SBATCH --array=1-6   # Runs regs K = 1 to 8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=225G
#SBATCH --partition=gpu

# Load necessary modules 
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtual environment
source ~/pyvenv/bin/activate

# use path to import ipca_utils.py functions
export PYTHONPATH=/home/jfriasna/thesis/pythoncode:$PYTHONPATH

# Get timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Redirect output and error
exec > logs/elastic/test_ipca_K${SLURM_ARRAY_TASK_ID}_${timestamp}.out 2>&1
#exec 2> testlogs/reg_ipca_K${SLURM_ARRAY_TASK_ID}_${timestamp}.err


# Run your script, unbuffered -u to display output on the go
srun python -u newdata_check_ipca.py $SLURM_ARRAY_TASK_ID

