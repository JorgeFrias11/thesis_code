#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --job-name=factors_reg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=120G
#SBATCH --partition=compute

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
exec > logs/04_reg_obsfactors.out 2>&1

# Run script unbuffered -u to display output 
srun python -u 04_ipca_obsfactors.py $SLURM_ARRAY_TASK_ID
