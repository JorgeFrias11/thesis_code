#!/bin/bash
#SBATCH --job-name=oos_ipca
#SBATCH --array=6     # K = 1,...,6 latent factors
#SBATCH --nodes=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=100G
#SBATCH --partition=gpu

# Load Python
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtualenv
source ~/pyvenv/bin/activate

# Add code path
export PYTHONPATH=/home/jfriasna/thesis_code/pythoncode:$PYTHONPATH

# Get timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

exec > logs/oos_pred/oos_${SLURM_ARRAY_TASK_ID}_${timestamp}.out 2>&1

# Run
srun python -u 07_oos_pred.py $SLURM_ARRAY_TASK_ID
