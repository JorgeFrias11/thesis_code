#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --job-name=oos_ipca
#SBATCH --array=1-6     # K = 1,...,6 latent factors
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=250G
#SBATCH --partition=short

# Load necessary modules 
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtual environment
source ~/pyvenv/bin/activate

# use path to import ipca_utils.py functions
export PYTHONPATH=/home/jfriasna/thesis_code/pythoncode:$PYTHONPATH


# Redirect output
exec > logs/yearmissing/oos_pred/latent/fit_${SLURM_ARRAY_TASK_ID}.out 2>&1

# Run python script
srun python -u 12_oos_yearmissing.py $SLURM_ARRAY_TASK_ID