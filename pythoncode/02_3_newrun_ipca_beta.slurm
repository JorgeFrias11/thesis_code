#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --array=4   # 4 jobs, chunks of 10 chars each
#SBATCH --nodes=1
#SBATCH --job-name=6_beta
#SBATCH --cpus-per-task=100
#SBATCH --mem=850G
#SBATCH --partition=highmem

# DEFINE K!
K=6

# Load necessary modules 
module purge
module load python-3.11.9-gcc-12.2.0-2j6qzmv

# Activate virtual environment
source ~/pyvenv/bin/activate

# use path to import ipca_utils.py functions
export PYTHONPATH=/home/jfriasna/thesis_code/pythoncode:$PYTHONPATH

# Get timestamp
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Chunk size
CHUNK=10
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

# Cap END at 41 (exclusive upper bound)
if [ $END -gt 41 ]; then
  END=41
fi

# Redirect output and error
exec > logs/new_reg_beta/ipca_K${K}_${START}to${END}_${timestamp}.out 2>&1

# Run your script, unbuffered -u to display output on the go
srun python -u 02_3_newrun_ipca_beta.py $K $START $END
