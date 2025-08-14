#!/bin/bash
#SBATCH --job-name=build_pred
#SBATCH --output=logs/03_clean_100mill.log
#SBATCH --error=logs/03_clean_100mill.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=highmem

export R_LIBS=~/R_LIBS
echo "Running job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Allocated hosts: $SLURM_JOB_NODELIST"

. /opt/apps/2023-04-11_lmod.bash
module purge
module load r-4.4.1-gcc-12.2.0-x6ecfp6

srun Rscript 03_clean_data.R


