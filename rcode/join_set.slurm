#!/bin/bash
#SBATCH --job-name=build_pred
#SBATCH --output=logs/join.log
#SBATCH --error=logs/join.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=compute-test

export R_LIBS=~/R_LIBS
echo "Running job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Allocated hosts: $SLURM_JOB_NODELIST"

. /opt/apps/2023-04-11_lmod.bash
module purge
module load r-4.5.1-gcc-12.2.0-tfchyxd

srun Rscript join_set.R

