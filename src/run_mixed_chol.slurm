#!/bin/bash
#SBATCH --job-name=mixed_cholesky
#SBATCH --output=cholesky_%j.out
#SBATCH --error=cholesky_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=accel  # Adjust based on Alps GPU partition name
#SBATCH --constraint=gh200  # Optional: restrict to GH200 nodes if required

#start uenv 
uenv start prgenv-gnu/25.06:rc4 --view=modules


# Load required modules (adjust for your Alps environment)
module load cuda gcc cmake

# Run one task per GPU
srun --gpu-bind=closest ./mixed_cholesky
