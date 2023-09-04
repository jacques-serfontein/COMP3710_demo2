#!/bin/bash
time=0-01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=45339961_test
#SBATCH --cpus-per-task 1
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

conda activate /home/Student/s4533996/conda-torch
python test.py