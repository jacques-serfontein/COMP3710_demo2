#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=45339961_test
#SBATCH --cpus-per-task 1
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

# source /home/Student/s4533996/miniconda3/bin/activate conda-torch
conda activate conda-torch

python ~/COMP3710_demo2/Q2P2_rangpur.py