#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=base_noDSM_s43
#SBATCH --output=slurm-%j.out

python train.py  --suffix_index 2 --seed 43 --resume --no-evaluate_during_training
