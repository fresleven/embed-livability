#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=as_s2
#SBATCH --output=slurm-%j.out

python train.py --anysat --suffix_index 2 --resume --no-evaluate_during_training
