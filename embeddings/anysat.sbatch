#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=cpu
#SBATCH --gpus=0
#SBATCH --mem=100G
#SBATCH --job-name=anysat
#SBATCH --output=logs/anysat_%j.out
#SBATCH --error=logs/anysat_%j.err

mkdir -p logs anysat

conda activate livability
python make_anysat.py \
    --out-dir  anysat \
    --year     2020 \
    --buffer-m 255 \
    --workers  28 \
    --device   auto

