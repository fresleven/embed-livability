#!/bin/bash
#SBATCH --account=bcrm-tgirails
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --job-name=terramind
#SBATCH --output=logs/terramind_%j.out
#SBATCH --error=logs/terramind_%j.err

mkdir -p logs terramind

conda activate livability
python make_terramind.py \
    --out-dir  terramind \
    --year     2020 \
    --buffer-m 255 \
    --workers  28 \
    --device   cuda

