#!/bin/bash
#SBATCH --job-name=anysat
#SBATCH --partition=gpu,gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/anysat_%j.out
#SBATCH --error=logs/anysat_%j.err

set -e

mkdir -p logs anysat

# Activate the environment that has rs-embed installed (see embeddings/README.md)
conda activate livability

# Some rs-embed backends download model weights from the Hugging Face Hub.
export HF_TOKEN="${HF_TOKEN:?Set your Hugging Face token first: export HF_TOKEN=hf_...}"

# Run from the directory containing make_anysat.py and the RS_*.tif input grids
cd "$(dirname "$0")"

python make_anysat.py \
    --out-dir     anysat \
    --year        2020 \
    --buffer-m    250 \
    --workers     12 \
    --device      cuda
