#!/bin/bash
#SBATCH --job-name=terramind
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/terramind_%j.out
#SBATCH --error=logs/terramind_%j.err

#set -e

mkdir -p logs terramind

# Activate the environment that has rs-embed installed (see embeddings/README.md)
conda activate livability

# Some rs-embed backends download model weights from the Hugging Face Hub.
export HF_TOKEN="${HF_TOKEN:?Set your Hugging Face token first: export HF_TOKEN=hf_...}"

# Run from the directory containing make_terramind.py and the RS_*.tif input grids
cd "$(dirname "$0")"

python make_terramind.py \
    --out-dir     terramind \
    --year        2020 \
    --buffer-m    250 \
    --workers     28 \
    --device      cuda \
