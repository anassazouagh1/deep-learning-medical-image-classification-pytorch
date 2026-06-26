#!/bin/bash
#SBATCH --job-name=effb7_xray_768
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs runs

module purge || true
module load cuda || true

source .venv/bin/activate

python -m raravis_dl.train \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir "$DATA_DIR" \
  --output-dir runs/efficientnet_b7_768
