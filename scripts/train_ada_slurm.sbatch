#!/bin/bash
# ============================================================
# Slurm training script
# Chest X-Ray Multi-Class Classification with PyTorch
# ============================================================
#
# This script is an example of how the training pipeline can be
# executed in a GPU/HPC environment using Slurm.
#
# The script was designed for research workflows where experiments
# need to run for several hours, saving logs, checkpoints, metrics
# and reproducible outputs in a structured folder.
#
# NOTE:
# Adjust partition, account, modules and environment activation
# depending on the specific HPC cluster configuration.
# ============================================================

#SBATCH --job-name=xray_effb7_768
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# If your cluster requires a partition or account, uncomment and adapt:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account_name

set -euo pipefail

echo "============================================================"
echo "Slurm job started"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job name: ${SLURM_JOB_NAME:-N/A}"
echo "Node list: ${SLURM_JOB_NODELIST:-N/A}"
echo "Submit directory: ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Dataset must follow ImageFolder structure:
# DATA_DIR/
#   train/
#   val/
#   test/
DATA_DIR="${DATA_DIR:-/path/to/chest_xray_dataset}"

CONFIG_FILE="${CONFIG_FILE:-configs/efficientnet_b7_768.yaml}"

RUN_NAME="${RUN_NAME:-efficientnet_b7_768_slurm_${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"

LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-}"

mkdir -p logs/slurm
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

echo "Project directory: ${PROJECT_DIR}"
echo "Dataset directory: ${DATA_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
#
# Adapt this block to your HPC environment.
# Examples:
#
# module purge
# module load python/3.10
# module load cuda/11.8
#
# source ~/.bashrc
# conda activate xray-dl
#
# Or with a virtual environment:
# source .venv/bin/activate

if [ -d ".venv" ]; then
    echo "Activating local virtual environment: .venv"
    source .venv/bin/activate
else
    echo "No .venv folder found. Using current Python environment."
fi

echo "============================================================"
echo "Python and CUDA information"
echo "============================================================"
which python
python --version

python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "============================================================"
echo "Installing package in editable mode"
echo "============================================================"
pip install -e .

echo "============================================================"
echo "Starting training"
echo "============================================================"

TRAIN_CMD=(
    python -m raravis_dl.train
    --config "${CONFIG_FILE}"
    --data-dir "${DATA_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --evaluate-test-at-end
)

if [ -n "${LOCAL_WEIGHTS}" ]; then
    echo "Using local pretrained weights: ${LOCAL_WEIGHTS}"
    TRAIN_CMD+=(--local-weights "${LOCAL_WEIGHTS}")
fi

echo "Command:"
printf '%q ' "${TRAIN_CMD[@]}"
echo

"${TRAIN_CMD[@]}"

echo "============================================================"
echo "Training finished"
echo "============================================================"

BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoints/best_model.pt"

if [ -f "${BEST_CHECKPOINT}" ]; then
    echo "Best checkpoint found: ${BEST_CHECKPOINT}"
    echo "Running final explicit test evaluation..."

    python -m raravis_dl.evaluate \
        --config "${CONFIG_FILE}" \
        --data-dir "${DATA_DIR}" \
        --checkpoint "${BEST_CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}/test_eval" \
        --split test
else
    echo "WARNING: best checkpoint not found at ${BEST_CHECKPOINT}"
    echo "Skipping explicit final evaluation."
fi

echo "============================================================"
echo "Experiment outputs"
echo "============================================================"
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort || true

echo "============================================================"
echo "Slurm job finished"
echo "End time: $(date)"
echo "============================================================"
