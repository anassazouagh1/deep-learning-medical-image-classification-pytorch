#!/bin/bash
# ============================================================
# Local training script
# Chest X-Ray Multi-Class Classification with PyTorch
# ============================================================
#
# This script provides a simple local execution workflow for the
# training pipeline.
#
# It is useful for:
#   - Testing the code before submitting a Slurm job.
#   - Running small experiments locally.
#   - Checking that the dataset structure is correct.
#   - Verifying that training, evaluation and output folders work.
#
# Expected dataset format:
#   DATA_DIR/
#     train/
#     val/
#     test/
#
# Each split must contain one folder per class.
# ============================================================

set -euo pipefail

echo "============================================================"
echo "Local training script"
echo "============================================================"

# ------------------------------------------------------------
# User configuration
# ------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/path/to/chest_xray_dataset}"
CONFIG_FILE="${CONFIG_FILE:-configs/efficientnet_b7_768.yaml}"
RUN_NAME="${RUN_NAME:-efficientnet_b7_768_local}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-}"

echo "Dataset directory: ${DATA_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Dataset directory does not exist: ${DATA_DIR}"
    echo "Please set DATA_DIR before running the script."
    echo ""
    echo "Example:"
    echo "DATA_DIR=/path/to/dataset bash scripts/train_local.sh"
    exit 1
fi

if [ ! -d "${DATA_DIR}/train" ]; then
    echo "ERROR: Missing train/ folder inside DATA_DIR"
    exit 1
fi

if [ ! -d "${DATA_DIR}/val" ]; then
    echo "ERROR: Missing val/ folder inside DATA_DIR"
    exit 1
fi

if [ ! -d "${DATA_DIR}/test" ]; then
    echo "ERROR: Missing test/ folder inside DATA_DIR"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs/local

# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------
echo "============================================================"
echo "Python environment"
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
else:
    print("Running on CPU. Training may be slow.")
PY

# ------------------------------------------------------------
# Install package locally
# ------------------------------------------------------------
echo "============================================================"
echo "Installing package in editable mode"
echo "============================================================"

pip install -e .

# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------
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

"${TRAIN_CMD[@]}" 2>&1 | tee "logs/local/${RUN_NAME}_training.log"

echo "============================================================"
echo "Training finished"
echo "============================================================"

# ------------------------------------------------------------
# Final evaluation
# ------------------------------------------------------------
BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoints/best_model.pt"

if [ -f "${BEST_CHECKPOINT}" ]; then
    echo "Best checkpoint found: ${BEST_CHECKPOINT}"
    echo "Running final test evaluation..."

    python -m raravis_dl.evaluate \
        --config "${CONFIG_FILE}" \
        --data-dir "${DATA_DIR}" \
        --checkpoint "${BEST_CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}/test_eval" \
        --split test \
        2>&1 | tee "logs/local/${RUN_NAME}_evaluation.log"
else
    echo "WARNING: Best checkpoint not found at ${BEST_CHECKPOINT}"
    echo "Skipping final evaluation."
fi

# ------------------------------------------------------------
# Output summary
# ------------------------------------------------------------
echo "============================================================"
echo "Generated outputs"
echo "============================================================"

find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort || true

echo "============================================================"
echo "Local experiment completed"
echo "============================================================"
