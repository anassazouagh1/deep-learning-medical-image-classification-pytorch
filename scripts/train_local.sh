#!/usr/bin/env bash
set -euo pipefail

python -m raravis_dl.train \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir "${DATA_DIR:-/path/to/dataset}" \
  --output-dir runs/efficientnet_b7_768
