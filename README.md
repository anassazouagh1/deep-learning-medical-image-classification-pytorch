# Chest X-Ray Multi-Class Classification with PyTorch

Production-oriented **Deep Learning pipeline for multi-class chest X-ray classification** using PyTorch, Transfer Learning and GPU/HPC execution.

This repository summarizes a real research internship workflow developed at the **University of Jaén** in the context of medical image classification. The focus is not only training a model, but building a reproducible technical pipeline: dataset auditing, duplicate detection, train/validation/test management, model benchmarking, metric export, confusion matrices, interpretability and HPC execution with Slurm.

> **Data note**  
> The original medical images are not included because of privacy, size and usage restrictions. The code is dataset-agnostic and works with any `ImageFolder`-style dataset with `train/`, `val/` and `test/` folders.

---

## Highlights

- End-to-end PyTorch pipeline for **7-class chest X-ray classification**.
- Support for **ResNet, DenseNet121, MobileNetV3, EfficientNet-B7** and a CNN-Transformer prototype.
- Reproducible experiments with YAML configuration, fixed seeds and CSV logging.
- Data quality utilities: duplicate detection, class distribution reporting and split validation.
- Training strategies: Transfer Learning, fine-tuning, class weighting, WeightedRandomSampler, early stopping and learning-rate scheduling.
- GPU/HPC ready: Slurm `sbatch` script, CUDA support and offline pretrained weights workflow.
- Evaluation with Accuracy, Precision, Recall, F1, Macro-F1, AUC macro OVR, sensitivity, specificity and confusion matrices.
- Interpretability with Grad-CAM and saliency maps.

---

## Best confirmed experiment

Best confirmed test-set experiment using **EfficientNet-B7 at 768×768 resolution**:

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The result corresponds to a **multi-class setup with 7 diagnostic categories**, which is considerably harder than a binary classification task.

---

## Dataset used during the internship

The final audited dataset contained **4,105 unique chest X-ray images** after removing 6 duplicates.

| Class | Unique images |
|---|---:|
| Atelectasis | 682 |
| Effusion | 533 |
| Emphysema | 332 |
| No finding | 1,060 |
| Nodule | 656 |
| Pneumonia | 497 |
| Pneumothorax | 345 |
| **Total** | **4,105** |

Split used in the experiments:

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

---

## Repository structure

```text
.
├── configs/                         # YAML experiment configurations
├── docs/                            # Technical notes, experiment report and assets
├── results/                         # Public summary results and example outputs
├── scripts/                         # Local and Slurm/HPC execution scripts
├── src/raravis_dl/                  # Main Python package
│   ├── data.py                      # Datasets, transforms, class weights, samplers
│   ├── models.py                    # Model factory and classifier-head replacement
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation entry point
│   ├── engine.py                    # Training/validation loops
│   ├── metrics.py                   # Metrics, AUC, specificity, confusion matrix
│   ├── interpretability.py          # Grad-CAM / saliency helpers
│   ├── duplicate_audit.py           # Hash-based duplicate detection
│   ├── split_dataset.py             # Reproducible dataset splitting utility
│   └── utils.py                     # Seeds, logging, checkpoints, config loading
├── tests/                           # Lightweight tests for metric utilities
├── requirements.txt
└── README.md
```

---

## Expected dataset format

```text
dataset/
├── train/
│   ├── Atelectasis/
│   ├── Effusion/
│   ├── Emphysema/
│   ├── No finding/
│   ├── Nodule/
│   ├── Pneumonia/
│   └── Pneumothorax/
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Installation

```bash
git clone https://github.com/<your-user>/deep-learning-medical-image-classification-pytorch.git
cd deep-learning-medical-image-classification-pytorch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train a model

```bash
python -m raravis_dl.train \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir /path/to/dataset \
  --output-dir runs/efficientnet_b7_768
```

Useful options:

```bash
python -m raravis_dl.train --help
```

---

## Evaluate a checkpoint

```bash
python -m raravis_dl.evaluate \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir /path/to/dataset \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --output-dir runs/efficientnet_b7_768/test_eval
```

The evaluation script exports:

- `metrics_test.json`
- `predictions_test.csv`
- `confusion_matrix_test.csv`

---

## HPC / Slurm execution

```bash
sbatch scripts/train_ada_slurm.sbatch
```

The Slurm script is designed for GPU nodes and stores logs, checkpoints and metrics in a structured experiment folder.

---

## Data quality utilities

Duplicate audit:

```bash
python -m raravis_dl.duplicate_audit \
  --data-dir /path/to/raw_dataset \
  --output results/duplicate_report.csv
```

Create stratified train/validation/test splits:

```bash
python -m raravis_dl.split_dataset \
  --input-dir /path/to/raw_dataset \
  --output-dir /path/to/split_dataset \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

---

## Interpretability

```bash
python -m raravis_dl.interpretability \
  --config configs/efficientnet_b7_768.yaml \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --image-path /path/to/image.png \
  --output-dir runs/efficientnet_b7_768/gradcam_examples
```

---

## Experiment summary

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|---:|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 |
| EffNetV2 + Transformer | 768×768 | 0.5851 | 0.5861 | 0.8791 |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 |

---

## Technical decisions

- **Macro-F1** was used for model selection because the dataset was imbalanced across classes.
- **AUC macro OVR** was included to measure discriminative capacity across the seven classes.
- **WeightedRandomSampler and class weighting** were considered to reduce the impact of class imbalance.
- **High-resolution inputs** improved discriminative capacity but increased GPU memory usage, requiring smaller batch sizes.
- **HPC execution** required a robust offline workflow for pretrained weights and reproducible experiment outputs.

---

## Author

**Anass Azouagh**  
Computer Engineer specialized in Artificial Intelligence, Computer Vision, Backend Development and Big Data.

LinkedIn: https://www.linkedin.com/in/anass-azouagh
