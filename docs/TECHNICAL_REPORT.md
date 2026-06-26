# Technical Report

## 1. Context

This project implements a Deep Learning workflow for multi-class classification of chest X-ray images. The work was developed in an applied research context and focused on building a reproducible experimental pipeline rather than a single isolated model.

The original internship objective was to study Deep Learning techniques for medical image classification, compare CNN architectures and document the results with objective metrics.

## 2. Engineering contribution

The repository is organized to reflect the full technical process:

1. Dataset audit and duplicate detection.
2. Clean train/validation/test split.
3. Config-driven model training.
4. Experiment tracking through CSV and JSON files.
5. Model evaluation with clinical and ML metrics.
6. Confusion matrix and prediction export.
7. Interpretability with Grad-CAM and saliency maps.
8. Slurm/HPC execution for GPU training.

This structure was designed to make the project easier to reproduce, debug and extend.

## 3. Dataset audit

The dataset was audited using file-level duplicate detection. The final cleaned dataset contained 4,105 unique images from an initial 4,111 files. Six duplicates were identified in the `No finding` class and removed before model training.

## 4. Training strategy

The training loop includes:

- deterministic seed control,
- transfer learning,
- fine-tuning of pretrained models,
- AdamW optimizer,
- ReduceLROnPlateau scheduling,
- early stopping,
- optional class weighting,
- optional WeightedRandomSampler,
- mixed precision support,
- model checkpointing based on validation Macro-F1.

Macro-F1 was selected as the main validation metric because the dataset was imbalanced and a simple accuracy score could hide weak performance on minority classes.

## 5. Evaluation strategy

The test evaluation produces:

- global metrics,
- macro/weighted metrics,
- per-class metrics,
- AUC macro OVR,
- sensitivity,
- specificity,
- confusion matrix,
- per-image predictions.

The final experiment with EfficientNet-B7 at 768×768 reached:

- Accuracy: 0.6742
- Macro-F1: 0.6681
- AUC macro OVR: 0.9113
- Macro specificity: 0.9445

## 6. HPC constraints solved

During the work, several practical constraints had to be solved:

- Running long trainings on GPU nodes through Slurm.
- Organizing logs and checkpoints per experiment.
- Loading pretrained weights in environments without direct internet access.
- Managing GPU memory when increasing the input resolution to 768×768.
- Exporting metrics in a format suitable for academic reporting.

## 7. Possible extensions

Future improvements could include:

- medical-domain pretrained weights,
- Vision Transformers or EVA-X style backbones,
- cross-validation,
- external validation datasets,
- threshold calibration,
- model serving through a REST API,
- dashboard for monitoring predictions and metrics.
