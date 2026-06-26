# Chest X-Ray Multi-Class Classification with PyTorch

Production-oriented Deep Learning pipeline for multi-class chest X-ray classification using PyTorch, Transfer Learning, Grad-CAM interpretability and GPU/HPC execution.

This repository summarizes the technical work I developed during my curricular internship at the University of Jaén, in the context of medical image classification. The project was not only about training a model, but about building a complete and reproducible workflow: dataset preparation, duplicate detection, train/validation/test management, model training, model comparison, metric export, confusion matrices, visual interpretability and HPC execution.

Although the original use case was chest X-ray classification, the same workflow can be adapted to other image classification problems or to visual representations of signals, such as spectrograms in bioacoustic classification.

---

## Data note

The original medical images are not included in this repository because of privacy, size and usage restrictions.

This repository is intended to show the technical structure of the pipeline, the training workflow, the evaluation process and the experiment documentation. The code is dataset-agnostic and can work with any ImageFolder-style dataset organized with `train/`, `val/` and `test/` folders.

---

## Highlights

- End-to-end PyTorch pipeline for 7-class chest X-ray classification.
- Support for ResNet, DenseNet121, MobileNetV3, EfficientNet-B7 and EfficientNetV2-based models.
- Experimental hybrid prototype combining an EfficientNet/EfficientNetV2 CNN backbone with a Transformer / ViT-style classification module.
- Reproducible experiments using YAML configuration files, fixed seeds and CSV/JSON logging.
- Dataset quality utilities: duplicate detection, class distribution reporting and split validation.
- Training strategies: Transfer Learning, fine-tuning, data augmentation, class weighting, WeightedRandomSampler, early stopping and learning-rate scheduling.
- GPU/HPC ready: CUDA support, Slurm `sbatch` script and offline pretrained weights workflow.
- Evaluation with Accuracy, Precision, Recall, F1-score, Macro-F1, AUC macro OVR, sensitivity, specificity and confusion matrices.
- Interpretability with Grad-CAM and saliency maps to inspect the regions used by the model during prediction.
- Public summary results and documentation without exposing the original private dataset.

---

## Project context

This project was developed during my curricular internship at the University of Jaén, within the TIC-279 SAICoG-Salud research group.

The main task was to develop and evaluate Deep Learning models for automatic classification of chest radiographs. The work covered the full technical process:

1. Preparing and auditing the dataset.
2. Detecting and removing duplicated images.
3. Creating train, validation and test splits.
4. Implementing training pipelines in PyTorch.
5. Comparing different CNN architectures.
6. Testing a hybrid CNN-Transformer / ViT-style approach.
7. Evaluating the models with multiple metrics.
8. Generating confusion matrices and experiment summaries.
9. Adding Grad-CAM visual explanations.
10. Running experiments in Linux/GPU/HPC environments.

The objective was to build a workflow that could be repeated and extended, instead of relying on isolated scripts or manual experiments.

---

## Dataset used during the internship

The dataset used during the internship contained chest X-ray images distributed across seven diagnostic categories:

- Atelectasis
- Effusion
- Emphysema
- No finding
- Nodule
- Pneumonia
- Pneumothorax

Before training, I performed a dataset audit to detect duplicated images using hash-based comparison. This step was important because duplicated medical images can affect the validity of the evaluation if the same or very similar images appear in different splits.

After removing duplicates, the final audited dataset contained **4,105 unique chest X-ray images**.

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

The dataset was split into training, validation and test sets:

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

The approximate split ratio was 70% training, 15% validation and 15% test.

---

## Repository structure

```text
.
├── configs/                         # YAML experiment configurations
│   ├── efficientnet_b7_768.yaml
│   ├── densenet121_768.yaml
│   └── mobilenetv3_512.yaml
│
├── docs/                            # Technical notes, experiment report and assets
│   ├── TECHNICAL_REPORT.md
│   ├── EXPERIMENTS.md
│   ├── INTERVIEW_NOTES.md
│   └── assets/
│       ├── confusion_matrix_best.png
│       ├── model_accuracy_comparison.png
│       └── model_macro_f1_comparison.png
│
├── results/                         # Public summary results and example outputs
│   ├── metrics_summary.csv
│   ├── confusion_matrix_best.csv
│   ├── dataset_audit_summary.csv
│   └── split_summary.csv
│
├── scripts/                         # Local and Slurm/HPC execution scripts
│   ├── train_local.sh
│   └── train_ada_slurm.sbatch
│
├── src/
│   └── raravis_dl/                  # Main Python package
│       ├── data.py                  # Datasets, transforms, class weights, samplers
│       ├── models.py                # Model factory and classifier-head replacement
│       ├── train.py                 # Training entry point
│       ├── evaluate.py              # Evaluation entry point
│       ├── engine.py                # Training and validation loops
│       ├── metrics.py               # Metrics, AUC, specificity, confusion matrix
│       ├── interpretability.py      # Grad-CAM and saliency helpers
│       ├── duplicate_audit.py       # Hash-based duplicate detection
│       ├── split_dataset.py         # Reproducible dataset splitting utility
│       ├── infer.py                 # Single-image inference
│       └── utils.py                 # Seeds, logging, checkpoints, config loading
│
├── tests/                           # Lightweight tests for metric utilities
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Expected dataset format

The code expects an ImageFolder-style dataset with one folder per class.

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
│
├── val/
│   ├── Atelectasis/
│   ├── Effusion/
│   └── ...
│
└── test/
    ├── Atelectasis/
    ├── Effusion/
    └── ...
```

This structure makes the project easy to adapt to other image classification datasets.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-user>/deep-learning-medical-image-classification-pytorch.git
cd deep-learning-medical-image-classification-pytorch
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

---

## Train a model

Example training command using EfficientNet-B7 at 768×768 resolution:

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

During training, the pipeline saves:

- Training and validation metrics.
- Best model checkpoint.
- Last model checkpoint.
- Experiment configuration.
- Logs and output files.
- CSV/JSON summaries for later analysis.

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

The evaluation includes metrics such as Accuracy, Precision, Recall, F1-score, Macro-F1, AUC macro OVR, sensitivity, specificity and confusion matrix.

---

## Run inference on a single image

```bash
python -m raravis_dl.infer \
  --config configs/efficientnet_b7_768.yaml \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --image-path /path/to/image.png
```

This command loads a trained checkpoint and returns the predicted class and class probabilities for a single image.

---

## Generate Grad-CAM visual explanations

The repository includes Grad-CAM and saliency-based interpretability utilities.

The goal is to inspect which regions of the image contributed the most to the model prediction. This is especially relevant in medical image analysis, where a model should not only return a prediction but also allow a qualitative check of its behaviour.

Example command:

```bash
python -m raravis_dl.interpretability \
  --config configs/efficientnet_b7_768.yaml \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --image-path /path/to/image.png \
  --output-dir runs/efficientnet_b7_768/gradcam_examples
```

The script generates heatmap-based outputs that can be used to inspect whether the model is focusing on meaningful image regions or on irrelevant artifacts.

---

## HPC / Slurm execution

Some experiments were designed to run in a Linux/GPU/HPC environment.

Example Slurm command:

```bash
sbatch scripts/train_ada_slurm.sbatch
```

The Slurm script is prepared for GPU nodes and stores logs, checkpoints and metrics in a structured experiment folder.

A relevant part of the work was dealing with practical HPC issues such as:

- CUDA/GPU execution.
- Long training jobs.
- Checkpoint management.
- Offline pretrained weights.
- Experiment logging.
- Recovering outputs after cluster executions.
- Keeping experiments reproducible across different runs.

---

## Data quality utilities

### Duplicate audit

```bash
python -m raravis_dl.duplicate_audit \
  --data-dir /path/to/raw_dataset \
  --output results/duplicate_report.csv
```

This utility computes hashes for the images and reports duplicated files. It was used to clean the dataset before creating the final split.

### Create train/validation/test splits

```bash
python -m raravis_dl.split_dataset \
  --input-dir /path/to/raw_dataset \
  --output-dir /path/to/split_dataset \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

The split utility helps create a reproducible dataset structure for training and evaluation.

---

## Training strategies used

Several training strategies were used during the experiments:

- Transfer Learning from pretrained models.
- Fine-tuning of CNN backbones.
- Data augmentation.
- Class weighting.
- WeightedRandomSampler.
- Early stopping.
- Learning-rate scheduling.
- Weight decay regularization.
- Mixed precision training when supported.
- High-resolution image inputs.
- Checkpoint-based model selection.

These strategies were used to improve stability, reduce overfitting and manage class imbalance.

---

## Models tested

The following architectures were tested or prepared during the experimental process:

- ResNet
- DenseNet121
- MobileNetV3
- EfficientNet-B7
- EfficientNetV2
- EfficientNetV2 + Transformer / ViT-style classification head

The hybrid EfficientNetV2 + Transformer / ViT-style prototype was included to explore whether adding an attention-based component could improve global feature representation.

Although the hybrid model did not outperform the best EfficientNet-B7 experiment, it was useful to compare a pure CNN approach against a CNN-Transformer design and to better understand the trade-off between complexity, stability and performance.

---

## Best confirmed experiment

The best confirmed test-set experiment was obtained with EfficientNet-B7 at 768×768 resolution.

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The result corresponds to a multi-class setup with 7 diagnostic categories, which is considerably harder than a binary classification task.

The most relevant result was the AUC macro OVR of 0.91, because it showed strong discriminative capacity across the different classes, even in a difficult multiclass medical image classification problem.

---

## Experiment summary

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|---:|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 |
| EfficientNetV2 + Transformer / ViT-style head | 768×768 | 0.5851 | 0.5861 | 0.8791 |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 |

These results were obtained using the same evaluation logic and were used to compare the behaviour of different architectures under the same multiclass classification task.

---

## Technical decisions

- Macro-F1 was used because the dataset was imbalanced across classes.
- AUC macro OVR was included to measure discriminative capacity across the seven diagnostic categories.
- Accuracy was reported, but it was not treated as the only metric because of the multiclass and imbalanced nature of the problem.
- WeightedRandomSampler and class weighting were considered to reduce the impact of class imbalance during training.
- High-resolution inputs improved feature representation but increased GPU memory usage, requiring smaller batch sizes.
- EfficientNet-B7 at 768×768 provided the best balance between performance and stability in the confirmed experiments.
- The EfficientNetV2 + Transformer / ViT-style prototype was tested as an experimental architecture to explore global attention mechanisms.
- HPC execution required a robust offline workflow for pretrained weights, checkpoint management and reproducible experiment outputs.
- Grad-CAM was added to support visual interpretability and qualitative analysis of model predictions.

---

## Practical issues solved during development

During the internship, part of the work was solving practical problems that appear in real Deep Learning projects:

- Preparing and cleaning a medical image dataset.
- Detecting duplicated images to avoid data leakage.
- Managing class imbalance in a multiclass dataset.
- Running long training jobs in a GPU/HPC environment.
- Handling pretrained weights in a cluster environment without direct internet access.
- Organizing experiment outputs to avoid losing metrics after long executions.
- Exporting predictions, confusion matrices and metrics for later analysis.
- Comparing models using the same split and evaluation protocol.
- Adding Grad-CAM outputs to make the predictions more interpretable.
- Documenting the workflow so that the experiments could be understood and repeated.

These parts were important because the project was not developed as a simple notebook, but as a more structured research-oriented pipeline.

---

## Why this project matters

For me, the most valuable part of this work was learning how to move from a basic model training script to a complete experimental pipeline.

In a real Deep Learning project, the model is only one part of the work. Dataset quality, preprocessing, class imbalance, reproducibility, evaluation metrics, computational constraints and interpretation of the results are just as important.

This project helped me improve in:

- Computer Vision.
- Medical image classification.
- PyTorch training pipelines.
- Transfer Learning and fine-tuning.
- GPU training.
- HPC/Slurm execution.
- Metric analysis.
- Grad-CAM interpretability.
- Technical documentation.
- Reproducible experimentation.

---

## Limitations

This repository does not include the original dataset or trained checkpoints.

The reasons are:

- The dataset is large.
- The images belong to a research/academic workflow.
- The repository is intended to show the technical structure of the pipeline.
- The code can be adapted to any compatible dataset with the same folder structure.

The reported results correspond to the experiments carried out during the internship.

---

## Possible future improvements

Some possible improvements for future work are:

- Add external validation with a different dataset.
- Test Vision Transformers and more recent architectures.
- Use medical pretrained weights.
- Add experiment tracking with MLflow or Weights & Biases.
- Improve calibration of predicted probabilities.
- Add more advanced explainability methods.
- Deploy the best model as an API for inference.
- Adapt the pipeline to audio spectrogram classification for bioacoustic analysis.

---

## Author

**Anass Azouagh**  
Computer Engineer specialized in Artificial Intelligence, Computer Vision, Backend Development and Big Data.

LinkedIn: https://www.linkedin.com/in/anass-azouagh  
Email: anassazouagh1@gmail.com
