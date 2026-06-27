# Model Card  
## Chest X-Ray Multi-Class Classification Model

This model card summarizes the main characteristics, intended use, limitations and evaluation results of the Deep Learning models developed in this repository.

The project was developed as part of a curricular internship at the University of Jaén, within the TIC-279 SAICoG-Salud research group.

---

## 1. Model overview

This repository contains a PyTorch-based pipeline for multi-class classification of chest X-ray images.

The main confirmed model was:

```text
EfficientNet-B7 at 768×768 resolution
```

The model was trained using Transfer Learning and fine-tuning. Several architectures were tested and compared, including:

- ResNet
- DenseNet121
- MobileNetV3
- EfficientNet-B7
- EfficientNetV2
- EfficientNetV2 + Transformer / ViT-style classification head

The objective was not only to train a single model, but to build a complete and reproducible workflow for dataset preparation, training, evaluation, model comparison and interpretability.

---

## 2. Intended use

The intended use of this project is educational and research-oriented.

The repository is designed to show:

- How to organize an image classification pipeline.
- How to train CNN models with PyTorch.
- How to use Transfer Learning and fine-tuning.
- How to evaluate multiclass models with multiple metrics.
- How to handle class imbalance.
- How to generate Grad-CAM visual explanations.
- How to prepare experiments for GPU/HPC execution.

The code can also be adapted to other classification problems based on images or visual representations of signals, such as spectrogram classification in bioacoustics.

---

## 3. Not intended use

This model is not intended for direct clinical use.

It should not be used to make medical decisions, diagnose patients or replace professional medical evaluation.

The reported results correspond to academic internship experiments and should be interpreted as part of a research workflow, not as a certified medical system.

---

## 4. Dataset

The dataset used during the internship contained chest X-ray images distributed across seven diagnostic categories:

- Atelectasis
- Effusion
- Emphysema
- No finding
- Nodule
- Pneumonia
- Pneumothorax

After duplicate removal, the final audited dataset contained:

```text
4,105 unique images
```

Dataset split:

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

Class distribution:

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

The original images are not included in this repository because of privacy, size and data usage restrictions.

---

## 5. Data quality work

Before training, the dataset was audited to reduce quality issues.

The data preparation process included:

- Class folder organization.
- Image count verification.
- Duplicate detection using SHA256 hashing.
- Removal of duplicated samples.
- Train/validation/test split creation.
- Class distribution reporting.
- Split consistency checks.

This was important to reduce the risk of data leakage and to make the evaluation more reliable.

---

## 6. Training approach

The training pipeline was implemented in PyTorch and included:

- Transfer Learning.
- Fine-tuning.
- Data augmentation.
- Class-weighted loss.
- WeightedRandomSampler.
- AdamW optimizer.
- Learning-rate scheduling.
- Early stopping.
- Mixed precision training when CUDA was available.
- Checkpoint saving.
- CSV and JSON metric logging.

The training process was designed to be reproducible using YAML configuration files and fixed random seeds.

---

## 7. Evaluation metrics

The models were evaluated using several metrics:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- AUC macro OVR
- Sensitivity
- Specificity
- Confusion matrix

Accuracy alone was not considered enough because the dataset was multiclass and imbalanced.

Macro-F1 and AUC macro OVR were especially important because they provide a better view of performance across all classes.

---

## 8. Best confirmed result

The best confirmed experiment was obtained with EfficientNet-B7 at 768×768 resolution.

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The most relevant result was the AUC macro OVR of **0.9113**, because it showed strong discriminative capacity across the seven classes.

---

## 9. Model comparison

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|---:|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 |
| EfficientNetV2 + Transformer / ViT-style head | 768×768 | 0.5851 | 0.5861 | 0.8791 |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 |

EfficientNet-B7 was selected as the best confirmed model because it provided the strongest balance between Accuracy, Macro-F1 and AUC macro OVR.

---

## 10. Hybrid CNN-Transformer prototype

An experimental hybrid model was also tested.

The model combined:

- EfficientNetV2 as a CNN backbone.
- A Transformer / ViT-style classification head.
- Spatial tokens extracted from convolutional feature maps.

The objective was to explore whether self-attention could improve global image representation.

Although this model did not outperform EfficientNet-B7, it was useful as a research prototype and helped compare CNN-only architectures against a CNN-Transformer design.

---

## 11. Interpretability

The repository includes Grad-CAM and saliency-based interpretability tools.

The objective was to inspect which regions of the radiograph contributed most to the model prediction.

This is important in medical image analysis because model predictions should not be treated only as black-box outputs.

The interpretability script can generate:

- Grad-CAM heatmap.
- Overlay over the original image.
- Summary figure.
- Prediction probabilities.
- JSON output with prediction metadata.

---

## 12. Limitations

The project has several limitations:

- The dataset is limited compared to large-scale medical datasets.
- The task is multiclass and contains visually similar findings.
- The dataset is imbalanced.
- The model was not externally validated on a completely independent dataset.
- No trained checkpoints are included in the public repository.
- The model is not clinically certified.
- Grad-CAM provides qualitative support, but it is not a full explanation of model reasoning.

These limitations are important when interpreting the reported results.

---

## 13. Ethical and responsible use

This repository is intended for research, education and technical demonstration.

The model should not be used for real medical diagnosis or clinical decision-making.

Any real-world use in healthcare would require:

- External validation.
- Clinical review.
- Regulatory approval.
- Bias analysis.
- Robustness testing.
- Security and privacy controls.
- Expert supervision.

---

## 14. Reproducibility

The project includes several elements to support reproducibility:

- YAML configuration files.
- Fixed random seeds.
- Dataset split utilities.
- Duplicate audit utility.
- Training history export.
- Evaluation reports.
- Confusion matrices.
- Prediction-level CSV files.
- Checkpoint metadata.
- Slurm scripts for HPC execution.
- GitHub Actions tests for metric utilities.

The objective was to make the workflow understandable, repeatable and extendable.

---

## 15. Possible future improvements

Possible future improvements include:

- External validation using another dataset.
- Medical pretrained weights.
- More advanced Vision Transformer architectures.
- Better probability calibration.
- MLflow or Weights & Biases tracking.
- More detailed error analysis.
- More advanced explainability methods.
- Deployment as an inference API.
- Adaptation to audio spectrogram classification for bioacoustic monitoring.

---

## 16. Final summary

This project represents a complete Deep Learning workflow for medical image classification.

The strongest confirmed model was EfficientNet-B7 at 768×768 resolution, achieving:

```text
Accuracy:       0.6742
Macro-F1:       0.6681
AUC macro OVR:  0.9113
```

The main value of the project is not only the final metric, but the complete technical pipeline: dataset auditing, model training, evaluation, experiment comparison, Grad-CAM interpretability and GPU/HPC execution.
