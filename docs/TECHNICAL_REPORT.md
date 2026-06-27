# Technical Report  
## Chest X-Ray Multi-Class Classification with Deep Learning

This document summarizes the technical workflow developed during my curricular internship at the University of Jaén, within the TIC-279 SAICoG-Salud research group.

The project focused on multi-class classification of chest X-ray images using Deep Learning and PyTorch. The main goal was not only to train a neural network, but to build a complete experimental pipeline that could be repeated, evaluated and extended.

The work included dataset preparation, duplicate detection, train/validation/test split management, model training, benchmarking, metric analysis, Grad-CAM interpretability and GPU/HPC execution.

---

## 1. Project objective

The objective was to develop and evaluate Deep Learning models capable of classifying chest radiographs into seven diagnostic categories:

- Atelectasis
- Effusion
- Emphysema
- No finding
- Nodule
- Pneumonia
- Pneumothorax

The task was treated as a **multi-class image classification problem**. This made the project more challenging than a binary classification task because the model had to distinguish between several thoracic findings, some of which can be visually similar.

---

## 2. Dataset preparation

The dataset contained chest X-ray images organized by diagnostic class.

A first technical task was to audit the dataset before training. This step was important because medical datasets can contain duplicated images, inconsistent folders or unbalanced classes.

The final audited dataset contained **4,105 unique images** after removing **6 duplicated samples**.

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

The data was divided into:

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

The approximate split ratio was:

- 70% training
- 15% validation
- 15% test

---

## 3. Duplicate detection

Duplicate detection was performed using a hash-based strategy.

The reason for doing this was to reduce the risk of **data leakage**. If the same image appears in both training and test sets, the final evaluation can become unrealistic because the model may have already seen that sample during training.

The duplicate audit was included as a separate utility in the repository:

```bash
python -m raravis_dl.duplicate_audit \
  --data-dir /path/to/raw_dataset \
  --output results/duplicate_report.csv
```

This made the dataset cleaning process reproducible and easier to document.

---

## 4. Data loading and preprocessing

The pipeline uses an ImageFolder-style dataset structure:

```text
dataset/
├── train/
├── val/
└── test/
```

Each split contains one folder per class.

The preprocessing pipeline includes:

- Image resizing.
- Tensor conversion.
- ImageNet normalization.
- Conservative data augmentation during training.
- Separate transforms for training and evaluation.

The augmentation was intentionally moderate because the dataset contains medical images. Strong transformations can alter clinically relevant visual patterns.

Training transforms included:

- Small rotations.
- Slight translations.
- Small scale variation.
- Brightness and contrast adjustment.

Horizontal flipping was avoided by default because laterality can be meaningful in medical images.

---

## 5. Class imbalance handling

The dataset was imbalanced, especially because the number of samples was not equal across the seven classes.

To reduce the impact of this imbalance, the training pipeline supports:

- Class-weighted CrossEntropyLoss.
- WeightedRandomSampler.
- Macro-based metrics for model selection.

This was important because accuracy alone can be misleading in imbalanced multi-class problems. A model could perform well on majority classes while failing on minority classes.

---

## 6. Models tested

Several architectures were tested during the experimental workflow:

| Model | Type |
|---|---|
| ResNet | CNN |
| DenseNet121 | CNN |
| MobileNetV3 | Lightweight CNN |
| EfficientNet-B7 | CNN |
| EfficientNetV2 | CNN |
| EfficientNetV2 + Transformer / ViT-style head | Hybrid CNN-Transformer |

The goal was to compare different families of models and understand the trade-off between performance, complexity and computational cost.

---

## 7. EfficientNet-B7 experiment

The best confirmed experiment was obtained with **EfficientNet-B7** using **768×768** input resolution.

This model gave the best balance between:

- Accuracy.
- Macro-F1.
- AUC macro OVR.
- Stability.
- Generalization.
- Computational cost.

Best confirmed test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The AUC macro OVR of **0.91** was especially relevant because it showed strong discriminative capacity across the seven diagnostic categories.

---

## 8. Hybrid EfficientNetV2 + Transformer / ViT-style prototype

In addition to pure CNN architectures, an experimental hybrid model was tested.

The idea was to combine:

- An **EfficientNetV2 CNN backbone** for local visual feature extraction.
- A **Transformer / ViT-style classification head** to model global spatial relations between image regions.

The motivation was that some medical findings may depend on local patterns, while others may benefit from a more global representation of the radiograph.

Confirmed test results:

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|---:|
| EfficientNetV2 + Transformer / ViT-style head | 768×768 | 0.5851 | 0.5861 | 0.8791 |

Although this hybrid model did not outperform EfficientNet-B7, it was useful as a research prototype. It helped compare a pure CNN approach against a CNN-Transformer design and showed the trade-off between architectural complexity and practical performance.

---

## 9. Training pipeline

The training pipeline was implemented in PyTorch and designed to be reusable across different experiments.

The pipeline includes:

- YAML-based configuration.
- Fixed seed for reproducibility.
- Dataset validation.
- Train/validation/test loading.
- Model factory.
- Transfer Learning.
- Fine-tuning.
- Class weighting.
- Weighted sampling.
- Mixed precision training.
- Gradient clipping.
- Learning-rate scheduling.
- Early stopping.
- Best checkpoint saving.
- Last checkpoint saving.
- CSV and JSON logging.

Example command:

```bash
python -m raravis_dl.train \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir /path/to/dataset \
  --output-dir runs/efficientnet_b7_768
```

The goal was to avoid isolated manual scripts and create a more structured workflow that could be repeated with different models and settings.

---

## 10. Evaluation pipeline

The evaluation pipeline exports both global and detailed results.

It produces:

- Global metrics.
- Per-class metrics.
- Confusion matrix.
- Prediction-level CSV.
- Error analysis.
- JSON summary.

Example command:

```bash
python -m raravis_dl.evaluate \
  --config configs/efficientnet_b7_768.yaml \
  --data-dir /path/to/dataset \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --output-dir runs/efficientnet_b7_768/test_eval
```

The evaluation was based on several metrics:

- Accuracy
- Balanced accuracy
- Precision macro
- Recall macro
- F1 macro
- Precision weighted
- Recall weighted
- F1 weighted
- AUC macro OVR
- Sensitivity
- Specificity
- Confusion matrix

This allowed a more complete analysis of model behaviour.

---

## 11. Experiment comparison

Summary of confirmed experiments:

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|---:|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 |
| EfficientNetV2 + Transformer / ViT-style head | 768×768 | 0.5851 | 0.5861 | 0.8791 |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 |

EfficientNet-B7 was selected as the best confirmed model because it obtained the highest combination of Accuracy, Macro-F1 and AUC macro OVR.

---

## 12. Confusion matrix analysis

The confusion matrix was used to analyze which classes were easier or harder to classify.

This was important because global metrics do not show all the details. In medical image classification, two models can have similar accuracy but very different behaviour depending on the class.

The confusion matrix helped identify:

- Classes with better recognition.
- Classes frequently confused with others.
- Possible weaknesses of the model.
- Impact of class imbalance.
- Cases where visual similarity between findings affected performance.

The evaluation script exports the matrix as CSV to support later analysis.

---

## 13. Interpretability with Grad-CAM

Grad-CAM was included to make the model predictions more interpretable.

In medical image classification, interpretability is important because the model should not be treated as a black box. Even when the prediction is correct, it is useful to inspect whether the network is focusing on meaningful regions of the radiograph.

The repository includes a Grad-CAM script that:

- Loads a trained checkpoint.
- Loads and preprocesses an input image.
- Selects the target convolutional layer.
- Generates the Grad-CAM heatmap.
- Creates an overlay over the original image.
- Saves a summary figure.
- Exports prediction probabilities in JSON.

Example command:

```bash
python -m raravis_dl.interpretability \
  --config configs/efficientnet_b7_768.yaml \
  --checkpoint runs/efficientnet_b7_768/checkpoints/best_model.pt \
  --image-path /path/to/image.png \
  --output-dir runs/efficientnet_b7_768/gradcam_examples
```

The output can be used to visually inspect which areas contributed most to the model prediction.

---

## 14. GPU and HPC execution

Some experiments were prepared for execution in a Linux/GPU/HPC environment.

The repository includes a Slurm script:

```bash
sbatch scripts/train_ada_slurm.sbatch
```

The HPC workflow included:

- CUDA execution.
- Slurm job submission.
- Long training jobs.
- Checkpoint management.
- Offline pretrained weight handling.
- Logging of outputs.
- Recovery of metrics after executions.
- Structured experiment folders.

A practical difficulty was working in an environment where pretrained weights could not always be downloaded directly from the internet. For this reason, the pipeline includes support for loading local pretrained weights manually.

---

## 15. Main technical problems solved

During the project, several practical technical issues appeared:

### Dataset issues

- Need to organize images correctly by class.
- Need to remove duplicated samples.
- Need to keep train, validation and test splits consistent.
- Need to manage class imbalance.

### Training issues

- GPU memory limitations with high-resolution images.
- Smaller batch sizes required for 768×768 inputs.
- Need for early stopping to reduce overfitting.
- Need for learning-rate scheduling.
- Checkpoint compatibility between experiments.

### HPC issues

- Cluster execution with Slurm.
- Pretrained weights in environments without direct internet access.
- Long-running jobs requiring structured logging.
- Need to avoid losing results after training.

### Evaluation issues

- Accuracy alone was not enough.
- Need to export predictions and confusion matrices.
- Need to compare models using the same evaluation logic.
- Need to analyze per-class behaviour.

### Interpretability issues

- Need to inspect what the model was focusing on.
- Need to generate Grad-CAM overlays.
- Need to save visual and JSON outputs for later analysis.

---

## 16. What I learned

This project helped me improve technically in several areas:

- PyTorch training pipelines.
- Computer Vision.
- Medical image classification.
- Transfer Learning.
- Fine-tuning.
- Dataset auditing.
- Class imbalance handling.
- GPU training.
- HPC/Slurm execution.
- Metric analysis.
- Grad-CAM interpretability.
- Experiment documentation.
- Reproducible research workflows.

The most important lesson was that a Deep Learning project is not only about the model. The dataset, preprocessing, metrics, reproducibility, infrastructure and interpretation of results are just as important.

---

## 17. Limitations

This repository does not include the original dataset or trained checkpoints.

The reasons are:

- The dataset is large.
- The images belong to an academic/research workflow.
- The objective of the repository is to show the technical pipeline.
- The code can be adapted to another compatible dataset.

The reported results correspond to the confirmed experiments carried out during the internship.

---

## 18. Possible future work

Possible improvements include:

- External validation with another dataset.
- More advanced Vision Transformer architectures.
- Medical pretrained weights.
- Better probability calibration.
- MLflow or Weights & Biases experiment tracking.
- More advanced interpretability methods.
- Deployment as an inference API.
- Adaptation of the pipeline to audio spectrogram classification for bioacoustic monitoring.

---

## 19. Connection with other domains

Although this project was developed for medical image classification, the same workflow can be adapted to other classification problems.

For example, in bioacoustics, audio recordings can be transformed into spectrograms and processed as images. In that case, many parts of this repository remain useful:

- Dataset organization.
- CNN and Transformer-based classification.
- Transfer Learning.
- Train/validation/test split.
- Metric export.
- Confusion matrix analysis.
- Reproducible experiments.
- HPC execution.

This makes the project technically relevant beyond medical imaging.

---

## 20. Conclusion

The project provided a complete practical experience in Deep Learning applied to image classification.

The final result was not only a trained model, but a structured pipeline covering the full workflow: data preparation, training, evaluation, model comparison, interpretability and experiment documentation.

The best confirmed model was EfficientNet-B7 at 768×768 resolution, with an AUC macro OVR of 0.9113, Macro-F1 of 0.6681 and Accuracy of 0.6742.

The project helped me consolidate my skills in Deep Learning, PyTorch, Computer Vision, GPU/HPC execution and applied AI experimentation.
