# Project Walkthrough Notes

These notes summarize how I would explain this project in a technical interview.

The goal is to describe the work clearly: what problem I worked on, what pipeline I built, what models I tested, how I evaluated them, what problems appeared and what I learned from the process.

---

## 1. Short project explanation

This project was developed during my curricular internship at the University of Jaén, within the TIC-279 SAICoG-Salud research group.

The objective was to build a Deep Learning pipeline for multi-class classification of chest X-ray images.

The task consisted of classifying radiographs into seven categories:

- Atelectasis
- Effusion
- Emphysema
- No finding
- Nodule
- Pneumonia
- Pneumothorax

The project was not only about training a model. I worked on the full technical workflow: dataset preparation, duplicate detection, train/validation/test splitting, model training, evaluation, experiment comparison, confusion matrices, Grad-CAM interpretability and GPU/HPC execution.

---

## 2. How I would explain the pipeline

The pipeline follows these steps:

1. Prepare the dataset in an ImageFolder-style structure.
2. Audit the dataset and detect duplicated images.
3. Remove duplicates to reduce the risk of data leakage.
4. Create train, validation and test splits.
5. Apply preprocessing and conservative data augmentation.
6. Train different CNN models using Transfer Learning and fine-tuning.
7. Handle class imbalance using class weights and WeightedRandomSampler.
8. Evaluate models with multiple metrics, not only accuracy.
9. Compare architectures using the same evaluation logic.
10. Generate Grad-CAM and saliency maps for visual interpretability.
11. Export metrics, predictions and confusion matrices for later analysis.

---

## 3. Why this task was challenging

This was a difficult problem for several reasons:

- It was a multi-class classification problem, not binary.
- Some thoracic findings can look visually similar.
- The dataset was imbalanced.
- Medical images require careful preprocessing and evaluation.
- Accuracy alone was not enough to understand model behaviour.
- High-resolution images increased GPU memory usage.
- Some experiments had to run in a Linux/HPC environment.

For that reason, I focused on building a reproducible and structured workflow instead of only training one isolated model.

---

## 4. Dataset summary

The final audited dataset contained **4,105 unique chest X-ray images** after removing duplicated samples.

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

The dataset contained seven diagnostic categories.

| Class | Unique images |
|---|---:|
| Atelectasis | 682 |
| Effusion | 533 |
| Emphysema | 332 |
| No finding | 1,060 |
| Nodule | 656 |
| Pneumonia | 497 |
| Pneumothorax | 345 |

---

## 5. Models tested

During the project, I tested and compared several architectures:

| Model | Type |
|---|---|
| ResNet | CNN baseline |
| DenseNet121 | CNN baseline |
| MobileNetV3 | Lightweight CNN |
| EfficientNet-B7 | High-capacity CNN |
| EfficientNetV2 | Modern CNN backbone |
| EfficientNetV2 + Transformer / ViT-style head | Hybrid CNN-Transformer prototype |

The goal was to compare different model families and understand the trade-off between performance, complexity and computational cost.

---

## 6. Best confirmed result

The best confirmed experiment was obtained with **EfficientNet-B7** using **768×768** image resolution.

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The most important result was the **AUC macro OVR of 0.9113**, because it showed strong discriminative capacity across the seven classes.

The accuracy was 0.6742, which is a realistic result for a multiclass medical image classification task with class imbalance and visually similar categories.

---

## 7. Why I used more metrics than accuracy

Accuracy was not enough because the dataset was imbalanced.

If a model performs well on majority classes but poorly on minority classes, the accuracy can still look acceptable. For that reason, I also used:

- Macro-F1
- Weighted-F1
- AUC macro OVR
- Sensitivity
- Specificity
- Confusion matrix
- Per-class reports

Macro-F1 was especially useful because it gives equal importance to all classes.

AUC macro OVR was useful because it evaluates the discriminative capacity of the model using predicted probabilities.

---

## 8. Hybrid EfficientNetV2 + Transformer model

I also tested an experimental hybrid model combining EfficientNetV2 and a Transformer / ViT-style classification head.

The idea was:

- EfficientNetV2 extracts local visual features from the radiograph.
- The Transformer head processes spatial tokens.
- The model tries to capture more global relationships between image regions.

The confirmed result was:

| Model | Accuracy | Macro-F1 | AUC macro OVR |
|---|---:|---:|---:|
| EfficientNetV2 + Transformer / ViT-style head | 0.5851 | 0.5861 | 0.8791 |

This model did not outperform EfficientNet-B7, but it was useful from a research point of view because it allowed comparison between a pure CNN model and a CNN-Transformer approach.

The result showed that adding attention mechanisms increased complexity and computational cost, but did not improve performance in this specific setup.

---

## 9. Grad-CAM explanation

Grad-CAM was added to inspect what parts of the radiograph influenced the model prediction.

This is important because in medical image classification, the model should not be treated as a black box.

The Grad-CAM script:

- Loads a trained checkpoint.
- Loads and preprocesses an image.
- Selects the target convolutional layer.
- Computes the Grad-CAM heatmap.
- Creates an overlay over the original image.
- Saves a summary figure and prediction information.

This helped me perform a qualitative analysis of the model behaviour.

---

## 10. HPC and GPU execution

Some experiments were prepared for Linux/GPU/HPC execution using Slurm.

The repository includes an `sbatch` script for running experiments on GPU nodes.

The HPC part was useful because high-resolution images and large CNN models require more computational resources than simple local experiments.

Some practical issues I had to handle were:

- GPU memory limits.
- Smaller batch sizes for 768×768 images.
- Long training jobs.
- Checkpoint saving.
- Offline pretrained weights.
- Structured experiment outputs.
- Logs and metrics recovery after execution.

---

## 11. Technical problems I solved

During the project, I worked on several practical technical problems:

- Dataset organization and validation.
- Duplicate image detection.
- Avoiding data leakage.
- Class imbalance handling.
- Model comparison under the same evaluation logic.
- GPU/HPC execution.
- Local loading of pretrained weights.
- Checkpoint management.
- Export of metrics and predictions.
- Confusion matrix analysis.
- Grad-CAM interpretability.
- Documentation of experiments.

These parts were important because they made the project closer to a real research workflow, not just a notebook experiment.

---

## 12. How this project connects with bioacoustics

Although this project was developed for medical image classification, the technical workflow can be adapted to other domains.

For example, in bioacoustics, audio recordings can be converted into spectrograms. A spectrogram is a visual representation of sound, so many Computer Vision methods can be applied.

The reusable parts of this project are:

- Dataset organization.
- Image/spectrogram classification.
- CNN and Transformer-based models.
- Transfer Learning.
- Train/validation/test splitting.
- Metric export.
- Confusion matrix analysis.
- Reproducible experiments.
- HPC execution.

This is why the experience is technically relevant for projects involving audio-based classification, such as bird sound recognition.

---

## 13. Possible interview questions and answers

### What was your role in the project?

My role was to develop and evaluate Deep Learning models for chest X-ray classification. I worked on the full pipeline: dataset preparation, model training, evaluation, comparison of architectures, Grad-CAM interpretability and experiment documentation.

### Why did you use Transfer Learning?

Because the dataset was limited compared to large-scale image datasets. Transfer Learning allowed me to start from models pretrained on large image datasets and adapt them to the medical classification task.

### Why EfficientNet-B7?

EfficientNet-B7 provided the best confirmed balance between performance and stability in the experiments. It achieved the strongest combination of Accuracy, Macro-F1 and AUC macro OVR.

### Why was AUC important?

AUC was important because it measures the discriminative capacity of the model using probabilities. In a multiclass medical task, this gives more information than only checking the final predicted class.

### Why not only accuracy?

Because the dataset was imbalanced. Accuracy can hide poor performance on minority classes. Macro-F1 and per-class metrics give a more complete view.

### What did Grad-CAM add?

Grad-CAM allowed me to inspect which image regions influenced the prediction. This is important in medical imaging because interpretability helps understand whether the model is focusing on meaningful regions.

### What was the main limitation?

The main limitation was that the dataset was not very large and the task was multiclass with visually similar findings. External validation with another dataset would be a useful future improvement.

### What would you improve next?

I would test medical pretrained weights, add experiment tracking with MLflow or Weights & Biases, improve probability calibration and perform external validation.

---

## 14. Short oral explanation

This is a short version I can use to explain the project:

> During my internship at the University of Jaén, I worked on a Deep Learning pipeline for multi-class classification of chest X-ray images. I prepared and cleaned a dataset of more than 4,100 images, removed duplicates, created train, validation and test splits, and trained several CNN architectures using PyTorch and Transfer Learning. I compared models such as DenseNet121, MobileNetV3, EfficientNet-B7 and an experimental EfficientNetV2 + Transformer model. The best confirmed result was obtained with EfficientNet-B7 at 768×768 resolution, reaching an AUC macro OVR of 0.9113, Macro-F1 of 0.6681 and Accuracy of 0.6742. I also added Grad-CAM interpretability, confusion matrix analysis and GPU/HPC execution support with Slurm.

---

## 15. Final takeaway

The most important part of the project was learning how to build a complete Deep Learning workflow.

The model was only one part of the work. The complete process included data quality, reproducibility, evaluation, interpretability and infrastructure.

This experience helped me improve in PyTorch, Computer Vision, Deep Learning experimentation, GPU/HPC execution and technical documentation.
