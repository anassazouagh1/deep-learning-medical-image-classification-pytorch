# Experiment Summary

This document summarizes the main experiments carried out during the internship workflow for multi-class chest X-ray classification.

The objective of these experiments was to compare different Deep Learning architectures under the same dataset structure and evaluation logic. The focus was not only on accuracy, but also on Macro-F1 and AUC macro OVR, because the task was multiclass and imbalanced.

---

## Dataset used for the experiments

The final audited dataset contained **4,105 unique chest X-ray images** after removing duplicated samples.

| Split | Images |
|---|---:|
| Train | 2,872 |
| Validation | 616 |
| Test | 617 |

The task included seven diagnostic categories:

- Atelectasis
- Effusion
- Emphysema
- No finding
- Nodule
- Pneumonia
- Pneumothorax

---

## Main comparison

| Model | Input size | Accuracy | Macro-F1 | AUC macro OVR | Notes |
|---|---:|---:|---:|---:|---|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 | Best confirmed experiment |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 | Strong baseline with lower input size |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 | Lightweight model with good AUC |
| EfficientNetV2 + Transformer / ViT-style head | 768×768 | 0.5851 | 0.5861 | 0.8791 | Experimental hybrid CNN-Transformer |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 | Lower performance in this setup |

---

## Best confirmed experiment

The best confirmed model was:

```text
EfficientNet-B7 at 768×768 resolution
```

Final test-set results:

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |
| Macro Recall / Sensitivity | 0.6863 |
| Macro Specificity | 0.9445 |
| Test loss | 1.1676 |

The most relevant result was the **AUC macro OVR of 0.9113**, because it showed strong discriminative capacity across the seven diagnostic classes.

Accuracy was also reported, but it was not the only metric considered. In this task, Macro-F1 and AUC were especially important because the dataset was imbalanced and the model had to distinguish between several visually similar findings.

---

## Experiment 1: EfficientNet-B7 768×768

### Objective

Test whether a high-capacity CNN with high-resolution input could improve the classification of chest X-rays.

### Configuration

| Parameter | Value |
|---|---|
| Model | EfficientNet-B7 |
| Input size | 768×768 |
| Number of classes | 7 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Loss | CrossEntropyLoss |
| Class imbalance | Class weights + WeightedRandomSampler |
| Scheduler | Cosine learning-rate scheduling |
| Early stopping | Enabled |
| Mixed precision | Enabled when CUDA was available |

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.6742 |
| Macro-F1 | 0.6681 |
| AUC macro OVR | 0.9113 |

### Technical conclusion

EfficientNet-B7 gave the best overall result. The higher input resolution helped preserve relevant visual details in the radiographs, although it also increased GPU memory usage and required smaller batch sizes.

This model was selected as the strongest confirmed experiment because it obtained the best balance between performance, stability and discriminative capacity.

---

## Experiment 2: EfficientNetV2 512×512

### Objective

Evaluate EfficientNetV2 as a modern CNN backbone with lower input resolution.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.6321 |
| Macro-F1 | 0.6305 |
| AUC macro OVR | 0.8883 |

### Technical conclusion

EfficientNetV2 produced a solid baseline result. It was more computationally manageable than the 768×768 experiments, but the lower input size may have limited the preservation of fine visual details.

The model performed reasonably well but did not outperform EfficientNet-B7 at higher resolution.

---

## Experiment 3: MobileNetV3 512×512

### Objective

Test a lightweight CNN architecture and evaluate the trade-off between performance and computational efficiency.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.6143 |
| Macro-F1 | 0.6133 |
| AUC macro OVR | 0.8949 |

### Technical conclusion

MobileNetV3 achieved a good AUC considering its lightweight design. This suggests that smaller architectures can still be useful when computational resources are limited.

However, it did not reach the same final balance as EfficientNet-B7.

---

## Experiment 4: EfficientNetV2 + Transformer / ViT-style head

### Objective

Explore a hybrid architecture combining convolutional feature extraction with a Transformer-style classification module.

The idea was to use:

- EfficientNetV2 as CNN backbone.
- Spatial tokens extracted from the CNN feature map.
- Transformer encoder layers to model global relations between image regions.
- A ViT-style classification token for final prediction.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.5851 |
| Macro-F1 | 0.5861 |
| AUC macro OVR | 0.8791 |

### Technical conclusion

The hybrid model did not outperform the best CNN-only configuration. However, it was useful as an experimental prototype because it allowed comparison between:

- Pure CNN-based classification.
- CNN feature extraction combined with self-attention.
- Local visual representation versus global spatial representation.

The result showed that adding a Transformer-style head increased architectural complexity and computational cost, but did not improve performance in this specific setup.

This experiment was still valuable because it helped analyze model complexity, stability and generalization.

---

## Experiment 5: DenseNet121 768×768

### Objective

Evaluate DenseNet121 as a CNN baseline at high resolution.

### Results

| Metric | Value |
|---|---:|
| Accuracy | 0.4814 |
| Macro-F1 | 0.4570 |
| AUC macro OVR | 0.8438 |

### Technical conclusion

DenseNet121 obtained the weakest result among the confirmed experiments. Although DenseNet architectures are useful in many medical imaging tasks, in this specific setup it did not generalize as well as the EfficientNet-based models.

---

## Why Macro-F1 and AUC were important

Accuracy alone was not enough to evaluate the models because the dataset was imbalanced.

For example, if a model performs well on majority classes but poorly on minority classes, accuracy can still look acceptable. Macro-F1 gives equal importance to all classes, making it more appropriate for this task.

AUC macro OVR was also important because it measures the discriminative capacity of the model using predicted probabilities. This is especially useful in medical classification problems where confidence and class separation matter.

---

## Main lessons from the experiments

The experiments showed several important points:

- Higher input resolution improved the best model, but increased GPU memory requirements.
- EfficientNet-B7 was the strongest confirmed architecture.
- Lightweight models like MobileNetV3 can still provide competitive AUC.
- The hybrid CNN-Transformer prototype was technically useful, but not better than EfficientNet-B7 in this setup.
- Macro-F1 and AUC macro OVR were more informative than accuracy alone.
- Class imbalance handling was necessary for a fairer training process.
- Reproducible configuration files made model comparison easier.
- Grad-CAM and saliency maps added qualitative support to the evaluation.

---

## Final conclusion

The best confirmed experiment was **EfficientNet-B7 at 768×768 resolution**, achieving:

```text
Accuracy:       0.6742
Macro-F1:       0.6681
AUC macro OVR:  0.9113
```

This model provided the best overall performance and was selected as the strongest confirmed result of the internship workflow.

The other experiments were useful to understand the behaviour of different model families and the trade-off between computational cost, complexity and performance.
