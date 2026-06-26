# Experiment Summary

## Dataset

| Class | Original | Unique | Duplicates removed |
|---|---:|---:|---:|
| Atelectasis | 682 | 682 | 0 |
| Effusion | 533 | 533 | 0 |
| Emphysema | 332 | 332 | 0 |
| No finding | 1066 | 1060 | 6 |
| Nodule | 656 | 656 | 0 |
| Pneumonia | 497 | 497 | 0 |
| Pneumothorax | 345 | 345 | 0 |
| **Total** | **4111** | **4105** | **6** |

## Split

| Split | Images | Approx. ratio |
|---|---:|---:|
| Train | 2872 | 70% |
| Validation | 616 | 15% |
| Test | 617 | 15% |

## Model comparison

| Model | Resolution | Accuracy | Macro-F1 | AUC macro OVR | Notes |
|---|---:|---:|---:|---:|---|
| EfficientNet-B7 | 768×768 | 0.6742 | 0.6681 | 0.9113 | Best confirmed experiment |
| EfficientNetV2 | 512×512 | 0.6321 | 0.6305 | 0.8883 | Strong baseline |
| MobileNetV3 | 512×512 | 0.6143 | 0.6133 | 0.8949 | Lightweight model |
| EffNetV2 + Transformer | 768×768 | 0.5851 | 0.5861 | 0.8791 | Experimental hybrid |
| DenseNet121 | 768×768 | 0.4814 | 0.4570 | 0.8438 | Lower generalization |

## Best model confusion matrix

See `results/confusion_matrix_best.csv` and `docs/assets/confusion_matrix_best.png`.
