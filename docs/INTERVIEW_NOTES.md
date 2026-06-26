# Interview Notes

## How to explain the project in 45 seconds

I developed a PyTorch pipeline for multi-class chest X-ray classification. The work included dataset cleaning, duplicate detection, train/validation/test split, model training, evaluation and interpretability. I compared several CNN architectures such as ResNet, DenseNet, MobileNet and EfficientNet, trained the experiments on GPU using a Slurm-based HPC cluster, and exported metrics, predictions and confusion matrices for reproducibility. The best confirmed model reached an AUC macro OVR of 0.91 with EfficientNet-B7 at 768×768 resolution.

## Connection with an ecoacoustics / PAM project

Although this project uses medical images, the technical workflow transfers well to ecoacoustics:

- audio can be transformed into spectrograms and treated as image-like inputs,
- CNNs can classify species or acoustic events,
- the same training/evaluation pipeline can be adapted to audio labels,
- class imbalance and rare-event detection are common in both domains,
- reproducible experiment tracking is essential for research projects,
- Grad-CAM can be replaced or adapted to spectrogram saliency.

## Technical points to highlight

- I did not only train a model; I built the complete workflow.
- I worked with Linux, GPU, Slurm and structured experiment folders.
- I handled practical issues such as duplicated data, class imbalance and offline pretrained weights.
- I used metrics beyond accuracy: Macro-F1, AUC, sensitivity and specificity.
- I generated interpretable outputs and documented results for academic reporting.
