from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except ValueError:
        return float("nan")


def specificity_from_confusion_matrix(cm: np.ndarray) -> tuple[float, list[float]]:
    per_class = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        per_class.append(float(tn / denom) if denom else float("nan"))
    return float(np.nanmean(per_class)), per_class


def compute_metrics(y_true, y_pred, y_prob=None, num_classes: int | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1) if len(y_true) else 0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_macro, _ = specificity_from_confusion_matrix(cm)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity_macro": specificity_macro,
    }
    if y_prob is not None:
        metrics["auc_macro_ovr"] = _safe_auc(y_true, np.asarray(y_prob), num_classes)
    return metrics


def per_class_report(y_true, y_pred, classes: list[str]) -> list[dict[str, float | str]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    _, spec = specificity_from_confusion_matrix(cm)
    report = []
    for idx, label in enumerate(classes):
        binary_true = (y_true == idx).astype(int)
        binary_pred = (y_pred == idx).astype(int)
        report.append({
            "class": label,
            "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(binary_true, binary_pred, zero_division=0)),
            "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            "specificity": spec[idx],
        })
    return report
