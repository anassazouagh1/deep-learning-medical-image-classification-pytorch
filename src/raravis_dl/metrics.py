from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_numpy(values: list[int] | list[list[float]] | np.ndarray) -> np.ndarray:
    """Convert lists or arrays into a NumPy array."""
    return np.asarray(values)


def _safe_float(value: Any) -> float | None:
    """Convert metric values to standard Python floats.

    NaN values are converted to None to make JSON export cleaner and easier to
    read in experiment reports.
    """
    try:
        value = float(value)
    except Exception:
        return None

    if np.isnan(value) or np.isinf(value):
        return None

    return value


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_specificity_per_class(cm: np.ndarray) -> list[float]:
    """Compute specificity for each class from a confusion matrix.

    Specificity is calculated as:

    TN / (TN + FP)

    In a multiclass setting, each class is treated as one-vs-rest.
    """
    specificities: list[float] = []

    total = np.sum(cm)

    for class_idx in range(cm.shape[0]):
        true_positive = cm[class_idx, class_idx]
        false_positive = np.sum(cm[:, class_idx]) - true_positive
        false_negative = np.sum(cm[class_idx, :]) - true_positive
        true_negative = total - true_positive - false_positive - false_negative

        specificity = _safe_divide(
            float(true_negative),
            float(true_negative + false_positive),
        )

        specificities.append(float(specificity))

    return specificities


def compute_sensitivity_per_class(cm: np.ndarray) -> list[float]:
    """Compute sensitivity/recall for each class from a confusion matrix."""
    sensitivities: list[float] = []

    for class_idx in range(cm.shape[0]):
        true_positive = cm[class_idx, class_idx]
        false_negative = np.sum(cm[class_idx, :]) - true_positive

        sensitivity = _safe_divide(
            float(true_positive),
            float(true_positive + false_negative),
        )

        sensitivities.append(float(sensitivity))

    return sensitivities


def compute_auc_macro_ovr(
    y_true: list[int] | np.ndarray,
    y_prob: list[list[float]] | np.ndarray | None,
    num_classes: int,
) -> float | None:
    """Compute multiclass AUC macro OVR.

    AUC is useful because it measures the discriminative capacity of the model
    using class probabilities, not only the final predicted class.

    The function is defensive because AUC can fail when a validation or test
    split does not contain at least one sample from every class.
    """
    if y_prob is None:
        return None

    y_true_np = _as_numpy(y_true).astype(int)
    y_prob_np = _as_numpy(y_prob).astype(float)

    if y_prob_np.ndim != 2:
        return None

    if y_prob_np.shape[1] != num_classes:
        return None

    try:
        auc = roc_auc_score(
            y_true_np,
            y_prob_np,
            labels=list(range(num_classes)),
            multi_class="ovr",
            average="macro",
        )
        return _safe_float(auc)
    except Exception:
        return None


def compute_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix with fixed class labels."""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Normalize a confusion matrix by row.

    Each row represents the real class. The normalized value shows the
    percentage of samples from that class predicted as each possible class.
    """
    row_sums = cm.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(cm, row_sums, where=row_sums != 0)

    normalized[np.isnan(normalized)] = 0.0
    return normalized


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[list[float]] | np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float | int | None]:
    """Compute global classification metrics.

    The project uses more than accuracy because the dataset is multiclass and
    imbalanced. Macro-F1, AUC macro OVR, sensitivity and specificity give a more
    complete view of model behaviour.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.
    y_pred:
        Predicted class indices.
    y_prob:
        Optional class probability matrix with shape [n_samples, n_classes].
    num_classes:
        Number of classes. If None, it is inferred from y_true, y_pred and y_prob.

    Returns
    -------
    dict
        Dictionary with global metrics.
    """
    y_true_np = _as_numpy(y_true).astype(int)
    y_pred_np = _as_numpy(y_pred).astype(int)

    if y_true_np.size == 0:
        raise ValueError("Cannot compute metrics with empty y_true.")

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got {y_true_np.shape[0]} and {y_pred_np.shape[0]}."
        )

    if num_classes is None:
        inferred_from_labels = int(max(y_true_np.max(), y_pred_np.max()) + 1)

        if y_prob is not None:
            y_prob_np = _as_numpy(y_prob)
            if y_prob_np.ndim == 2:
                inferred_from_probs = int(y_prob_np.shape[1])
                num_classes = max(inferred_from_labels, inferred_from_probs)
            else:
                num_classes = inferred_from_labels
        else:
            num_classes = inferred_from_labels

    labels = list(range(num_classes))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    cm = compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)
    specificities = compute_specificity_per_class(cm)
    sensitivities = compute_sensitivity_per_class(cm)

    auc_macro_ovr = compute_auc_macro_ovr(
        y_true=y_true_np,
        y_prob=y_prob,
        num_classes=num_classes,
    )

    metrics = {
        "accuracy": _safe_float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": _safe_float(
            balanced_accuracy_score(y_true_np, y_pred_np)
        ),
        "precision_macro": _safe_float(precision_macro),
        "recall_macro": _safe_float(recall_macro),
        "f1_macro": _safe_float(f1_macro),
        "precision_weighted": _safe_float(precision_weighted),
        "recall_weighted": _safe_float(recall_weighted),
        "f1_weighted": _safe_float(f1_weighted),
        "auc_macro_ovr": auc_macro_ovr,
        "sensitivity_macro": _safe_float(np.mean(sensitivities)),
        "specificity_macro": _safe_float(np.mean(specificities)),
        "num_classes": int(num_classes),
        "num_samples": int(y_true_np.shape[0]),
    }

    return metrics


def per_class_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    classes: list[str],
) -> list[dict[str, Any]]:
    """Generate a per-class report.

    The report includes precision, recall/sensitivity, F1-score, support and
    specificity for each class.
    """
    y_true_np = _as_numpy(y_true).astype(int)
    y_pred_np = _as_numpy(y_pred).astype(int)

    num_classes = len(classes)
    labels = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        average=None,
        zero_division=0,
    )

    cm = compute_confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)
    specificities = compute_specificity_per_class(cm)

    report: list[dict[str, Any]] = []

    for idx, class_name in enumerate(classes):
        report.append(
            {
                "class_index": int(idx),
                "class_name": class_name,
                "precision": _safe_float(precision[idx]),
                "recall": _safe_float(recall[idx]),
                "sensitivity": _safe_float(recall[idx]),
                "specificity": _safe_float(specificities[idx]),
                "f1_score": _safe_float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    return report


def confusion_matrix_dataframe(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    classes: list[str],
    normalize: bool = False,
) -> pd.DataFrame:
    """Return a confusion matrix as a labeled pandas DataFrame."""
    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=len(classes),
    )

    if normalize:
        cm = normalize_confusion_matrix(cm)

    index = [f"true_{name}" for name in classes]
    columns = [f"pred_{name}" for name in classes]

    return pd.DataFrame(cm, index=index, columns=columns)


def summarize_metric_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Convert a metrics dictionary into a two-column DataFrame."""
    rows = []

    for key, value in metrics.items():
        rows.append(
            {
                "metric": key,
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def compare_experiments_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a clean comparison table for multiple experiments.

    Expected keys in each row can include:
    model, input_size, accuracy, f1_macro, auc_macro_ovr.
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    preferred_columns = [
        "model",
        "input_size",
        "accuracy",
        "f1_macro",
        "auc_macro_ovr",
        "sensitivity_macro",
        "specificity_macro",
        "loss",
    ]

    ordered_columns = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]

    return df[ordered_columns + remaining_columns]
