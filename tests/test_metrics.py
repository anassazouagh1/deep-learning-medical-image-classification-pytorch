import numpy as np
import pandas as pd
import pytest

from raravis_dl.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    compute_sensitivity_per_class,
    compute_specificity_per_class,
    confusion_matrix_dataframe,
    normalize_confusion_matrix,
    per_class_report,
    summarize_metric_table,
)


def test_compute_metrics_basic_multiclass_case():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 1, 1]

    y_prob = [
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.10, 0.60, 0.30],
        [0.20, 0.70, 0.10],
    ]

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=3,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["num_classes"] == 3
    assert metrics["num_samples"] == 4

    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "auc_macro_ovr" in metrics
    assert "sensitivity_macro" in metrics
    assert "specificity_macro" in metrics

    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["specificity_macro"] <= 1.0


def test_compute_metrics_without_probabilities():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
        num_classes=2,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["auc_macro_ovr"] is None
    assert metrics["num_classes"] == 2
    assert metrics["num_samples"] == 4


def test_compute_metrics_raises_error_for_empty_input():
    with pytest.raises(ValueError):
        compute_metrics(
            y_true=[],
            y_pred=[],
            y_prob=None,
            num_classes=2,
        )


def test_compute_metrics_raises_error_for_different_lengths():
    with pytest.raises(ValueError):
        compute_metrics(
            y_true=[0, 1, 1],
            y_pred=[0, 1],
            y_prob=None,
            num_classes=2,
        )


def test_confusion_matrix_fixed_labels():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 0]

    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=3,
    )

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)
    assert cm.sum() == 5


def test_specificity_and_sensitivity_per_class():
    cm = np.array(
        [
            [2, 0, 0],
            [1, 3, 0],
            [0, 1, 4],
        ]
    )

    specificities = compute_specificity_per_class(cm)
    sensitivities = compute_sensitivity_per_class(cm)

    assert len(specificities) == 3
    assert len(sensitivities) == 3

    for value in specificities:
        assert 0.0 <= value <= 1.0

    for value in sensitivities:
        assert 0.0 <= value <= 1.0


def test_normalize_confusion_matrix_rows():
    cm = np.array(
        [
            [2, 0],
            [1, 1],
        ]
    )

    normalized = normalize_confusion_matrix(cm)

    assert normalized.shape == (2, 2)
    assert normalized[0].sum() == pytest.approx(1.0)
    assert normalized[1].sum() == pytest.approx(1.0)


def test_per_class_report_structure():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    classes = ["Atelectasis", "Effusion", "Pneumonia"]

    report = per_class_report(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
    )

    assert isinstance(report, list)
    assert len(report) == 3

    first_row = report[0]

    expected_keys = {
        "class_index",
        "class_name",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1_score",
        "support",
    }

    assert expected_keys.issubset(first_row.keys())
    assert report[0]["class_name"] == "Atelectasis"


def test_confusion_matrix_dataframe():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    classes = ["Class A", "Class B"]

    df = confusion_matrix_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        normalize=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.index) == ["true_Class A", "true_Class B"]
    assert list(df.columns) == ["pred_Class A", "pred_Class B"]


def test_normalized_confusion_matrix_dataframe():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    classes = ["Class A", "Class B"]

    df = confusion_matrix_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        normalize=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)

    assert df.loc["true_Class A"].sum() == pytest.approx(1.0)
    assert df.loc["true_Class B"].sum() == pytest.approx(1.0)


def test_summarize_metric_table():
    metrics = {
        "accuracy": 0.75,
        "f1_macro": 0.70,
        "auc_macro_ovr": 0.82,
    }

    df = summarize_metric_table(metrics)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["metric", "value"]
    assert len(df) == 3
