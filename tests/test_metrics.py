from raravis_dl.metrics import compute_metrics


def test_compute_metrics_basic():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_prob = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, num_classes=2)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == 0.75
