import numpy as np

from ids_repro.metrics import assert_evaluation_consistency, evaluate_predictions


def test_binary_metrics_use_attack_as_positive():
    metrics, matrix, _ = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        task="binary",
        class_names=("normal", "attack"),
    )
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert metrics["recall_attack"] == 1.0
    assert metrics["precision_attack"] == 2 / 3
    assert metrics["f1_micro"] == metrics["accuracy"]
    assert "matthews_correlation_coefficient" in metrics
    assert metrics["predicted_BENIGN/normal"] == 1
    assert metrics["predicted_attack"] == 3


def test_binary_probability_metrics_are_computed_from_scores():
    metrics, _, _ = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        task="binary",
        class_names=("normal", "attack"),
        probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_confusion_and_report_consistency_guard():
    truth = np.array([0, 0, 1])
    prediction = np.array([0, 1, 1])
    _, matrix, report = evaluate_predictions(
        truth, prediction, task="binary", class_names=("normal", "attack")
    )
    bad = matrix.copy()
    bad[0, 0] += 1
    import pytest

    with pytest.raises(AssertionError, match="total"):
        assert_evaluation_consistency(
            truth,
            prediction,
            bad,
            report,
            labels=[0, 1],
            target_names=["BENIGN/normal", "attack"],
        )
