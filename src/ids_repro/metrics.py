from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    task: str,
    class_names: Sequence[str],
    probabilities: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, dict]:
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("truth and prediction must be aligned rank-1 arrays")
    labels = list(range(2 if task == "binary" else len(class_names)))
    target_names = ["BENIGN/normal", "attack"] if task == "binary" else list(class_names)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    report = classification_report(
        truth,
        prediction,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    if task == "binary":
        tn, fp, fn, tp = matrix.ravel()
        metrics = {
            "accuracy": accuracy_score(truth, prediction),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "precision_attack": precision_score(truth, prediction, pos_label=1, zero_division=0),
            "recall_attack": recall_score(truth, prediction, pos_label=1, zero_division=0),
            "f1_attack": f1_score(truth, prediction, pos_label=1, zero_division=0),
            "matthews_correlation_coefficient": matthews_corrcoef(truth, prediction),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
            "error_rate": (fp + fn) / matrix.sum() if matrix.sum() else 0.0,
            "precision_macro": precision_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "recall_micro": recall_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "f1_micro": f1_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "precision_weighted": precision_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
        }
        if probabilities is not None:
            scores = np.asarray(probabilities).reshape(-1)
            if len(scores) != len(truth):
                raise ValueError("Binary probabilities do not align with truth")
            if len(np.unique(truth)) == 2:
                metrics["roc_auc"] = roc_auc_score(truth, scores)
                metrics["pr_auc"] = average_precision_score(truth, scores)
            else:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
    else:
        metrics = {
            "accuracy": accuracy_score(truth, prediction),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
            "matthews_correlation_coefficient": matthews_corrcoef(truth, prediction),
            "precision_macro": precision_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                truth, prediction, labels=labels, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "recall_micro": recall_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "f1_micro": f1_score(
                truth, prediction, labels=labels, average="micro", zero_division=0
            ),
            "precision_weighted": precision_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                truth, prediction, labels=labels, average="weighted", zero_division=0
            ),
            "error_rate": 1.0 - accuracy_score(truth, prediction),
        }
        if probabilities is not None:
            scores = np.asarray(probabilities)
            if scores.shape != (len(truth), len(labels)):
                raise ValueError(
                    "Multiclass probabilities must have shape (samples, classes)"
                )
            if set(np.unique(truth)) == set(labels):
                binary_truth = label_binarize(truth, classes=labels)
                metrics["roc_auc_macro_ovr"] = roc_auc_score(
                    binary_truth, scores, average="macro", multi_class="ovr"
                )
                metrics["pr_auc_macro"] = average_precision_score(
                    binary_truth, scores, average="macro"
                )
            else:
                metrics["roc_auc_macro_ovr"] = None
                metrics["pr_auc_macro"] = None
    for code, name in enumerate(target_names):
        metrics[f"support_{name}"] = int(np.sum(truth == code))
        metrics[f"predicted_{name}"] = int(np.sum(prediction == code))
    assert_evaluation_consistency(
        truth, prediction, matrix, report, labels=labels, target_names=target_names
    )
    return metrics, matrix, report


def assert_evaluation_consistency(
    truth: np.ndarray,
    prediction: np.ndarray,
    matrix: np.ndarray,
    report: dict,
    *,
    labels: Sequence[int],
    target_names: Sequence[str],
) -> None:
    """Fail if saved-array counts and report/confusion totals disagree."""

    if int(matrix.sum()) != len(truth) or len(prediction) != len(truth):
        raise AssertionError("Confusion-matrix total and saved array lengths differ")
    truth_counts = np.bincount(np.asarray(truth, dtype=np.int64), minlength=len(labels))
    prediction_counts = np.bincount(
        np.asarray(prediction, dtype=np.int64), minlength=len(labels)
    )
    if not np.array_equal(matrix.sum(axis=1), truth_counts[: len(labels)]):
        raise AssertionError("Confusion-matrix row totals do not equal truth support")
    if not np.array_equal(matrix.sum(axis=0), prediction_counts[: len(labels)]):
        raise AssertionError("Confusion-matrix column totals do not equal prediction counts")
    for index, name in enumerate(target_names):
        if int(report[name]["support"]) != int(matrix[index].sum()):
            raise AssertionError(f"Classification-report support differs for {name}")


def matrix_distance(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"Matrix shape mismatch: actual={actual.shape}, expected={expected.shape}")
    difference = actual.astype(np.int64) - expected.astype(np.int64)
    return {
        "absolute_cell_error": int(np.abs(difference).sum()),
        "maximum_cell_error": int(np.abs(difference).max(initial=0)),
        "matching_cells": int((difference == 0).sum()),
        "total_cells": int(difference.size),
        "signed_cell_difference": difference.tolist(),
        "actual_errors": int(actual.sum() - np.trace(actual)),
        "expected_errors": int(expected.sum() - np.trace(expected)),
    }


def predictions_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix into labels for independently checking paper metrics."""

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError("matrix must be a non-negative square array")
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    for actual in range(matrix.shape[0]):
        for predicted in range(matrix.shape[1]):
            count = int(matrix[actual, predicted])
            if count:
                truth.append(np.full(count, actual, dtype=np.uint8))
                prediction.append(np.full(count, predicted, dtype=np.uint8))
    return np.concatenate(truth), np.concatenate(prediction)
