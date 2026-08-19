from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .config import (
    PAPER_BINARY_CONFUSIONS,
    PAPER_CIC_MULTICLASS_NAMES,
    PAPER_MULTICLASS_CONFUSIONS,
)
from .data import deterministic_subset, load_prepared
from .metrics import (
    evaluate_predictions,
    matrix_distance,
    predictions_from_matrix,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def paper_matrix_metrics() -> dict:
    """Recalculate every aggregate from the published matrices, without mixing schemes."""

    result: dict[str, dict] = {"binary": {}, "multiclass": {}}
    for algorithm, values in PAPER_BINARY_CONFUSIONS.items():
        matrix = np.asarray(values, dtype=np.int64)
        truth, prediction = predictions_from_matrix(matrix)
        metrics, _, report = evaluate_predictions(
            truth,
            prediction,
            task="binary",
            class_names=("BENIGN", "attack"),
        )
        result["binary"][algorithm] = {
            "matrix": matrix.tolist(),
            "metrics": metrics,
            "classification_report": report,
        }
    for algorithm, values in PAPER_MULTICLASS_CONFUSIONS.items():
        matrix = np.asarray(values, dtype=np.int64)
        truth, prediction = predictions_from_matrix(matrix)
        metrics, _, report = evaluate_predictions(
            truth,
            prediction,
            task="multiclass",
            class_names=PAPER_CIC_MULTICLASS_NAMES,
        )
        result["multiclass"][algorithm] = {
            "matrix": matrix.tolist(),
            "metrics": metrics,
            "classification_report": report,
        }
    return result


def audit_saved_run(
    run_dir: Path | str,
    cache_dir: Path | str,
    output_dir: Path | str,
    *,
    task: str,
    paper_algorithm: str,
) -> dict:
    """Independently audit a saved run and write only to a new directory."""

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audit artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.npy"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions_path}")
    predictions = np.load(predictions_path)
    dataset = load_prepared(cache_dir)
    metrics_path = run_dir / "metrics.json"
    saved_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    full_truth = dataset.labels("test", task)
    if len(predictions) == len(full_truth):
        truth = np.asarray(full_truth)
        truth_source = "complete cached test labels"
    else:
        seed = int(saved_metrics.get("seed", 42))
        claimed = saved_metrics.get("sample_counts", {}).get("test")
        if claimed != len(predictions):
            raise AssertionError(
                "Prediction count matches neither the test set nor the saved sample count"
            )
        indices = deterministic_subset(len(full_truth), len(predictions), seed + 2)
        truth = np.asarray(full_truth[indices])
        truth_source = "deterministic saved-run test subset reconstructed from seed"

    probabilities_path = run_dir / "probabilities.npy"
    probabilities = np.load(probabilities_path) if probabilities_path.exists() else None
    metrics, matrix, classification = evaluate_predictions(
        truth,
        predictions,
        task=task,
        class_names=dataset.class_names,
        probabilities=probabilities,
    )
    expected = np.asarray(
        PAPER_BINARY_CONFUSIONS[paper_algorithm]
        if task == "binary"
        else PAPER_MULTICLASS_CONFUSIONS[paper_algorithm],
        dtype=np.int64,
    )
    comparison = (
        matrix_distance(matrix, expected)
        if matrix.shape == expected.shape and matrix.sum() == expected.sum()
        else None
    )
    source_hashes = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    result = {
        "audit_type": "independent recalculation from saved predictions",
        "source_run": str(run_dir.resolve()),
        "source_cache": str(Path(cache_dir).resolve()),
        "truth_source": truth_source,
        "task": task,
        "paper_algorithm": paper_algorithm,
        "prediction_count": int(len(predictions)),
        "probabilities_available": probabilities is not None,
        "auc_status": (
            "recalculated from saved probabilities"
            if probabilities is not None
            else "unavailable: the legacy run did not save probabilities"
        ),
        "metrics": metrics,
        "paper_confusion_comparison": comparison,
        "saved_metrics_match_recalculation": {
            key: bool(np.isclose(saved_metrics.get("metrics", {}).get(key, np.nan), value))
            for key, value in metrics.items()
            if key in saved_metrics.get("metrics", {})
        },
        "source_artifact_sha256": source_hashes,
        "audit_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    np.save(output_dir / "y_true.npy", truth)
    np.save(output_dir / "predictions.npy", predictions)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    _write_json(output_dir / "classification_report.json", classification)
    _write_json(output_dir / "audit.json", result)
    return result


def write_paper_tables(output_dir: Path | str, submitted_audit: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = paper_matrix_metrics()
    if submitted_audit is not None:
        payload["submitted_saved_run"] = submitted_audit
    _write_json(output_dir / "paper_metrics_recalculated.json", payload)

    rows = [
        "protocol,task,optimizer,accuracy,balanced_accuracy,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted"
    ]
    markdown = [
        "| Source | Task | Optimizer | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("binary", "multiclass"):
        for algorithm in ("pso", "ssa"):
            metrics = payload[task][algorithm]["metrics"]
            row = [
                "paper_matrix",
                task,
                algorithm,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["precision_macro"],
                metrics["recall_macro"],
                metrics["f1_macro"],
                metrics["precision_weighted"],
                metrics["recall_weighted"],
                metrics["f1_weighted"],
            ]
            rows.append(",".join(str(value) for value in row))
            markdown.append(
                f"| Published matrix | {task} | {algorithm.upper()} | "
                f"{metrics['accuracy']:.6f} | {metrics['balanced_accuracy']:.6f} | "
                f"{metrics['precision_macro']:.6f} | {metrics['recall_macro']:.6f} | "
                f"{metrics['f1_macro']:.6f} | {metrics['f1_weighted']:.6f} |"
            )
    if submitted_audit is not None:
        metrics = submitted_audit["metrics"]
        row = [
            "submitted_saved_fixed_parameters",
            submitted_audit["task"],
            "fixed-ssa-preset",
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["precision_weighted"],
            metrics["recall_weighted"],
            metrics["f1_weighted"],
        ]
        rows.append(",".join(str(value) for value in row))
        markdown.append(
            f"| Submitted saved fixed-preset run | {submitted_audit['task']} | Fixed SSA preset | "
            f"{metrics['accuracy']:.6f} | {metrics['balanced_accuracy']:.6f} | "
            f"{metrics['precision_macro']:.6f} | {metrics['recall_macro']:.6f} | "
            f"{metrics['f1_macro']:.6f} | {metrics['f1_weighted']:.6f} |"
        )
    (output_dir / "paper_metrics_recalculated.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (output_dir / "paper_metrics_recalculated.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    if submitted_audit is not None:
        paper = payload[submitted_audit["task"]][submitted_audit["paper_algorithm"]][
            "metrics"
        ]
        comparison_rows = [
            "metric,submitted,paper,absolute_difference,percentage_point_difference"
        ]
        comparison_markdown = [
            "| Metric | Submitted | Paper | Absolute difference | Percentage-point difference |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ):
            submitted_value = float(submitted_audit["metrics"][name])
            paper_value = float(paper[name])
            difference = submitted_value - paper_value
            comparison_rows.append(
                f"{name},{submitted_value},{paper_value},{abs(difference)},{difference * 100.0}"
            )
            comparison_markdown.append(
                f"| {name} | {submitted_value:.9f} | {paper_value:.9f} | "
                f"{abs(difference):.9f} | {difference * 100.0:+.6f} |"
            )
        (output_dir / "submitted_vs_paper.csv").write_text(
            "\n".join(comparison_rows) + "\n", encoding="utf-8"
        )
        (output_dir / "submitted_vs_paper.md").write_text(
            "\n".join(comparison_markdown) + "\n", encoding="utf-8"
        )
    return payload
