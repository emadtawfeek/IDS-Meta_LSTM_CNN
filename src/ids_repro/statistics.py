from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

CONTROLLED_SEEDS = (42, 52, 62, 72, 82)


def mcnemar_comparison(
    first_prediction: Sequence[int],
    second_prediction: Sequence[int],
    first_truth: Sequence[int],
    second_truth: Sequence[int],
    *,
    first_split_checksum: str,
    second_split_checksum: str,
    exact_threshold: int = 25,
) -> dict:
    """Prediction-level paired McNemar test on an explicitly identical split."""

    first_prediction = np.asarray(first_prediction)
    second_prediction = np.asarray(second_prediction)
    first_truth = np.asarray(first_truth)
    second_truth = np.asarray(second_truth)
    shapes = {
        first_prediction.shape,
        second_prediction.shape,
        first_truth.shape,
        second_truth.shape,
    }
    if len(shapes) != 1 or first_truth.ndim != 1:
        raise ValueError("McNemar inputs must be aligned rank-1 arrays")
    if not first_split_checksum or first_split_checksum != second_split_checksum:
        raise ValueError("McNemar comparison requires identical non-empty split checksums")
    if not np.array_equal(first_truth, second_truth):
        raise ValueError("McNemar y_true arrays are not identical")

    first_correct = first_prediction == first_truth
    second_correct = second_prediction == first_truth
    both_correct = int(np.sum(first_correct & second_correct))
    first_only_correct = int(np.sum(first_correct & ~second_correct))
    second_only_correct = int(np.sum(~first_correct & second_correct))
    both_wrong = int(np.sum(~first_correct & ~second_correct))
    discordant = first_only_correct + second_only_correct
    if discordant == 0:
        test_name = "exact_binomial"
        statistic = 0.0
        p_value = 1.0
    elif discordant <= exact_threshold:
        test_name = "exact_binomial"
        statistic = float(min(first_only_correct, second_only_correct))
        p_value = float(
            stats.binomtest(
                min(first_only_correct, second_only_correct),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        test_name = "mcnemar_chi_square_continuity_corrected"
        statistic = (abs(first_only_correct - second_only_correct) - 1.0) ** 2 / discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {
        "n": int(len(first_truth)),
        "split_checksum": first_split_checksum,
        "discordant_table": {
            "both_correct": both_correct,
            "first_only_correct": first_only_correct,
            "second_only_correct": second_only_correct,
            "both_wrong": both_wrong,
        },
        "discordant_total": discordant,
        "test": test_name,
        "exact_threshold": exact_threshold,
        "statistic": float(statistic),
        "p_value": p_value,
        "samples_each_method_alone_correct": {
            "first": first_only_correct,
            "second": second_only_correct,
        },
    }


def mcnemar_result_directories(
    first_dir: Path | str, second_dir: Path | str, output: Path | str
) -> dict:
    first_dir, second_dir = Path(first_dir), Path(second_dir)
    first_manifest = json.loads(
        (first_dir / "subset_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second_dir / "subset_manifest.json").read_text(encoding="utf-8")
    )
    first_dataset_split = first_manifest.get("dataset_split_checksum_sha256")
    second_dataset_split = second_manifest.get("dataset_split_checksum_sha256")
    first_subset = first_manifest["test"]["indices_checksum_sha256"]
    second_subset = second_manifest["test"]["indices_checksum_sha256"]
    if (
        not first_dataset_split
        or first_dataset_split != second_dataset_split
        or first_subset != second_subset
    ):
        raise ValueError(
            "McNemar run artifacts require identical dataset-split and test-subset checksums"
        )
    aligned_split_identity = f"{first_dataset_split}:{first_subset}"
    result = mcnemar_comparison(
        np.load(first_dir / "predictions.npy"),
        np.load(second_dir / "predictions.npy"),
        np.load(first_dir / "y_true.npy"),
        np.load(second_dir / "y_true.npy"),
        first_split_checksum=aligned_split_identity,
        second_split_checksum=aligned_split_identity,
    )
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(values: Sequence[float], confidence: float = 0.95) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two scalar runs are required")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    margin = critical * std / np.sqrt(len(values))
    return {
        "n": int(len(values)),
        "mean": mean,
        "standard_deviation": std,
        "confidence": confidence,
        "confidence_interval": [mean - margin, mean + margin],
    }


def paired_comparison(first: Sequence[float], second: Sequence[float]) -> dict:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 3:
        raise ValueError("Paired comparisons require aligned vectors with at least 3 runs")
    difference = first - second
    shapiro = stats.shapiro(difference)
    if shapiro.pvalue >= 0.05:
        test = stats.ttest_rel(first, second)
        test_name = "paired_t_test"
    else:
        test = stats.wilcoxon(first, second, zero_method="wilcox")
        test_name = "wilcoxon_signed_rank"
    standard_deviation = difference.std(ddof=1)
    effect = float(difference.mean() / standard_deviation) if standard_deviation else 0.0
    return {
        "n": int(len(first)),
        "normality_test": {
            "name": "shapiro_wilk",
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
        },
        "selected_test": test_name,
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "mean_paired_difference": float(difference.mean()),
        "effect_size_cohen_dz": effect,
        "interpretation_policy": "No significance claim unless p<0.05 and assumptions are documented",
    }


def analyze_result_files(
    paths: Sequence[Path | str], metric: str, output: Path | str
) -> dict:
    grouped: dict[str, list[float]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = payload.get("algorithm", payload.get("swarm_preset", path.stem))
        metrics = payload.get("metrics", payload)
        grouped.setdefault(str(label), []).append(float(metrics[metric]))
    result = {label: summarize(values) for label, values in grouped.items()}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
