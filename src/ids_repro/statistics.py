from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

CONTROLLED_SEEDS = (42, 52, 62, 72, 82)


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
