from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score

from .config import (
    PAPER_BINARY_CONFUSIONS,
    PAPER_MULTICLASS_CONFUSIONS,
    FitnessName,
    HyperParameters,
    ModelName,
    ModelingMode,
    ProtocolName,
    SwarmName,
    Task,
)
from .data import PreparedDataset, deterministic_subset, reshape_for_model
from .metrics import evaluate_predictions, matrix_distance
from .models import build_model


def set_reproducible_seed(seed: int, deterministic_ops: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_ops:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def _select(features, labels, maximum, seed):
    indices = deterministic_subset(len(labels), maximum, seed)
    return features[indices], labels[indices], indices


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _environment(tf) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "physical_devices": [
            {"name": device.name, "type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS"),
    }


def _model_summary(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def _best_binary_threshold(truth, scores) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    candidates = np.unique(np.concatenate(([0.5], candidates)))
    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        score = f1_score(
            truth, scores >= threshold, average="macro", zero_division=0
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def _write_run_contract(
    output_dir: Path,
    *,
    dataset: PreparedDataset,
    task: Task,
    model_name: ModelName,
    swarm_name: SwarmName,
    params: HyperParameters,
    protocol: ProtocolName,
    modeling_mode: ModelingMode,
    seed: int,
    deterministic_ops: bool,
    threshold: float,
    sample_limits: dict,
    run_mode: str,
) -> None:
    contract = {
        "dataset": dataset.metadata["dataset"],
        "cache_dir": str(dataset.cache_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "task": task,
        "model": model_name,
        "protocol": protocol,
        "modeling_mode": modeling_mode,
        "optimizer": "fixed_published_preset_or_supplied_parameters",
        "run_mode": run_mode,
        "swarm_preset": swarm_name,
        "seed": seed,
        "deterministic_ops": deterministic_ops,
        "threshold": threshold,
        "sample_limits": sample_limits,
        "hyperparameters": params.to_dict(),
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )


def train_and_evaluate(
    dataset: PreparedDataset,
    *,
    task: Task,
    model_name: ModelName,
    swarm_name: SwarmName,
    params: HyperParameters,
    output_dir: Path | str,
    seed: int = 42,
    epochs_override: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    verbose: int = 2,
    protocol: ProtocolName = "paper_replication",
    modeling_mode: ModelingMode = "feature_axis_replication",
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    deterministic_ops: bool = True,
    allow_existing: bool = False,
    run_mode: str = "full",
) -> dict:
    import psutil
    import tensorflow as tf

    dataset.assert_integrity()
    cached_protocol = dataset.metadata.get("protocol")
    if cached_protocol is None and dataset.metadata.get("paper_split_match"):
        cached_protocol = "paper_replication"
    elif cached_protocol is None:
        cached_protocol = "legacy_unspecified"
    requested_rigorous = protocol in {"rigorous", "rigorous_evaluation"}
    cached_rigorous = cached_protocol in {"rigorous", "rigorous_evaluation"}
    if cached_protocol and (
        (requested_rigorous and not cached_rigorous)
        or (protocol == "paper_replication" and cached_protocol != "paper_replication")
    ):
        raise ValueError(
            f"Requested protocol={protocol} but cache was prepared as {cached_protocol}. "
            "Prepare a separate cache for the requested protocol."
        )
    if modeling_mode == "temporal_window" and not dataset.metadata.get(
        "temporal_window_supported", False
    ):
        raise ValueError(
            "Temporal-window modeling is unavailable: "
            + dataset.metadata.get(
                "temporal_window_reason", "dataset has no temporal identifiers"
            )
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    set_reproducible_seed(seed, deterministic_ops)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Refusing to overwrite existing run artifacts in {output_dir}. "
            "Choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if epochs_override is not None:
        if epochs_override < 1:
            raise ValueError("epochs_override must be positive")
        params = replace(params, epochs=epochs_override)

    train_x, train_y, train_subset = _select(
        dataset.train_x, dataset.labels("train", task), max_train_samples, seed
    )
    val_x, val_y, val_subset = _select(
        dataset.val_x, dataset.labels("val", task), max_val_samples, seed + 1
    )
    test_x, test_y, test_subset = _select(
        dataset.test_x, dataset.labels("test", task), max_test_samples, seed + 2
    )
    train_x = reshape_for_model(train_x, model_name, modeling_mode)
    val_x = reshape_for_model(val_x, model_name, modeling_mode)
    test_x = reshape_for_model(test_x, model_name, modeling_mode)

    class_count = 2 if task == "binary" else len(dataset.class_names)
    model = build_model(
        model_name,
        task,
        params,
        feature_count=(
            int(train_x.shape[2])
            if modeling_mode == "temporal_window"
            else len(dataset.feature_names)
        ),
        class_count=class_count,
        modeling_mode=modeling_mode,
        time_steps=(
            int(train_x.shape[1]) if modeling_mode == "temporal_window" else None
        ),
    )

    class EpochTimer(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.times: list[float] = []
            self._start = 0.0

        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.times.append(time.perf_counter() - self._start)

    timer = EpochTimer()
    callbacks: list = [timer]
    if requested_rigorous:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        )
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    fit_started = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=params.batch_size,
        epochs=params.epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    fit_seconds = time.perf_counter() - fit_started

    threshold_selection = {"policy": "fixed", "validation_score": None}
    if task == "binary" and optimize_threshold:
        val_scores = model.predict(
            val_x, batch_size=params.batch_size, verbose=0
        ).reshape(-1)
        threshold, validation_score = _best_binary_threshold(val_y, val_scores)
        threshold_selection = {
            "policy": "validation_macro_f1",
            "validation_score": validation_score,
        }

    predict_started = time.perf_counter()
    probabilities = model.predict(
        test_x, batch_size=params.batch_size, verbose=verbose
    )
    predict_seconds = time.perf_counter() - predict_started
    if task == "binary":
        probabilities = probabilities.reshape(-1)
        predictions = (probabilities >= threshold).astype(np.uint8)
    else:
        predictions = np.argmax(probabilities, axis=1).astype(np.uint8)
    metrics, matrix, report = evaluate_predictions(
        test_y,
        predictions,
        task=task,
        class_names=dataset.class_names,
        probabilities=probabilities,
    )

    expected = np.asarray(
        PAPER_BINARY_CONFUSIONS[swarm_name]
        if task == "binary"
        else PAPER_MULTICLASS_CONFUSIONS[swarm_name],
        dtype=np.int64,
    )
    comparison = None
    if matrix.shape == expected.shape and len(test_y) == int(expected.sum()):
        comparison = matrix_distance(matrix, expected)

    model.save(output_dir / "model.keras")
    (output_dir / "model_summary.txt").write_text(
        _model_summary(model), encoding="utf-8"
    )
    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (output_dir / "history.json").write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=","
    )
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "probabilities.npy", probabilities.astype(np.float32))
    np.save(output_dir / "y_true.npy", test_y)
    np.save(output_dir / "train_subset_indices.npy", train_subset)
    np.save(output_dir / "validation_subset_indices.npy", val_subset)
    np.save(output_dir / "test_subset_indices.npy", test_subset)
    (output_dir / "best_parameters.json").write_text(
        json.dumps(params.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "feature_names.json").write_text(
        json.dumps(list(dataset.feature_names), indent=2), encoding="utf-8"
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            dataset.metadata.get(
                "label_mapping",
                {name: index for index, name in enumerate(dataset.class_names)},
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    preprocessing_report = dataset.metadata.get(
        "preprocessing_report",
        {
            "status": "legacy cache metadata; detailed forensic counts are in AUDIT_AND_REPRODUCTION_REPORT.md",
            "normalization": dataset.metadata.get("normalization"),
            "raw_rows": dataset.metadata.get("raw_rows"),
            "invalid_rows_removed": dataset.metadata.get("invalid_rows_removed"),
            "duplicate_rows_removed_after_invalid_filter": dataset.metadata.get(
                "duplicate_rows_removed_after_invalid_filter"
            ),
        },
    )
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(_json_ready(preprocessing_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    preprocessor_path = dataset.cache_dir / "preprocessor.joblib"
    if preprocessor_path.exists():
        shutil.copy2(preprocessor_path, output_dir / "preprocessor.joblib")
    label_encoder_path = dataset.cache_dir / "label_encoder.joblib"
    if label_encoder_path.exists():
        shutil.copy2(label_encoder_path, output_dir / "label_encoder.joblib")
    mapping_report_path = dataset.cache_dir / "dataset_mapping_report.json"
    if mapping_report_path.exists():
        shutil.copy2(mapping_report_path, output_dir / "dataset_mapping_report.json")
    for split_name in ("train", "val", "test"):
        cached_indices = dataset.cache_dir / f"{split_name}_indices.npy"
        if cached_indices.exists():
            shutil.copy2(cached_indices, output_dir / cached_indices.name)

    best_metric_name = (
        "val_accuracy" if "val_accuracy" in history_payload else "val_loss"
    )
    best_values = np.asarray(history_payload[best_metric_name])
    best_offset = int(
        np.argmax(best_values)
        if best_metric_name != "val_loss"
        else np.argmin(best_values)
    )
    threshold_payload = {"threshold": threshold, **threshold_selection}
    timing = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_epoch_seconds": float(sum(timer.times)),
        "epoch_seconds": timer.times,
        "observed_peak_rss_bytes": int(
            max(initial_rss, process.memory_info().rss)
        ),
    }
    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8"
    )
    environment = _environment(tf)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    dataset_manifest = {
        "cache_dir": str(dataset.cache_dir.resolve()),
        "metadata": dataset.metadata,
        "cache_control_sha256": {
            path.name: _sha256(path)
            for path in (
                dataset.cache_dir / "metadata.json",
                dataset.cache_dir / "preprocessor.joblib",
            )
            if path.exists()
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(_json_ready(dataset_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "threshold.json").write_text(
        json.dumps(threshold_payload, indent=2), encoding="utf-8"
    )
    _write_run_contract(
        output_dir,
        dataset=dataset,
        task=task,
        model_name=model_name,
        swarm_name=swarm_name,
        params=params,
        protocol=protocol,
        modeling_mode=modeling_mode,
        seed=seed,
        deterministic_ops=deterministic_ops,
        threshold=threshold,
        sample_limits={
            "train": max_train_samples,
            "validation": max_val_samples,
            "test": max_test_samples,
        },
        run_mode=run_mode,
    )

    result = {
        "dataset": dataset.metadata["dataset"],
        "task": task,
        "model": model_name,
        "modeling_mode": modeling_mode,
        "protocol": protocol,
        "run_type": "fixed parameters; not an optimizer search",
        "run_mode": run_mode,
        "swarm_preset": swarm_name,
        "seed": seed,
        "parameters": params.to_dict(),
        "parameter_count": int(model.count_params()),
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape),
        "sample_counts": {
            "train": int(len(train_y)),
            "validation": int(len(val_y)),
            "test": int(len(test_y)),
        },
        "best_validation_epoch": best_offset + 1,
        "best_validation_metric": {
            "name": best_metric_name,
            "value": float(best_values[best_offset]),
        },
        "threshold": threshold_payload,
        "metrics": _json_ready(metrics),
        "paper_confusion_comparison": comparison,
        "timing": timing,
        "tensorflow_version": tf.__version__,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def validation_fitness(
    dataset: PreparedDataset,
    *,
    task: Task,
    model_name: ModelName,
    params: HyperParameters,
    seed: int,
    epoch_cap: int | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    fitness_name: FitnessName = "accuracy",
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    modeling_mode: ModelingMode = "feature_axis_replication",
) -> float:
    """Evaluate one candidate on validation only; test arrays are never accessed."""

    import tensorflow as tf

    set_reproducible_seed(seed)
    if epoch_cap is not None:
        params = replace(params, epochs=min(params.epochs, epoch_cap))
    train_x, train_y, _ = _select(
        dataset.train_x,
        dataset.labels("train", task),
        max_train_samples,
        seed,
    )
    val_x, val_y, _ = _select(
        dataset.val_x,
        dataset.labels("val", task),
        max_val_samples,
        seed + 1,
    )
    train_x = reshape_for_model(train_x, model_name, modeling_mode)
    val_x = reshape_for_model(val_x, model_name, modeling_mode)
    model = build_model(
        model_name,
        task,
        params,
        feature_count=(
            int(train_x.shape[2])
            if modeling_mode == "temporal_window"
            else len(dataset.feature_names)
        ),
        class_count=2 if task == "binary" else len(dataset.class_names),
        modeling_mode=modeling_mode,
        time_steps=(
            int(train_x.shape[1]) if modeling_mode == "temporal_window" else None
        ),
    )
    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=params.batch_size,
        epochs=params.epochs,
        shuffle=True,
        verbose=0,
    )
    probabilities = model.predict(val_x, batch_size=params.batch_size, verbose=0)
    prediction = (
        (probabilities.reshape(-1) >= 0.5).astype(np.uint8)
        if task == "binary"
        else np.argmax(probabilities, axis=1).astype(np.uint8)
    )
    if fitness_name == "accuracy":
        fitness = accuracy_score(val_y, prediction)
    elif fitness_name == "macro_f1":
        fitness = f1_score(val_y, prediction, average="macro", zero_division=0)
    elif fitness_name == "cost_sensitive":
        if task == "binary":
            false_positives = np.sum((val_y == 0) & (prediction == 1))
            false_negatives = np.sum((val_y == 1) & (prediction == 0))
            true_negatives = np.sum((val_y == 0) & (prediction == 0))
            true_positives = np.sum((val_y == 1) & (prediction == 1))
            false_alarm_rate = false_positives / max(
                1, false_positives + true_negatives
            )
            false_negative_rate = false_negatives / max(
                1, false_negatives + true_positives
            )
            cost = (
                false_positive_cost * false_alarm_rate
                + false_negative_cost * false_negative_rate
            ) / max(1e-12, false_positive_cost + false_negative_cost)
        else:
            cost = np.mean(val_y != prediction) * false_negative_cost
        fitness = -float(cost)
    else:
        raise ValueError(f"Unknown fitness: {fitness_name}")
    del model, train_x, train_y, val_x, val_y, probabilities, prediction
    tf.keras.backend.clear_session()
    gc.collect()
    return float(fitness)
