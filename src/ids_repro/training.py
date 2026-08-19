from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

from .config import (
    PAPER_BINARY_CONFUSIONS,
    PAPER_CIC_MULTICLASS_NAMES,
    PAPER_MULTICLASS_CONFUSIONS,
    FitnessName,
    HyperParameters,
    ModelName,
    ModelingMode,
    ProtocolName,
    SwarmName,
    Task,
    get_paper_preset,
)
from .data import (
    PreparedDataset,
    cache_identity,
    deterministic_stratified_subset,
    prepared_split_checksum,
    reshape_for_model,
    subset_manifest,
)
from .metrics import evaluate_predictions, matrix_distance
from .models import build_model
from .selection import (
    cost_equation,
    validation_selection_callback,
    validate_fitness_configuration,
)


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
    indices = deterministic_stratified_subset(labels, maximum, seed)
    return features[indices], labels[indices], indices


class _MemorySampler:
    """Periodically sample process RSS while fit and prediction are running."""

    def __init__(self, process, interval_seconds: float = 0.05):
        self.process = process
        self.interval_seconds = interval_seconds
        self.peak = int(process.memory_info().rss)
        self.samples = 1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            self.samples += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self.samples += 1


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
    selection_provenance: dict,
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
        "selection_source": selection_provenance["selection_source"],
        "selection_provenance": selection_provenance,
        "run_mode": run_mode,
        "seed": seed,
        "deterministic_ops": deterministic_ops,
        "threshold": threshold,
        "sample_limits": sample_limits,
        "hyperparameters": params.to_dict(),
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
    )


def _paper_comparison_eligibility(
    dataset: PreparedDataset,
    *,
    task: Task,
    model_name: ModelName,
    params: HyperParameters,
    protocol: ProtocolName,
    selection_provenance: dict,
    max_test_samples: int | None,
) -> tuple[bool, list[str], str | None]:
    reasons: list[str] = []
    algorithm = selection_provenance.get("algorithm")
    split_checksum = prepared_split_checksum(dataset)
    metadata_checksum = dataset.metadata.get("split", {}).get("checksum_sha256")
    if dataset.metadata.get("dataset") != "cicids2017":
        reasons.append("paper comparison is CIC-IDS2017 only")
    if protocol != "paper_replication":
        reasons.append("protocol is not paper_replication")
    if selection_provenance.get("selection_source") != "paper_preset":
        reasons.append("selection_source is not paper_preset")
    if algorithm not in {"pso", "ssa"}:
        reasons.append("paper preset algorithm is not pso or ssa")
    if max_test_samples is not None:
        reasons.append("test set was subsampled")
    if not dataset.metadata.get("paper_split_match", False):
        reasons.append("cached class counts do not match the paper split")
    if tuple(dataset.class_names) != tuple(PAPER_CIC_MULTICLASS_NAMES):
        reasons.append("class order does not match the paper")
    if split_checksum is None or metadata_checksum != split_checksum:
        reasons.append("saved split checksum is missing or does not match persisted indices")
    if algorithm in {"pso", "ssa"} and params != get_paper_preset(task, model_name, algorithm):
        reasons.append("parameters do not exactly match the selected paper preset")
    return not reasons, reasons, split_checksum


def _validate_selection_provenance(
    provenance: dict,
    *,
    dataset: PreparedDataset,
    task: Task,
    model_name: ModelName,
    params: HyperParameters,
) -> None:
    source = provenance.get("selection_source")
    expected_algorithms = {
        "pso_search": "pso",
        "ssa_search": "ssa",
        "random_search": "random",
    }
    allowed = {
        "paper_preset",
        "pso_search",
        "ssa_search",
        "random_search",
        "manual",
        "transferred_cic_preset",
    }
    if source not in allowed:
        raise ValueError(f"Invalid or missing selection_source: {source!r}")
    expected_algorithm = expected_algorithms.get(source)
    if expected_algorithm and provenance.get("algorithm") != expected_algorithm:
        raise ValueError(
            f"selection_source={source} requires algorithm={expected_algorithm}"
        )
    expected_fields = {
        "dataset": dataset.metadata.get("dataset"),
        "task": task,
        "model": model_name,
    }
    for key, expected in expected_fields.items():
        actual = provenance.get(key)
        if actual != expected:
            raise ValueError(
                f"Selection provenance {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    decoded = provenance.get("decoded_parameters")
    if decoded is not None and decoded != params.to_dict():
        raise ValueError("Selection provenance decoded parameters do not match params")


def train_and_evaluate(
    dataset: PreparedDataset,
    *,
    task: Task,
    model_name: ModelName,
    swarm_name: SwarmName | None = None,
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
    selection_provenance: dict | None = None,
    selection_fitness: FitnessName = "macro_f1",
    selection_patience: int = 10,
    selection_min_delta: float = 0.0,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> dict:
    import psutil
    import tensorflow as tf

    dataset.assert_integrity()
    validate_fitness_configuration(
        task, selection_fitness, false_positive_cost, false_negative_cost
    )
    if selection_patience < 0:
        raise ValueError("selection_patience cannot be negative")
    if not math.isfinite(selection_min_delta) or selection_min_delta < 0:
        raise ValueError("selection_min_delta must be finite and non-negative")
    if selection_provenance is None:
        if swarm_name is None:
            raise ValueError("selection_provenance is required")
        selection_provenance = {
            "selection_source": "paper_preset",
            "algorithm": swarm_name,
            "seed": seed,
            "dataset": dataset.metadata.get("dataset"),
            "task": task,
            "model": model_name,
            "cache_identity": cache_identity(dataset),
            "decoded_parameters": params.to_dict(),
            "legacy_api_inference": True,
        }
    _validate_selection_provenance(
        selection_provenance,
        dataset=dataset,
        task=task,
        model_name=model_name,
        params=params,
    )
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
    selection_provenance = {
        **selection_provenance,
        "execution_parameters": params.to_dict(),
        "epochs_override": epochs_override,
    }

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
    selector = validation_selection_callback(
        val_x,
        val_y,
        task=task,
        fitness_name=selection_fitness,
        batch_size=params.batch_size,
        patience=selection_patience,
        min_delta=selection_min_delta,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        restore_best_weights=requested_rigorous,
        early_stopping=requested_rigorous,
    )
    callbacks: list = [selector, timer]
    process = psutil.Process()
    with _MemorySampler(process) as memory_sampler:
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

    with _MemorySampler(process) as prediction_memory_sampler:
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

    paper_eligible, paper_reasons, split_checksum = _paper_comparison_eligibility(
        dataset,
        task=task,
        model_name=model_name,
        params=params,
        protocol=protocol,
        selection_provenance=selection_provenance,
        max_test_samples=max_test_samples,
    )
    comparison = None
    paper_algorithm = selection_provenance.get("algorithm")
    if paper_eligible:
        expected = np.asarray(
            PAPER_BINARY_CONFUSIONS[paper_algorithm]
            if task == "binary"
            else PAPER_MULTICLASS_CONFUSIONS[paper_algorithm],
            dtype=np.int64,
        )
        if matrix.shape != expected.shape or len(test_y) != int(expected.sum()):
            raise AssertionError(
                "Paper-comparison eligibility passed but test matrix shape/count differs"
            )
        comparison = matrix_distance(matrix, expected)

    latency_batch_size = min(len(test_x), max(1, params.batch_size))
    latency_batch = test_x[:latency_batch_size]
    model.predict_on_batch(latency_batch)  # explicit warm-up, excluded from timings
    latency_samples = []
    for _ in range(5):
        latency_started = time.perf_counter()
        model.predict_on_batch(latency_batch)
        latency_samples.append(time.perf_counter() - latency_started)

    model_path = output_dir / "model.keras"
    model.save(model_path)
    serialized_model_size = model_path.stat().st_size
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
    selection_outcome = selector.outcome()
    (output_dir / "selection_history.json").write_text(
        json.dumps(
            {
                "fitness_name": selection_fitness,
                "best_epoch": selection_outcome.best_epoch,
                "best_fitness": selection_outcome.best_score,
                "patience": selection_patience,
                "min_delta": selection_min_delta,
                "early_stopping": requested_rigorous,
                "best_weights_restored": requested_rigorous,
                "history": selection_outcome.history,
                "cost": (
                    cost_equation(false_positive_cost, false_negative_cost)
                    if selection_fitness == "cost_sensitive"
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
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
    subset_payload = {
        "selection_policy": "deterministic_stratified_by_split_labels",
        "dataset_split_checksum_sha256": split_checksum,
        "seed_policy": {"train": seed, "validation": seed + 1, "test": seed + 2},
        "train": subset_manifest(train_subset, dataset.labels("train", task)),
        "validation": subset_manifest(val_subset, dataset.labels("val", task)),
        "test": subset_manifest(test_subset, dataset.labels("test", task)),
    }
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(subset_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "selection_provenance.json").write_text(
        json.dumps(_json_ready(selection_provenance), indent=2, sort_keys=True),
        encoding="utf-8",
    )
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

    threshold_payload = {"threshold": threshold, **threshold_selection}
    latency_median = float(np.median(latency_samples))
    timing = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "inference_seconds_per_sample": predict_seconds / max(1, len(test_y)),
        "inference_samples_per_second": len(test_y) / max(predict_seconds, 1e-12),
        "warmed_batch_latency": {
            "batch_size": latency_batch_size,
            "warmup_batches_excluded": 1,
            "timed_batches": len(latency_samples),
            "sample_seconds": latency_samples,
            "median_batch_seconds": latency_median,
            "median_seconds_per_sample": latency_median / latency_batch_size,
        },
        "total_epoch_seconds": float(sum(timer.times)),
        "epoch_seconds": timer.times,
        "peak_rss_bytes_periodic": int(
            max(memory_sampler.peak, prediction_memory_sampler.peak)
        ),
        "rss_sampling_interval_seconds": memory_sampler.interval_seconds,
        "rss_sample_count": memory_sampler.samples + prediction_memory_sampler.samples,
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
        "cache_identity": cache_identity(dataset),
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
        selection_provenance=selection_provenance,
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
        "selection_source": selection_provenance["selection_source"],
        "selection_provenance": selection_provenance,
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
        "serialized_model_size_bytes": int(serialized_model_size),
        "best_validation_epoch": selection_outcome.best_epoch,
        "best_validation_metric": {
            "name": selection_fitness,
            "value": selection_outcome.best_score,
            "selection_rule": (
                "maximum validation fitness; restored best epoch weights"
                if requested_rigorous
                else "maximum validation fitness recorded; paper-protocol endpoint weights retained"
            ),
            "best_weights_restored": requested_rigorous,
            "patience": selection_patience,
            "min_delta": selection_min_delta,
        },
        "selection_fitness_definition": (
            cost_equation(false_positive_cost, false_negative_cost)
            if selection_fitness == "cost_sensitive"
            else {"objective": f"maximize validation {selection_fitness}"}
        ),
        "threshold": threshold_payload,
        "metrics": _json_ready(metrics),
        "paper_confusion_comparison": comparison,
        "paper_comparison_eligibility": {
            "eligible": paper_eligible,
            "reasons": paper_reasons,
            "split_checksum_sha256": split_checksum,
        },
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
    patience: int = 10,
    min_delta: float = 0.0,
    return_details: bool = False,
) -> float | dict:
    """Evaluate one candidate on validation only; test arrays are never accessed."""

    import tensorflow as tf

    validate_fitness_configuration(
        task, fitness_name, false_positive_cost, false_negative_cost
    )
    model = None
    try:
        set_reproducible_seed(seed)
        if epoch_cap is not None:
            params = replace(params, epochs=min(params.epochs, epoch_cap))
        train_labels = dataset.labels("train", task)
        validation_labels = dataset.labels("val", task)
        train_x, train_y, train_indices = _select(
            dataset.train_x, train_labels, max_train_samples, seed
        )
        val_x, val_y, validation_indices = _select(
            dataset.val_x, validation_labels, max_val_samples, seed + 1
        )
        train_manifest = subset_manifest(train_indices, train_labels)
        validation_manifest = subset_manifest(validation_indices, validation_labels)
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
        selector = validation_selection_callback(
            val_x,
            val_y,
            task=task,
            fitness_name=fitness_name,
            batch_size=params.batch_size,
            patience=patience,
            min_delta=min_delta,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            batch_size=params.batch_size,
            epochs=params.epochs,
            shuffle=True,
            callbacks=[selector],
            verbose=0,
        )
        outcome = selector.outcome()
        details = {
            "fitness": outcome.best_score,
            "fitness_name": fitness_name,
            "best_epoch": outcome.best_epoch,
            "epochs_completed": outcome.epochs_completed,
            "fitness_history": outcome.history,
            "selection_rule": "maximum per-epoch validation fitness; best weights restored",
            "patience": patience,
            "min_delta": min_delta,
            "train_subset": train_manifest,
            "validation_subset": validation_manifest,
            "cost": (
                cost_equation(false_positive_cost, false_negative_cost)
                if fitness_name == "cost_sensitive"
                else None
            ),
        }
        return details if return_details else float(outcome.best_score)
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()
