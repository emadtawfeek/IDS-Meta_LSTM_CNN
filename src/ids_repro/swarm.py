from __future__ import annotations

import json
import math
import time
import csv
import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import yaml

from .config import HyperParameters, ModelName


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: Literal["choice", "integer", "float"]
    values: tuple[int, ...] | None = None
    lower: float | None = None
    upper: float | None = None

    def decode(self, normalized: float) -> int | float:
        value = float(np.clip(normalized, 0.0, 1.0))
        if self.kind == "choice":
            assert self.values is not None
            index = int(np.rint(value * (len(self.values) - 1)))
            return self.values[index]
        if self.lower is None or self.upper is None:
            raise ValueError(f"Bounds missing for {self.name}")
        decoded = self.lower + value * (self.upper - self.lower)
        if self.kind == "integer":
            return int(np.rint(decoded))
        return float(decoded)


CHOICES = (16, 32, 64, 128, 256)
KERNELS = (3, 5, 7, 9, 11)
BATCHES = (16, 32, 64, 128, 256)


def search_space(model_name: ModelName) -> tuple[ParameterSpec, ...]:
    common = (
        ParameterSpec("num_dense_layers", "integer", lower=1, upper=5),
        ParameterSpec("dense_units", "choice", values=CHOICES),
        ParameterSpec("dropout_rate", "float", lower=0.1, upper=0.5),
        ParameterSpec("learning_rate", "float", lower=1e-5, upper=1e-2),
        ParameterSpec("batch_size", "choice", values=BATCHES),
        ParameterSpec("epochs", "integer", lower=10, upper=100),
    )
    if model_name == "cnn":
        return (
            ParameterSpec("num_filters", "choice", values=CHOICES),
            ParameterSpec("kernel_size", "choice", values=KERNELS),
            ParameterSpec("pooling_size", "integer", lower=2, upper=6),
            *common,
        )
    if model_name == "lstm":
        return (*common, ParameterSpec("lstm_units", "choice", values=CHOICES))
    if model_name == "cnn-lstm":
        return (
            ParameterSpec("num_filters", "choice", values=CHOICES),
            ParameterSpec("kernel_size", "choice", values=KERNELS),
            ParameterSpec("pooling_size", "integer", lower=2, upper=6),
            *common,
            ParameterSpec("lstm_units", "choice", values=CHOICES),
        )
    raise ValueError(f"Unknown model: {model_name}")


def decode_position(position: np.ndarray, model_name: ModelName) -> HyperParameters:
    specs = search_space(model_name)
    if len(position) != len(specs):
        raise ValueError(f"Expected {len(specs)} dimensions, received {len(position)}")
    values = {spec.name: spec.decode(float(position[i])) for i, spec in enumerate(specs)}
    return HyperParameters(
        num_filters=int(values["num_filters"]) if "num_filters" in values else None,
        kernel_size=int(values["kernel_size"]) if "kernel_size" in values else None,
        pooling_size=int(values["pooling_size"]) if "pooling_size" in values else None,
        num_dense_layers=int(values["num_dense_layers"]),
        dense_units=int(values["dense_units"]),
        dropout_rate=float(values["dropout_rate"]),
        learning_rate=float(values["learning_rate"]),
        batch_size=int(values["batch_size"]),
        epochs=int(values["epochs"]),
        lstm_units=int(values["lstm_units"]) if "lstm_units" in values else None,
    )


class SearchPaused(RuntimeError):
    """Raised only by the test/debug evaluation limit after a safe checkpoint."""


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _rng_state(rng: np.random.RandomState) -> dict:
    name, keys, position, has_gauss, cached_gaussian = rng.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _restore_rng(payload: dict) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.set_state(
        (
            payload["name"],
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    )
    return rng


class CachedObjective:
    """Persistent decoded-candidate cache with failure isolation and accounting."""

    def __init__(self, objective: Callable, key_context: dict):
        self.objective = objective
        self.key_context = _json_ready(key_context)
        self.cache: dict[str, dict] = {}
        self.seen_keys: set[str] = set()

    def _key(self, parameters: HyperParameters) -> str:
        payload = {
            "parameters": parameters.to_dict(),
            **self.key_context,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def __call__(self, parameters: HyperParameters, evaluation: int) -> dict:
        key = self._key(parameters)
        self.seen_keys.add(key)
        if key in self.cache:
            return {
                **self.cache[key],
                "candidate_cache_key": key,
                "cache_hit": True,
                "fit_attempted": False,
                "model_fit": False,
            }
        try:
            value = self.objective(parameters, evaluation)
            outcome = dict(value) if isinstance(value, dict) else {"fitness": value}
            fitness = float(outcome["fitness"])
            if not math.isfinite(fitness):
                raise FloatingPointError(f"non-finite objective value: {fitness}")
            stored = {
                **_json_ready(outcome),
                "fitness": fitness,
                "cache_hit": False,
                "fit_attempted": True,
                "model_fit": True,
                "status": "completed",
            }
            self.cache[key] = stored
            return {**stored, "candidate_cache_key": key}
        except Exception as exc:
            return {
                "fitness": None,
                "candidate_cache_key": key,
                "cache_hit": False,
                "fit_attempted": True,
                "model_fit": False,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def state_dict(self) -> dict:
        return {
            "key_context": self.key_context,
            "cache": self.cache,
            "seen_keys": sorted(self.seen_keys),
        }

    def load_state_dict(self, payload: dict) -> None:
        if payload.get("key_context") != self.key_context:
            raise ValueError("Candidate-cache identity differs from checkpoint")
        self.cache = dict(payload.get("cache", {}))
        self.seen_keys = set(payload.get("seen_keys", []))


@dataclass
class SearchResult:
    algorithm: str
    best_position: np.ndarray
    best_parameters: HyperParameters
    best_fitness: float
    convergence: list[float]
    evaluations: int
    evaluation_trace: list[dict]
    position_history: list[list[list[float]]]
    seed: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        counters = _trace_counters(self.evaluation_trace)
        return {
            "algorithm": self.algorithm,
            "best_position": self.best_position.tolist(),
            "best_parameters": self.best_parameters.to_dict(),
            "best_fitness": self.best_fitness,
            "convergence": self.convergence,
            "evaluations": self.evaluations,
            **counters,
            "evaluation_trace": self.evaluation_trace,
            "position_history": self.position_history,
            "seed": self.seed,
            "metadata": self.metadata,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_artifacts(self, output_dir: Path | str) -> None:
        """Write human- and machine-readable search evidence beside the JSON result."""

        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite search artifacts in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "best_parameters.json").write_text(
            json.dumps(self.best_parameters.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fieldnames = [
            "evaluation",
            "iteration",
            "member",
            "fitness",
            "elapsed_seconds",
            "position_json",
            "parameters_json",
            "seed",
            "candidate_seed",
            "status",
            "cache_hit",
            "fit_attempted",
            "model_fit",
            "candidate_cache_key",
            "best_epoch",
            "fitness_history_json",
            "error_type",
            "error",
        ]
        with (output_dir / "optimization_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.evaluation_trace:
                writer.writerow(
                    {
                        "evaluation": record["evaluation"],
                        "iteration": record["iteration"],
                        "member": record["member"],
                        "fitness": record["fitness"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "position_json": json.dumps(record["position"]),
                        "parameters_json": json.dumps(
                            record["parameters"], sort_keys=True
                        ),
                        "seed": self.seed,
                        "candidate_seed": record.get("candidate_seed", self.seed),
                        "status": record.get("status", "completed"),
                        "cache_hit": record.get("cache_hit", False),
                        "fit_attempted": record.get("fit_attempted", True),
                        "model_fit": record.get("model_fit", True),
                        "candidate_cache_key": record.get("candidate_cache_key"),
                        "best_epoch": record.get("best_epoch"),
                        "fitness_history_json": json.dumps(
                            record.get("fitness_history", [])
                        ),
                        "error_type": record.get("error_type"),
                        "error": record.get("error"),
                    }
                )
        (output_dir / "convergence.json").write_text(
            json.dumps(self.convergence, indent=2), encoding="utf-8"
        )
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(range(1, len(self.convergence) + 1), self.convergence, marker="o")
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Best validation fitness")
            axis.set_title(f"{self.algorithm.upper()} convergence")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(output_dir / "convergence_curve.png", dpi=160)
            plt.close(figure)
        finally:
            pass
        (output_dir / "environment.json").write_text(
            json.dumps(
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "search_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(self.metadata, sort_keys=False), encoding="utf-8"
        )


def _evaluation_record(
    *,
    evaluation: int,
    iteration: int,
    member: int,
    position: np.ndarray,
    parameters: HyperParameters,
    outcome: dict,
    elapsed_seconds: float,
) -> dict:
    return {
        "evaluation": evaluation,
        "iteration": iteration,
        "member": member,
        "position": position.tolist(),
        "parameters": parameters.to_dict(),
        "fitness": outcome.get("fitness"),
        "elapsed_seconds": elapsed_seconds,
        **{key: value for key, value in outcome.items() if key != "fitness"},
    }


def _trace_counters(trace: list[dict]) -> dict:
    return {
        "proposal_count": len(trace),
        "unique_configuration_count": len(
            {record.get("candidate_cache_key") for record in trace if record.get("candidate_cache_key")}
        ),
        "cache_hits": sum(bool(record.get("cache_hit")) for record in trace),
        "nn_fit_attempts": sum(bool(record.get("fit_attempted", True)) for record in trace),
        "completed_nn_fits": sum(bool(record.get("model_fit", True)) for record in trace),
        "failed_candidates": sum(record.get("status") == "failed" for record in trace),
    }


def _objective_state(objective: Callable) -> dict | None:
    method = getattr(objective, "state_dict", None)
    return _json_ready(method()) if method else None


def _load_objective_state(objective: Callable, payload: dict | None) -> None:
    if payload is not None:
        method = getattr(objective, "load_state_dict", None)
        if method is None:
            raise ValueError("Checkpoint contains objective state but objective cannot restore it")
        method(payload)


def _normalize_outcome(value: Any) -> tuple[dict, float]:
    outcome = dict(value) if isinstance(value, dict) else {"fitness": value}
    raw_fitness = outcome.get("fitness")
    try:
        fitness = float(raw_fitness)
    except (TypeError, ValueError):
        fitness = -np.inf
    if not math.isfinite(fitness):
        outcome["fitness"] = None
        outcome.setdefault("status", "failed")
        outcome.setdefault("error_type", "InvalidFitness")
        outcome.setdefault("error", f"Rejected non-finite fitness {raw_fitness!r}")
        return _json_ready(outcome), -np.inf
    outcome["fitness"] = fitness
    outcome.setdefault("status", "completed")
    outcome.setdefault("cache_hit", False)
    outcome.setdefault("fit_attempted", True)
    outcome.setdefault("model_fit", True)
    return _json_ready(outcome), fitness


def _evaluate_candidate(
    objective: Callable,
    parameters: HyperParameters,
    evaluation: int,
) -> tuple[dict, float, float]:
    started = time.perf_counter()
    try:
        value = objective(parameters, evaluation)
    except Exception as exc:
        value = {
            "fitness": None,
            "status": "failed",
            "cache_hit": False,
            "fit_attempted": True,
            "model_fit": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - started
    outcome, fitness = _normalize_outcome(value)
    return outcome, fitness, elapsed


def _checkpoint_identity(
    algorithm: str,
    model_name: str,
    population_size: int,
    iterations: int,
    seed: int,
    run_identity: dict | None,
) -> dict:
    return {
        "algorithm": algorithm,
        "model": model_name,
        "population_size": population_size,
        "iterations": iterations,
        "seed": seed,
        "run_identity": _json_ready(run_identity or {}),
    }


def _read_checkpoint(path: Path, identity: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        raise ValueError("Checkpoint configuration/dataset identity does not match this run")
    return payload


def _write_checkpoint(
    path: Path | None,
    *,
    identity: dict,
    state: dict,
    objective: Callable,
    complete: bool = False,
) -> None:
    if path is not None:
        _atomic_json(
            path,
            {
                "version": 1,
                "identity": identity,
                "complete": complete,
                "state": state,
                "objective_state": _objective_state(objective),
            },
        )


def _finish_result(
    algorithm: str,
    model_name: ModelName,
    best_position: np.ndarray,
    best_fitness: float,
    convergence: list[float],
    evaluation_trace: list[dict],
    position_history: list,
    seed: int,
) -> SearchResult:
    if not math.isfinite(best_fitness):
        raise RuntimeError("Search completed without a finite candidate fitness")
    return SearchResult(
        algorithm=algorithm,
        best_position=best_position,
        best_parameters=decode_position(best_position, model_name),
        best_fitness=float(best_fitness),
        convergence=[float(value) for value in convergence],
        evaluations=len(evaluation_trace),
        evaluation_trace=evaluation_trace,
        position_history=position_history,
        seed=seed,
    )


def particle_swarm_search(
    objective: Callable[[HyperParameters, int], float | dict],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    inertia: float = 0.5,
    cognitive: float = 2.0,
    social: float = 2.0,
    velocity_clip: float | None = None,
    progress: Callable[[int, float], None] | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
    run_identity: dict | None = None,
    max_evaluations: int | None = None,
) -> SearchResult:
    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    if velocity_clip is not None and (
        not math.isfinite(velocity_clip) or velocity_clip <= 0
    ):
        raise ValueError("velocity_clip must be finite and positive")
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    identity = _checkpoint_identity(
        "pso",
        model_name,
        population_size,
        iterations,
        seed,
        {
            **(run_identity or {}),
            "pso_dynamics": {
                "inertia": inertia,
                "cognitive": cognitive,
                "social": social,
                "velocity_clip": velocity_clip,
            },
        },
    )
    dimensions = len(search_space(model_name))
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError("--resume requires an existing checkpoint")
        payload = _read_checkpoint(checkpoint, identity)
        state = payload["state"]
        _load_objective_state(objective, payload.get("objective_state"))
        rng = _restore_rng(state["rng_state"])
        positions = np.asarray(state["positions"], dtype=np.float64)
        velocities = np.asarray(state["velocities"], dtype=np.float64)
        personal_positions = np.asarray(state["personal_positions"], dtype=np.float64)
        personal_fitness = np.asarray(state["personal_fitness"], dtype=np.float64)
        global_position = np.asarray(state["global_position"], dtype=np.float64)
        global_fitness = float(state["global_fitness"])
        convergence = list(state["convergence"])
        evaluation_trace = list(state["evaluation_trace"])
        position_history = list(state["position_history"])
        iteration = int(state["iteration"])
        member = int(state["member"])
    else:
        if checkpoint is not None and checkpoint.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint}; pass resume=True to continue"
            )
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
        velocities = np.zeros_like(positions)
        personal_positions = positions.copy()
        personal_fitness = np.full(population_size, -np.inf)
        global_position = positions[0].copy()
        global_fitness = -np.inf
        convergence = []
        evaluation_trace = []
        position_history = []
        iteration = 0
        member = 0

    def state_dict() -> dict:
        return {
            "iteration": iteration,
            "member": member,
            "positions": positions,
            "velocities": velocities,
            "personal_positions": personal_positions,
            "personal_fitness": personal_fitness,
            "global_position": global_position,
            "global_fitness": global_fitness,
            "convergence": convergence,
            "evaluation_trace": evaluation_trace,
            "position_history": position_history,
            "rng_state": _rng_state(rng),
        }

    while iteration < iterations:
        if member == 0 and len(position_history) == iteration:
            position_history.append(positions.tolist())
        while member < population_size:
            particle = member
            position = positions[particle].copy()
            parameters = decode_position(position, model_name)
            evaluation = len(evaluation_trace)
            outcome, fitness, elapsed = _evaluate_candidate(
                objective, parameters, evaluation
            )
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluation,
                    iteration=iteration + 1,
                    member=particle,
                    position=position,
                    parameters=parameters,
                    outcome=outcome,
                    elapsed_seconds=elapsed,
                )
            )
            if fitness > personal_fitness[particle]:
                personal_fitness[particle] = fitness
                personal_positions[particle] = positions[particle].copy()
            if fitness > global_fitness:
                global_fitness = fitness
                global_position = positions[particle].copy()
            member += 1
            _write_checkpoint(
                checkpoint, identity=identity, state=state_dict(), objective=objective
            )
            if max_evaluations is not None and len(evaluation_trace) >= max_evaluations:
                raise SearchPaused("Search paused after a safely checkpointed evaluation")
        convergence.append(global_fitness)
        if progress:
            progress(iteration + 1, global_fitness)
        if iteration + 1 < iterations:
            r1 = rng.random_sample(size=positions.shape)
            r2 = rng.random_sample(size=positions.shape)
            velocities = (
                inertia * velocities
                + cognitive * r1 * (personal_positions - positions)
                + social * r2 * (global_position - positions)
            )
            if velocity_clip is not None:
                velocities = np.clip(velocities, -velocity_clip, velocity_clip)
            positions = np.clip(positions + velocities, 0.0, 1.0)
        iteration += 1
        member = 0
        _write_checkpoint(
            checkpoint,
            identity=identity,
            state=state_dict(),
            objective=objective,
            complete=iteration == iterations,
        )

    return _finish_result(
        "pso", model_name, global_position, global_fitness, convergence,
        evaluation_trace, position_history, seed
    )


def salp_swarm_search(
    objective: Callable[[HyperParameters, int], float | dict],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    progress: Callable[[int, float], None] | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
    run_identity: dict | None = None,
    max_evaluations: int | None = None,
) -> SearchResult:
    """Canonical leader/follower SSA using the paper's stated population budget.

    The paper pseudocode prints c3 <= 0 although c3 is sampled in [0, 1]. The
    canonical and operational threshold c3 < 0.5 is used here.
    """

    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    identity = _checkpoint_identity(
        "ssa", model_name, population_size, iterations, seed, run_identity
    )
    dimensions = len(search_space(model_name))
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError("--resume requires an existing checkpoint")
        payload = _read_checkpoint(checkpoint, identity)
        state = payload["state"]
        _load_objective_state(objective, payload.get("objective_state"))
        rng = _restore_rng(state["rng_state"])
        positions = np.asarray(state["positions"], dtype=np.float64)
        food_position = np.asarray(state["food_position"], dtype=np.float64)
        food_fitness = float(state["food_fitness"])
        convergence = list(state["convergence"])
        evaluation_trace = list(state["evaluation_trace"])
        position_history = list(state["position_history"])
        iteration = int(state["iteration"])
        member = int(state["member"])
    else:
        if checkpoint is not None and checkpoint.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint}; pass resume=True to continue"
            )
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
        food_position = positions[0].copy()
        food_fitness = -np.inf
        convergence = []
        evaluation_trace = []
        position_history = []
        iteration = 0
        member = 0

    def state_dict() -> dict:
        return {
            "iteration": iteration,
            "member": member,
            "positions": positions,
            "food_position": food_position,
            "food_fitness": food_fitness,
            "convergence": convergence,
            "evaluation_trace": evaluation_trace,
            "position_history": position_history,
            "rng_state": _rng_state(rng),
        }

    while iteration < iterations:
        if member == 0 and len(position_history) == iteration:
            position_history.append(positions.tolist())
        while member < population_size:
            salp = member
            position = positions[salp].copy()
            parameters = decode_position(position, model_name)
            evaluation = len(evaluation_trace)
            outcome, fitness, elapsed = _evaluate_candidate(
                objective, parameters, evaluation
            )
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluation,
                    iteration=iteration + 1,
                    member=salp,
                    position=position,
                    parameters=parameters,
                    outcome=outcome,
                    elapsed_seconds=elapsed,
                )
            )
            if fitness > food_fitness:
                food_fitness = fitness
                food_position = positions[salp].copy()
            member += 1
            _write_checkpoint(
                checkpoint, identity=identity, state=state_dict(), objective=objective
            )
            if max_evaluations is not None and len(evaluation_trace) >= max_evaluations:
                raise SearchPaused("Search paused after a safely checkpointed evaluation")
        convergence.append(food_fitness)
        if progress:
            progress(iteration + 1, food_fitness)
        if iteration + 1 < iterations:
            c1 = 2.0 * math.exp(-((4.0 * (iteration + 1) / iterations) ** 2))
            updated = positions.copy()
            c2 = rng.random_sample(dimensions)
            c3 = rng.random_sample(dimensions)
            direction = np.where(c3 < 0.5, 1.0, -1.0)
            updated[0] = food_position + direction * c1 * c2
            for salp in range(1, population_size):
                updated[salp] = 0.5 * (positions[salp] + updated[salp - 1])
            positions = np.clip(updated, 0.0, 1.0)
        iteration += 1
        member = 0
        _write_checkpoint(
            checkpoint,
            identity=identity,
            state=state_dict(),
            objective=objective,
            complete=iteration == iterations,
        )

    return _finish_result(
        "ssa", model_name, food_position, food_fitness, convergence,
        evaluation_trace, position_history, seed
    )


def random_search(
    objective: Callable[[HyperParameters, int], float | dict],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    progress: Callable[[int, float], None] | None = None,
    checkpoint_path: Path | str | None = None,
    resume: bool = False,
    run_identity: dict | None = None,
    max_evaluations: int | None = None,
) -> SearchResult:
    """Budget-matched uniform random-search baseline in normalized coordinates."""

    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    identity = _checkpoint_identity(
        "random", model_name, population_size, iterations, seed, run_identity
    )
    dimensions = len(search_space(model_name))
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError("--resume requires an existing checkpoint")
        payload = _read_checkpoint(checkpoint, identity)
        state = payload["state"]
        _load_objective_state(objective, payload.get("objective_state"))
        rng = _restore_rng(state["rng_state"])
        best_position = np.asarray(state["best_position"], dtype=np.float64)
        best_fitness = float(state["best_fitness"])
        convergence = list(state["convergence"])
        evaluation_trace = list(state["evaluation_trace"])
        position_history = list(state["position_history"])
        positions = (
            None
            if state.get("positions") is None
            else np.asarray(state["positions"], dtype=np.float64)
        )
        iteration = int(state["iteration"])
        member = int(state["member"])
    else:
        if checkpoint is not None and checkpoint.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint}; pass resume=True to continue"
            )
        rng = np.random.RandomState(seed)
        best_position = np.zeros(dimensions, dtype=np.float64)
        best_fitness = -np.inf
        convergence = []
        evaluation_trace = []
        position_history = []
        positions = None
        iteration = 0
        member = 0

    def state_dict() -> dict:
        return {
            "iteration": iteration,
            "member": member,
            "positions": positions,
            "best_position": best_position,
            "best_fitness": best_fitness,
            "convergence": convergence,
            "evaluation_trace": evaluation_trace,
            "position_history": position_history,
            "rng_state": _rng_state(rng),
        }

    while iteration < iterations:
        if positions is None:
            positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
            position_history.append(positions.tolist())
        while member < population_size:
            position = positions[member]
            parameters = decode_position(position, model_name)
            evaluation = len(evaluation_trace)
            outcome, fitness, elapsed = _evaluate_candidate(
                objective, parameters, evaluation
            )
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluation,
                    iteration=iteration + 1,
                    member=member,
                    position=position,
                    parameters=parameters,
                    outcome=outcome,
                    elapsed_seconds=elapsed,
                )
            )
            if fitness > best_fitness:
                best_fitness = fitness
                best_position = position.copy()
            member += 1
            _write_checkpoint(
                checkpoint, identity=identity, state=state_dict(), objective=objective
            )
            if max_evaluations is not None and len(evaluation_trace) >= max_evaluations:
                raise SearchPaused("Search paused after a safely checkpointed evaluation")
        convergence.append(float(best_fitness))
        if progress:
            progress(iteration + 1, float(best_fitness))
        iteration += 1
        member = 0
        positions = None
        _write_checkpoint(
            checkpoint,
            identity=identity,
            state=state_dict(),
            objective=objective,
            complete=iteration == iterations,
        )
    return _finish_result(
        "random", model_name, best_position, best_fitness, convergence,
        evaluation_trace, position_history, seed
    )
