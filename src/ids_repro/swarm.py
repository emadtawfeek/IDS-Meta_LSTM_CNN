from __future__ import annotations

import json
import math
import time
import csv
import platform
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable, Literal

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
        return {
            "algorithm": self.algorithm,
            "best_position": self.best_position.tolist(),
            "best_parameters": self.best_parameters.to_dict(),
            "best_fitness": self.best_fitness,
            "convergence": self.convergence,
            "evaluations": self.evaluations,
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
    fitness: float,
    elapsed_seconds: float,
) -> dict:
    return {
        "evaluation": evaluation,
        "iteration": iteration,
        "member": member,
        "position": position.tolist(),
        "parameters": parameters.to_dict(),
        "fitness": fitness,
        "elapsed_seconds": elapsed_seconds,
    }


def particle_swarm_search(
    objective: Callable[[HyperParameters, int], float],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    inertia: float = 0.5,
    cognitive: float = 2.0,
    social: float = 2.0,
    progress: Callable[[int, float], None] | None = None,
) -> SearchResult:
    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    rng = np.random.RandomState(seed)
    dimensions = len(search_space(model_name))
    positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
    velocities = np.zeros_like(positions)
    personal_positions = positions.copy()
    personal_fitness = np.full(population_size, -np.inf)
    global_position = positions[0].copy()
    global_fitness = -np.inf
    convergence: list[float] = []
    evaluation_trace: list[dict] = []
    position_history: list[list[list[float]]] = []
    evaluations = 0

    for iteration in range(iterations):
        position_history.append(positions.tolist())
        for particle in range(population_size):
            position = positions[particle].copy()
            parameters = decode_position(position, model_name)
            started = time.perf_counter()
            fitness = float(objective(parameters, evaluations))
            elapsed = time.perf_counter() - started
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluations,
                    iteration=iteration + 1,
                    member=particle,
                    position=position,
                    parameters=parameters,
                    fitness=fitness,
                    elapsed_seconds=elapsed,
                )
            )
            evaluations += 1
            if fitness > personal_fitness[particle]:
                personal_fitness[particle] = fitness
                personal_positions[particle] = positions[particle].copy()
            if fitness > global_fitness:
                global_fitness = fitness
                global_position = positions[particle].copy()
        convergence.append(global_fitness)
        if progress:
            progress(iteration + 1, global_fitness)
        if iteration + 1 == iterations:
            continue
        r1 = rng.random_sample(size=positions.shape)
        r2 = rng.random_sample(size=positions.shape)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_positions - positions)
            + social * r2 * (global_position - positions)
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)

    return SearchResult(
        algorithm="pso",
        best_position=global_position,
        best_parameters=decode_position(global_position, model_name),
        best_fitness=float(global_fitness),
        convergence=convergence,
        evaluations=evaluations,
        evaluation_trace=evaluation_trace,
        position_history=position_history,
        seed=seed,
    )


def salp_swarm_search(
    objective: Callable[[HyperParameters, int], float],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    progress: Callable[[int, float], None] | None = None,
) -> SearchResult:
    """Canonical leader/follower SSA using the paper's stated population budget.

    The paper pseudocode prints c3 <= 0 although c3 is sampled in [0, 1]. The
    canonical and operational threshold c3 < 0.5 is used here.
    """

    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    rng = np.random.RandomState(seed)
    dimensions = len(search_space(model_name))
    positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
    food_position = positions[0].copy()
    food_fitness = -np.inf
    convergence: list[float] = []
    evaluation_trace: list[dict] = []
    position_history: list[list[list[float]]] = []
    evaluations = 0

    for iteration in range(iterations):
        position_history.append(positions.tolist())
        for salp in range(population_size):
            position = positions[salp].copy()
            parameters = decode_position(position, model_name)
            started = time.perf_counter()
            fitness = float(objective(parameters, evaluations))
            elapsed = time.perf_counter() - started
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluations,
                    iteration=iteration + 1,
                    member=salp,
                    position=position,
                    parameters=parameters,
                    fitness=fitness,
                    elapsed_seconds=elapsed,
                )
            )
            evaluations += 1
            if fitness > food_fitness:
                food_fitness = fitness
                food_position = positions[salp].copy()
        convergence.append(food_fitness)
        if progress:
            progress(iteration + 1, food_fitness)
        if iteration + 1 == iterations:
            continue

        c1 = 2.0 * math.exp(-((4.0 * (iteration + 1) / iterations) ** 2))
        updated = positions.copy()
        c2 = rng.random_sample(dimensions)
        c3 = rng.random_sample(dimensions)
        direction = np.where(c3 < 0.5, 1.0, -1.0)
        updated[0] = food_position + direction * c1 * c2
        for salp in range(1, population_size):
            updated[salp] = 0.5 * (positions[salp] + updated[salp - 1])
        positions = np.clip(updated, 0.0, 1.0)

    return SearchResult(
        algorithm="ssa",
        best_position=food_position,
        best_parameters=decode_position(food_position, model_name),
        best_fitness=float(food_fitness),
        convergence=convergence,
        evaluations=evaluations,
        evaluation_trace=evaluation_trace,
        position_history=position_history,
        seed=seed,
    )


def random_search(
    objective: Callable[[HyperParameters, int], float],
    *,
    model_name: ModelName,
    population_size: int = 10,
    iterations: int = 100,
    seed: int = 42,
    progress: Callable[[int, float], None] | None = None,
) -> SearchResult:
    """Budget-matched uniform random-search baseline in normalized coordinates."""

    if population_size < 1 or iterations < 1:
        raise ValueError("population_size and iterations must be positive")
    rng = np.random.RandomState(seed)
    dimensions = len(search_space(model_name))
    best_position = np.zeros(dimensions, dtype=np.float64)
    best_fitness = -np.inf
    convergence: list[float] = []
    evaluation_trace: list[dict] = []
    position_history: list[list[list[float]]] = []
    evaluations = 0
    for iteration in range(iterations):
        positions = rng.uniform(0.0, 1.0, size=(population_size, dimensions))
        position_history.append(positions.tolist())
        for member, position in enumerate(positions):
            parameters = decode_position(position, model_name)
            started = time.perf_counter()
            fitness = float(objective(parameters, evaluations))
            elapsed = time.perf_counter() - started
            evaluation_trace.append(
                _evaluation_record(
                    evaluation=evaluations,
                    iteration=iteration + 1,
                    member=member,
                    position=position,
                    parameters=parameters,
                    fitness=fitness,
                    elapsed_seconds=elapsed,
                )
            )
            evaluations += 1
            if fitness > best_fitness:
                best_fitness = fitness
                best_position = position.copy()
        convergence.append(float(best_fitness))
        if progress:
            progress(iteration + 1, float(best_fitness))
    return SearchResult(
        algorithm="random",
        best_position=best_position,
        best_parameters=decode_position(best_position, model_name),
        best_fitness=float(best_fitness),
        convergence=convergence,
        evaluations=evaluations,
        evaluation_trace=evaluation_trace,
        position_history=position_history,
        seed=seed,
    )
