from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Task = Literal["binary", "multiclass"]
ModelName = Literal["cnn", "lstm", "cnn-lstm"]
SwarmName = Literal["pso", "ssa"]
ProtocolName = Literal["paper_replication", "rigorous_evaluation", "rigorous"]
ModelingMode = Literal["feature_axis_replication", "temporal_window"]
FitnessName = Literal["accuracy", "macro_f1", "cost_sensitive"]


@dataclass(frozen=True)
class HyperParameters:
    num_filters: int | None
    kernel_size: int | None
    pooling_size: int | None
    num_dense_layers: int
    dense_units: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    lstm_units: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


# Tables 6 and 7 of Karahan et al. (2025). Values are transcribed exactly.
PAPER_PRESETS: dict[tuple[Task, ModelName, SwarmName], HyperParameters] = {
    ("binary", "cnn", "pso"): HyperParameters(256, 11, 4, 2, 256, 0.1, 0.00241, 256, 89, None),
    ("binary", "cnn", "ssa"): HyperParameters(64, 7, 5, 1, 128, 0.36098, 0.00047, 256, 84, None),
    ("binary", "lstm", "pso"): HyperParameters(None, None, None, 1, 256, 0.45235, 0.00157, 256, 82, 256),
    ("binary", "lstm", "ssa"): HyperParameters(None, None, None, 1, 16, 0.49771, 0.00608, 64, 92, 256),
    ("binary", "cnn-lstm", "pso"): HyperParameters(256, 11, 3, 3, 256, 0.1, 0.00131, 256, 98, 256),
    ("binary", "cnn-lstm", "ssa"): HyperParameters(256, 9, 5, 2, 128, 0.213821, 0.001621, 256, 79, 128),
    ("multiclass", "cnn", "pso"): HyperParameters(256, 11, 4, 5, 256, 0.278, 0.00036, 256, 39, None),
    ("multiclass", "cnn", "ssa"): HyperParameters(128, 9, 5, 3, 256, 0.228, 0.00027, 256, 60, None),
    ("multiclass", "lstm", "pso"): HyperParameters(None, None, None, 2, 64, 0.1658, 0.0026, 256, 87, 256),
    ("multiclass", "lstm", "ssa"): HyperParameters(None, None, None, 3, 128, 0.1278, 0.0016, 256, 90, 256),
    ("multiclass", "cnn-lstm", "pso"): HyperParameters(256, 11, 7, 2, 128, 0.256, 0.0019, 256, 90, 256),
    ("multiclass", "cnn-lstm", "ssa"): HyperParameters(256, 9, 7, 2, 128, 0.118, 0.000192, 256, 82, 128),
}


def get_paper_preset(task: Task, model: ModelName, swarm: SwarmName) -> HyperParameters:
    try:
        return PAPER_PRESETS[(task, model, swarm)]
    except KeyError as exc:
        raise ValueError(f"No paper preset for task={task}, model={model}, swarm={swarm}") from exc


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable run contract; paths and protocols are never hidden in code."""

    dataset: Literal["cicids2017", "nsl-kdd"]
    cache_dir: str
    output_dir: str
    task: Task
    model: ModelName
    protocol: ProtocolName = "rigorous_evaluation"
    modeling_mode: ModelingMode = "feature_axis_replication"
    optimizer: Literal["fixed", "pso", "ssa", "random"] = "fixed"
    run_mode: Literal["smoke", "full"] = "full"
    population_size: int = 10
    iterations: int = 100
    epoch_cap: int | None = None
    swarm_preset: SwarmName = "ssa"
    seed: int = 42
    deterministic_ops: bool = True
    fitness: FitnessName = "macro_f1"
    false_positive_cost: float = 1.0
    false_negative_cost: float = 1.0
    threshold: float = 0.5
    window_size: int | None = None
    stride: int = 1
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    hyperparameters: dict[str, int | float | None] = field(default_factory=dict)

    def validate(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.modeling_mode == "temporal_window" and (
            self.window_size is None or self.window_size < 2
        ):
            raise ValueError("temporal_window mode requires window_size >= 2")
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("misclassification costs cannot be negative")
        if self.population_size < 1 or self.iterations < 1:
            raise ValueError("population_size and iterations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        self.validate()
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment config must be a YAML mapping")
        config = cls(**payload)
        config.validate()
        return config


PAPER_CIC_MULTICLASS_NAMES = (
    "BENIGN",
    "Bot",
    "Brute Force",
    "DDoS",
    "DoS",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "Web Attack",
)

PAPER_CIC_TEST_COUNTS = (418_898, 391, 1_754, 25_502, 39_014, 3, 6, 18_155, 437)

PAPER_BINARY_CONFUSIONS = {
    "pso": ((418_382, 516), (465, 84_797)),
    "ssa": ((418_391, 507), (355, 84_907)),
}

PAPER_MULTICLASS_CONFUSIONS = {
    "pso": (
        (418279, 1, 33, 45, 286, 0, 2, 229, 23),
        (237, 154, 0, 0, 0, 0, 0, 0, 0),
        (15, 0, 1735, 0, 4, 0, 0, 0, 0),
        (23, 0, 0, 25477, 2, 0, 0, 0, 0),
        (302, 0, 1, 0, 38711, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 3, 0, 0, 0),
        (3, 0, 0, 0, 0, 0, 3, 0, 0),
        (13, 0, 0, 0, 7, 0, 0, 18133, 2),
        (8, 0, 18, 0, 2, 0, 0, 0, 409),
    ),
    "ssa": (
        (418054, 47, 11, 18, 559, 0, 0, 207, 2),
        (215, 176, 0, 0, 0, 0, 0, 0, 0),
        (15, 0, 1738, 0, 1, 0, 0, 0, 0),
        (21, 0, 0, 25481, 0, 0, 0, 0, 0),
        (181, 0, 0, 0, 38832, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 3, 0, 0, 0),
        (3, 0, 0, 0, 0, 0, 3, 0, 0),
        (6, 0, 0, 0, 8, 0, 0, 18139, 2),
        (8, 0, 19, 0, 4, 0, 0, 0, 406),
    ),
}
