from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import HyperParameters, ModelName, SelectionSource, Task
from .data import PreparedDataset, cache_identity
from .swarm import decode_position


SEARCH_SOURCES = {
    "pso_search": "pso",
    "ssa_search": "ssa",
    "random_search": "random",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_parameter_selection(
    path: Path | str,
    *,
    selection_source: SelectionSource,
    dataset: PreparedDataset,
    task: Task,
    model: ModelName,
    seed: int,
    expected_fitness: str | None = None,
) -> tuple[HyperParameters, dict]:
    """Load parameters and reject search artifacts from a different experiment."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("best_parameters", payload.get("parameters", payload))
    params = HyperParameters(**parameters)
    expected_algorithm = SEARCH_SOURCES.get(selection_source)
    identity = cache_identity(dataset)
    metadata = payload.get("metadata", {})

    if expected_algorithm is not None:
        required = {
            "dataset",
            "task",
            "model",
            "algorithm",
            "seed",
            "fitness",
            "evaluation_budget",
            "cache_identity",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(f"Search parameter artifact lacks metadata fields: {missing}")
        actual_algorithm = payload.get("algorithm", metadata.get("algorithm"))
        if actual_algorithm != expected_algorithm:
            raise ValueError(
                f"selection_source={selection_source} requires algorithm="
                f"{expected_algorithm}; artifact says {actual_algorithm!r}"
            )
        expected = {
            "dataset": dataset.metadata["dataset"],
            "task": task,
            "model": model,
            "seed": seed,
        }
        for key, value in expected.items():
            actual = metadata.get(key)
            if actual != value:
                raise ValueError(
                    f"Parameter artifact {key} mismatch: expected {value!r}, got {actual!r}"
                )
        artifact_identity = metadata.get("cache_identity", {})
        if artifact_identity.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("Parameter artifact cache identity/checksum does not match")
        if expected_fitness is not None and metadata.get("fitness") != expected_fitness:
            raise ValueError(
                f"Parameter artifact fitness mismatch: expected {expected_fitness!r}, "
                f"got {metadata.get('fitness')!r}"
            )
        budget = int(metadata["evaluation_budget"])
        if budget < 1 or int(payload.get("evaluations", -1)) != budget:
            raise ValueError("Search artifact is incomplete or has an inconsistent budget")
        raw_position = payload.get("best_position")
        if raw_position is None:
            raise ValueError("Search artifact lacks the raw best position")
        decoded = decode_position(raw_position, model)
        if decoded != params:
            raise ValueError("Raw best position does not decode to saved best_parameters")

    provenance = {
        "selection_source": selection_source,
        "parameter_artifact_path": str(path.resolve()),
        "parameter_artifact_sha256": file_sha256(path),
        "algorithm": expected_algorithm,
        "seed": seed,
        "fitness": metadata.get("fitness"),
        "budget": metadata.get("evaluation_budget"),
        "dataset": dataset.metadata["dataset"],
        "task": task,
        "model": model,
        "cache_identity": identity,
        "raw_best_position": payload.get("best_position"),
        "decoded_parameters": params.to_dict(),
    }
    return params, provenance


def preset_provenance(
    *,
    selection_source: SelectionSource,
    paper_optimizer: str,
    dataset: PreparedDataset,
    task: Task,
    model: ModelName,
    seed: int,
    params: HyperParameters,
) -> dict:
    if selection_source == "paper_preset" and dataset.metadata["dataset"] != "cicids2017":
        raise ValueError("paper_preset is only valid for CIC-IDS2017")
    if selection_source == "transferred_cic_preset" and dataset.metadata["dataset"] != "nsl-kdd":
        raise ValueError("transferred_cic_preset is only valid for NSL-KDD")
    return {
        "selection_source": selection_source,
        "algorithm": paper_optimizer,
        "seed": seed,
        "fitness": "reported paper preset; search trace unavailable",
        "budget": None,
        "dataset": dataset.metadata["dataset"],
        "task": task,
        "model": model,
        "cache_identity": cache_identity(dataset),
        "raw_best_position": None,
        "decoded_parameters": params.to_dict(),
        "transfer_warning": (
            "CIC-IDS2017 paper preset transferred to NSL-KDD; not a paper result"
            if selection_source == "transferred_cic_preset"
            else None
        ),
    }
