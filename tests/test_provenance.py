import json
from pathlib import Path

import numpy as np
import pytest

from ids_repro.data import PreparedDataset, cache_identity
from ids_repro.provenance import load_parameter_selection
from ids_repro.swarm import decode_position


def _dataset(tmp_path):
    features = np.zeros((4, 3), dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    return PreparedDataset(
        cache_dir=Path(tmp_path),
        train_x=features,
        train_y_multiclass=labels,
        val_x=features,
        val_y_multiclass=labels,
        test_x=features,
        test_y_multiclass=labels,
        class_names=("normal", "attack"),
        feature_names=("a", "b", "c"),
        metadata={"dataset": "synthetic", "protocol": "rigorous_evaluation"},
    )


def test_random_search_artifact_cannot_be_labeled_pso(tmp_path):
    dataset = _dataset(tmp_path)
    position = np.zeros(7)
    params = decode_position(position, "lstm")
    payload = {
        "algorithm": "random",
        "best_position": position.tolist(),
        "best_parameters": params.to_dict(),
        "evaluations": 2,
        "metadata": {
            "dataset": "synthetic",
            "task": "binary",
            "model": "lstm",
            "algorithm": "random",
            "seed": 42,
            "fitness": "macro_f1",
            "evaluation_budget": 2,
            "cache_identity": cache_identity(dataset),
        },
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, provenance = load_parameter_selection(
        path,
        selection_source="random_search",
        dataset=dataset,
        task="binary",
        model="lstm",
        seed=42,
        expected_fitness="macro_f1",
    )
    assert loaded == params
    assert provenance["algorithm"] == "random"
    with pytest.raises(ValueError, match="requires algorithm=pso"):
        load_parameter_selection(
            path,
            selection_source="pso_search",
            dataset=dataset,
            task="binary",
            model="lstm",
            seed=42,
            expected_fitness="macro_f1",
        )
