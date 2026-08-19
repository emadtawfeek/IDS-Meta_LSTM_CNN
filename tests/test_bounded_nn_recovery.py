from pathlib import Path

import numpy as np
import pytest

from ids_repro.data import PreparedDataset
from ids_repro.swarm import CachedObjective, SearchPaused, random_search
from ids_repro.training import validation_fitness


def test_bounded_neural_search_resumes_with_epoch_fitness_history(tmp_path):
    rng = np.random.RandomState(5)
    train_x = rng.normal(size=(20, 4)).astype(np.float32)
    val_x = rng.normal(size=(10, 4)).astype(np.float32)
    test_x = rng.normal(size=(6, 4)).astype(np.float32)
    dataset = PreparedDataset(
        cache_dir=Path(tmp_path),
        train_x=train_x,
        train_y_multiclass=np.tile([0, 1], 10).astype(np.uint8),
        val_x=val_x,
        val_y_multiclass=np.tile([0, 1], 5).astype(np.uint8),
        test_x=test_x,
        test_y_multiclass=np.tile([0, 1], 3).astype(np.uint8),
        class_names=("normal", "attack"),
        feature_names=("a", "b", "c", "d"),
        metadata={"dataset": "synthetic", "protocol": "rigorous_evaluation"},
    )

    def build_objective():
        return CachedObjective(
            lambda params, _: validation_fitness(
                dataset,
                task="binary",
                model_name="lstm",
                params=params,
                seed=9,
                epoch_cap=1,
                max_train_samples=12,
                max_val_samples=8,
                fitness_name="macro_f1",
                patience=2,
                return_details=True,
            ),
            {
                "dataset_split_checksum": "synthetic-v1",
                "task": "binary",
                "model": "lstm",
                "training_subset": "fixed",
                "seed": 9,
                "epoch_cap": 1,
                "fitness": "macro_f1",
            },
        )

    checkpoint = tmp_path / "bounded-nn-checkpoint.json"
    with pytest.raises(SearchPaused):
        random_search(
            build_objective(),
            model_name="lstm",
            population_size=1,
            iterations=2,
            seed=13,
            checkpoint_path=checkpoint,
            run_identity={"dataset": "synthetic-v1"},
            max_evaluations=1,
        )
    result = random_search(
        build_objective(),
        model_name="lstm",
        population_size=1,
        iterations=2,
        seed=13,
        checkpoint_path=checkpoint,
        run_identity={"dataset": "synthetic-v1"},
        resume=True,
    )
    assert result.evaluations == 2
    assert all(record["best_epoch"] == 1 for record in result.evaluation_trace)
    assert all(len(record["fitness_history"]) == 1 for record in result.evaluation_trace)
    assert result.to_dict()["completed_nn_fits"] == 2
