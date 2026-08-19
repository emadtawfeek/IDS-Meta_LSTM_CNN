import json

import numpy as np
import pytest

from ids_repro.swarm import (
    CachedObjective,
    SearchPaused,
    particle_swarm_search,
    random_search,
    salp_swarm_search,
)


@pytest.mark.parametrize(
    "search", [particle_swarm_search, salp_swarm_search, random_search]
)
def test_checkpoint_resume_matches_uninterrupted_search(tmp_path, search):
    objective = lambda params, _: -(params.dropout_rate - 0.31) ** 2
    expected = search(
        objective, model_name="lstm", population_size=3, iterations=3, seed=17
    )
    checkpoint = tmp_path / f"{search.__name__}.json"
    with pytest.raises(SearchPaused):
        search(
            objective,
            model_name="lstm",
            population_size=3,
            iterations=3,
            seed=17,
            checkpoint_path=checkpoint,
            run_identity={"dataset_checksum": "abc"},
            max_evaluations=4,
        )
    actual = search(
        objective,
        model_name="lstm",
        population_size=3,
        iterations=3,
        seed=17,
        checkpoint_path=checkpoint,
        run_identity={"dataset_checksum": "abc"},
        resume=True,
    )
    assert np.array_equal(actual.best_position, expected.best_position)
    assert actual.convergence == expected.convergence
    assert [record["parameters"] for record in actual.evaluation_trace] == [
        record["parameters"] for record in expected.evaluation_trace
    ]


def test_resume_rejects_changed_dataset_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(SearchPaused):
        random_search(
            lambda params, _: params.dropout_rate,
            model_name="lstm",
            population_size=2,
            iterations=2,
            checkpoint_path=checkpoint,
            run_identity={"split": "first"},
            max_evaluations=1,
        )
    with pytest.raises(ValueError, match="identity"):
        random_search(
            lambda params, _: params.dropout_rate,
            model_name="lstm",
            population_size=2,
            iterations=2,
            checkpoint_path=checkpoint,
            run_identity={"split": "changed"},
            resume=True,
        )


def test_duplicate_cache_and_invalid_candidate_recovery():
    attempts = {"count": 0}

    def underlying(params, _):
        attempts["count"] += 1
        return {"fitness": params.dropout_rate, "best_epoch": 1}

    cached = CachedObjective(underlying, {"dataset": "same", "seed": 42})
    parameters = type("P", (), {})
    from ids_repro.config import get_paper_preset

    params = get_paper_preset("binary", "lstm", "ssa")
    first = cached(params, 0)
    second = cached(params, 1)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert attempts["count"] == 1

    calls = {"count": 0}

    def sometimes_invalid(params, _):
        calls["count"] += 1
        return np.nan if calls["count"] == 1 else params.dropout_rate

    result = random_search(
        sometimes_invalid, model_name="lstm", population_size=2, iterations=1, seed=3
    )
    assert result.evaluation_trace[0]["status"] == "failed"
    assert result.evaluation_trace[1]["status"] == "completed"
    assert result.best_fitness >= 0.1


def test_pso_velocity_clipping_is_checkpointed(tmp_path):
    checkpoint = tmp_path / "pso.json"
    with pytest.raises(SearchPaused):
        particle_swarm_search(
            lambda params, _: params.dropout_rate,
            model_name="lstm",
            population_size=3,
            iterations=3,
            seed=4,
            velocity_clip=0.01,
            checkpoint_path=checkpoint,
            max_evaluations=4,
        )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))["state"]
    assert np.max(np.abs(np.asarray(state["velocities"]))) <= 0.0100000001
