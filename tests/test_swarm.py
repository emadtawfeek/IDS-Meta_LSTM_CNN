import numpy as np

from ids_repro.swarm import (
    decode_position,
    particle_swarm_search,
    random_search,
    salp_swarm_search,
)


def test_decode_stays_inside_paper_space():
    params = decode_position(np.zeros(10), "cnn-lstm")
    assert params.num_filters == 16
    assert params.kernel_size == 3
    assert params.pooling_size == 2
    assert params.lstm_units == 16
    assert params.epochs == 10


def test_pso_evaluation_budget_and_direction():
    def objective(params, _evaluation):
        return params.dropout_rate

    result = particle_swarm_search(
        objective, model_name="lstm", population_size=3, iterations=2, seed=1
    )
    assert result.evaluations == 6
    assert 0.1 <= result.best_parameters.dropout_rate <= 0.5


def test_ssa_evaluation_budget():
    def objective(params, _evaluation):
        return -abs(params.learning_rate - 0.001)

    result = salp_swarm_search(
        objective, model_name="cnn", population_size=4, iterations=3, seed=2
    )
    assert result.evaluations == 12


def test_optimizer_coordinates_remain_normalized_and_traceable():
    result = particle_swarm_search(
        lambda params, _: -abs(params.dropout_rate - 0.3),
        model_name="lstm",
        population_size=4,
        iterations=4,
        seed=3,
    )
    history = np.asarray(result.position_history)
    assert history.shape == (4, 4, 7)
    assert np.all((history >= 0.0) & (history <= 1.0))
    assert len(result.evaluation_trace) == result.evaluations == 16


def test_budget_matched_random_search():
    result = random_search(
        lambda params, _: params.dropout_rate,
        model_name="cnn",
        population_size=3,
        iterations=2,
        seed=4,
    )
    assert result.algorithm == "random"
    assert result.evaluations == 6
    assert len(result.convergence) == 2


def test_ssa_maximization_is_monotonic_and_population_persists():
    def known_optimum(params, _):
        return -(params.dropout_rate - 0.3) ** 2

    result = salp_swarm_search(
        known_optimum,
        model_name="lstm",
        population_size=8,
        iterations=20,
        seed=7,
    )
    assert np.all(np.diff(result.convergence) >= 0)
    assert result.best_fitness > -0.0025
    history = np.asarray(result.position_history)
    assert not np.array_equal(history[0], history[1])


def test_pso_is_deterministic_for_fixed_seed():
    objective = lambda params, _: -(params.learning_rate - 0.001) ** 2
    first = particle_swarm_search(
        objective, model_name="cnn", population_size=4, iterations=4, seed=9
    )
    second = particle_swarm_search(
        objective, model_name="cnn", population_size=4, iterations=4, seed=9
    )
    assert np.array_equal(first.best_position, second.best_position)
    assert first.convergence == second.convergence


def test_search_artifacts_serialize_and_reload(tmp_path):
    result = random_search(
        lambda params, _: params.dropout_rate,
        model_name="lstm",
        population_size=2,
        iterations=2,
        seed=11,
    )
    path = tmp_path / "search.json"
    result.save(path)
    result.save_artifacts(tmp_path / "search")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["evaluations"] == 4
    assert (tmp_path / "search" / "optimization_trace.csv").exists()
    assert (tmp_path / "search" / "convergence_curve.png").exists()
