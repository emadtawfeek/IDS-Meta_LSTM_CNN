import pytest

from ids_repro.config import ExperimentConfig, PAPER_CIC_TEST_COUNTS, get_paper_preset


def test_paper_ssa_cnn_lstm_binary_preset():
    preset = get_paper_preset("binary", "cnn-lstm", "ssa")
    assert preset.num_filters == 256
    assert preset.kernel_size == 9
    assert preset.pooling_size == 5
    assert preset.lstm_units == 128
    assert preset.epochs == 79


def test_paper_test_total():
    assert sum(PAPER_CIC_TEST_COUNTS) == 504_160


def test_experiment_config_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    original = ExperimentConfig(
        dataset="cicids2017",
        cache_dir="cache",
        output_dir="results",
        task="binary",
        model="cnn-lstm",
        protocol="rigorous_evaluation",
        optimizer="pso",
        population_size=3,
        iterations=2,
    )
    original.save(path)
    assert ExperimentConfig.load(path) == original


def test_temporal_config_is_explicitly_unavailable():
    config = ExperimentConfig(
        dataset="cicids2017",
        cache_dir="cache",
        output_dir="results",
        task="binary",
        model="cnn-lstm",
        optimizer="pso",
        modeling_mode="temporal_window",
        window_size=10,
    )
    with pytest.raises(ValueError, match="unavailable"):
        config.validate()


def test_nsl_empty_paper_preset_is_rejected():
    config = ExperimentConfig(
        dataset="nsl-kdd",
        cache_dir="cache",
        output_dir="results",
        task="binary",
        model="cnn-lstm",
        selection_source="paper_preset",
    )
    with pytest.raises(ValueError, match="no paper preset"):
        config.validate()
