import numpy as np
import pytest

from ids_repro.data import reshape_for_model


def test_feature_axis_shapes_are_explicit():
    values = np.zeros((5, 70), dtype=np.float32)
    assert reshape_for_model(values, "cnn").shape == (5, 70, 1)
    assert reshape_for_model(values, "cnn-lstm").shape == (5, 70, 1)
    assert reshape_for_model(values, "lstm").shape == (5, 1, 70)


def test_temporal_mode_rejects_rank_two_input():
    with pytest.raises(ValueError, match="samples, time_steps, features"):
        reshape_for_model(
            np.zeros((5, 70), dtype=np.float32), "cnn-lstm", "temporal_window"
        )


def test_temporal_mode_preserves_rank_three_input():
    values = np.zeros((5, 8, 70), dtype=np.float32)
    assert reshape_for_model(values, "cnn-lstm", "temporal_window") is values
