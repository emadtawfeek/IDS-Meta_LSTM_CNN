import pytest

from ids_repro.config import HyperParameters
from ids_repro.models import build_model


PARAMS = HyperParameters(16, 3, 2, 1, 16, 0.2, 0.001, 16, 1, 16)


@pytest.mark.parametrize("model_name", ["cnn", "lstm", "cnn-lstm"])
def test_feature_axis_models_have_expected_rank(model_name):
    model = build_model(
        model_name, "multiclass", PARAMS, feature_count=70, class_count=9
    )
    assert len(model.input_shape) == 3
    assert model.output_shape[-1] == 9


def test_temporal_model_input_keeps_time_and_feature_axes():
    model = build_model(
        "cnn-lstm",
        "binary",
        PARAMS,
        feature_count=70,
        class_count=2,
        modeling_mode="temporal_window",
        time_steps=8,
    )
    assert model.input_shape == (None, 8, 70)
    assert model.output_shape == (None, 1)


def test_cnn_rejects_kernel_larger_than_temporal_window():
    with pytest.raises(ValueError, match="kernel_size"):
        build_model(
            "cnn-lstm",
            "binary",
            PARAMS,
            feature_count=70,
            class_count=2,
            modeling_mode="temporal_window",
            time_steps=2,
        )
