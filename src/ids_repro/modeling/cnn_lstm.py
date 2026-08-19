"""CNN-LSTM builder using the shared explicit tensor contract."""

from ..models import build_model


def build_cnn_lstm(task, params, **kwargs):
    return build_model("cnn-lstm", task, params, **kwargs)
