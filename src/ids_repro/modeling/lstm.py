"""LSTM builder using the shared explicit tensor contract."""

from ..models import build_model


def build_lstm(task, params, **kwargs):
    return build_model("lstm", task, params, **kwargs)
