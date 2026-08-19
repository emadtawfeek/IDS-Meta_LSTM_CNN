"""CNN builder using the shared explicit tensor contract."""

from ..models import build_model


def build_cnn(task, params, **kwargs):
    return build_model("cnn", task, params, **kwargs)
