"""Dataset-specific adapters and shared split/window contracts."""

from .cicids2017 import prepare_cicids2017
from .nsl_kdd import prepare_nsl_kdd

__all__ = ["prepare_cicids2017", "prepare_nsl_kdd"]
