"""Dataset-specific NSL-KDD extension adapter; no CIC feature mapping is assumed."""

from ..data import NSL_CLASS_NAMES, NSL_COLUMNS, prepare_nsl_kdd

__all__ = ["NSL_CLASS_NAMES", "NSL_COLUMNS", "prepare_nsl_kdd"]
