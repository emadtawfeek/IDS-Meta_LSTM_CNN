from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from .config import ProtocolName


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    method: str

    def assert_disjoint_and_complete(self, population_size: int) -> None:
        parts = [np.asarray(x, dtype=np.int64) for x in (self.train, self.validation, self.test)]
        merged = np.concatenate(parts)
        if len(merged) != population_size:
            raise AssertionError(
                f"Split has {len(merged)} rows but population has {population_size}"
            )
        if len(np.unique(merged)) != population_size:
            raise AssertionError("Train/validation/test indices overlap")
        if merged.min(initial=0) < 0 or merged.max(initial=-1) >= population_size:
            raise AssertionError("Split contains an out-of-range index")


def row_level_split(
    labels: np.ndarray,
    *,
    protocol: ProtocolName,
    seed: int = 42,
) -> SplitIndices:
    """Return 70/10/20 indices for the documented replication or rigorous protocol."""

    labels = np.asarray(labels)
    indices = np.arange(len(labels), dtype=np.int64)
    rigorous = protocol in {"rigorous", "rigorous_evaluation"}
    stratify = labels if rigorous else None
    trainval, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    trainval_labels = labels[trainval] if rigorous else None
    train, validation = train_test_split(
        trainval,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=trainval_labels,
    )
    method = (
        "two shuffled train_test_split calls (70/10/20), random_state=42, no stratification"
        if not rigorous
        else "two class-stratified shuffled train_test_split calls (70/10/20)"
    )
    split = SplitIndices(train, validation, test, method)
    split.assert_disjoint_and_complete(len(labels))
    return split


def official_train_validation_split(
    labels: np.ndarray,
    *,
    protocol: ProtocolName,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split an official training set while reserving the official test set untouched."""

    indices = np.arange(len(labels), dtype=np.int64)
    rigorous = protocol in {"rigorous", "rigorous_evaluation"}
    stratify = labels if rigorous else None
    train, validation = train_test_split(
        indices,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    if np.intersect1d(train, validation).size:
        raise AssertionError("Training and validation indices overlap")
    method = (
        "official train/test; unstratified 12.5% validation holdout from official train"
        if not rigorous
        else "official train/test; class-stratified 12.5% validation holdout from official train"
    )
    return train, validation, method


def chronological_group_split(
    groups: np.ndarray,
    timestamps: np.ndarray,
) -> SplitIndices:
    """Chronological 70/10/20 split inside each capture group.

    Each group contributes contiguous time ranges, making windows and their source rows
    unable to cross train/validation/test boundaries.
    """

    groups = np.asarray(groups)
    timestamps = np.asarray(timestamps)
    if len(groups) != len(timestamps):
        raise ValueError("groups and timestamps must have equal length")
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        ordered = members[np.argsort(timestamps[members], kind="stable")]
        first = int(np.floor(0.70 * len(ordered)))
        second = int(np.floor(0.80 * len(ordered)))
        train_parts.append(ordered[:first])
        val_parts.append(ordered[first:second])
        test_parts.append(ordered[second:])
    split = SplitIndices(
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
        "chronological 70/10/20 ranges within each capture file",
    )
    split.assert_disjoint_and_complete(len(groups))
    return split


def grouped_split(groups: np.ndarray, *, seed: int = 42) -> SplitIndices:
    """Create approximate 70/10/20 partitions with no group crossing a boundary."""

    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) < 3:
        raise ValueError("groups must be a rank-1 vector with at least three rows")
    indices = np.arange(len(groups), dtype=np.int64)
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_local, test_local = next(first.split(indices, groups=groups))
    trainval = indices[trainval_local]
    test = indices[test_local]
    second = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_local, validation_local = next(
        second.split(trainval, groups=groups[trainval])
    )
    split = SplitIndices(
        trainval[train_local],
        trainval[validation_local],
        test,
        "group-aware 70/10/20 GroupShuffleSplit; no group crosses a partition",
    )
    split.assert_disjoint_and_complete(len(groups))
    train_groups = set(groups[split.train])
    validation_groups = set(groups[split.validation])
    test_groups = set(groups[split.test])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise AssertionError("A group crosses train/validation/test boundaries")
    return split


def make_temporal_windows(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    *,
    window_size: int,
    stride: int = 1,
    label_rule: Literal["last", "majority"] = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build time-ordered windows without ever crossing a capture/session group."""

    features = np.asarray(features)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    timestamps = np.asarray(timestamps)
    if not (len(features) == len(labels) == len(groups) == len(timestamps)):
        raise ValueError("features, labels, groups, and timestamps must align")
    if features.ndim != 2:
        raise ValueError("Temporal source features must be a rank-2 row/feature matrix")
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be positive")

    windows: list[np.ndarray] = []
    targets: list[int] = []
    window_groups: list[object] = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        ordered = members[np.argsort(timestamps[members], kind="stable")]
        for start in range(0, len(ordered) - window_size + 1, stride):
            chosen = ordered[start : start + window_size]
            windows.append(features[chosen])
            if label_rule == "last":
                targets.append(int(labels[chosen[-1]]))
            elif label_rule == "majority":
                counts = np.bincount(labels[chosen].astype(np.int64))
                targets.append(int(np.flatnonzero(counts == counts.max())[-1]))
            else:
                raise ValueError(f"Unknown label_rule: {label_rule}")
            window_groups.append(group)
    if not windows:
        raise ValueError("No temporal windows could be constructed")
    return np.stack(windows), np.asarray(targets), np.asarray(window_groups)
