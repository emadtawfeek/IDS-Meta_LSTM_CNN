import numpy as np

from ids_repro.protocols import (
    chronological_group_split,
    grouped_split,
    make_temporal_windows,
    row_level_split,
)


def test_row_level_splits_are_disjoint_complete_and_stratified():
    labels = np.repeat(np.arange(3), 100)
    split = row_level_split(labels, protocol="rigorous_evaluation", seed=42)
    split.assert_disjoint_and_complete(len(labels))
    assert np.bincount(labels[split.test], minlength=3).tolist() == [20, 20, 20]
    assert np.bincount(labels[split.validation], minlength=3).tolist() == [10, 10, 10]


def test_paper_split_is_repeatable():
    labels = np.arange(200) % 3
    first = row_level_split(labels, protocol="paper_replication", seed=42)
    second = row_level_split(labels, protocol="paper_replication", seed=42)
    assert np.array_equal(first.train, second.train)
    assert "no stratification" in first.method


def test_temporal_windows_never_cross_groups_and_are_time_ordered():
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    labels = np.arange(12) % 2
    groups = np.array(["a"] * 6 + ["b"] * 6)
    timestamps = np.array([5, 1, 3, 2, 4, 0, 11, 7, 9, 8, 10, 6])
    windows, targets, window_groups = make_temporal_windows(
        features, labels, groups, timestamps, window_size=3, stride=2
    )
    assert windows.shape == (4, 3, 2)
    assert targets.shape == (4,)
    assert window_groups.tolist() == ["a", "a", "b", "b"]
    for window in windows:
        original_rows = window[:, 0].astype(int) // 2
        assert len(set(groups[original_rows])) == 1
        assert np.all(np.diff(timestamps[original_rows]) >= 0)


def test_chronological_split_has_no_overlap():
    groups = np.repeat([0, 1], 10)
    timestamps = np.tile(np.arange(10)[::-1], 2)
    split = chronological_group_split(groups, timestamps)
    split.assert_disjoint_and_complete(20)
    for group in [0, 1]:
        train_times = timestamps[split.train[groups[split.train] == group]]
        test_times = timestamps[split.test[groups[split.test] == group]]
        assert train_times.max() < test_times.min()


def test_grouped_split_never_leaks_a_group():
    groups = np.repeat(np.arange(20), 5)
    split = grouped_split(groups, seed=42)
    split.assert_disjoint_and_complete(len(groups))
    partitions = [set(groups[index]) for index in (split.train, split.validation, split.test)]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
