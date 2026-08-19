import numpy as np
import pytest

from ids_repro.statistics import mcnemar_comparison


def test_mcnemar_reports_each_method_alone_correct():
    truth = np.array([0, 0, 1, 1, 0, 1])
    first = np.array([0, 1, 1, 0, 0, 1])
    second = np.array([0, 0, 0, 1, 0, 1])
    result = mcnemar_comparison(
        first,
        second,
        truth,
        truth.copy(),
        first_split_checksum="same",
        second_split_checksum="same",
    )
    assert result["samples_each_method_alone_correct"] == {"first": 1, "second": 2}
    assert result["test"] == "exact_binomial"


def test_mcnemar_rejects_mismatched_truth_or_split():
    truth = np.array([0, 1])
    with pytest.raises(ValueError, match="checksums"):
        mcnemar_comparison(
            truth, truth, truth, truth, first_split_checksum="a", second_split_checksum="b"
        )
    with pytest.raises(ValueError, match="not identical"):
        mcnemar_comparison(
            truth,
            truth,
            truth,
            truth[::-1],
            first_split_checksum="a",
            second_split_checksum="a",
        )
