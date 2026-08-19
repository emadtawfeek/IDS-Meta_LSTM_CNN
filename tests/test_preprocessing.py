import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from ids_repro.data import (
    fit_training_label_encoder,
    normalize_column_names,
    numeric_validity,
)


def test_safe_column_normalization_preserves_order_and_rejects_duplicates():
    assert normalize_column_names([" A ", "B   C", "D"]) == ["A", "B C", "D"]
    with pytest.raises(ValueError, match="duplicates"):
        normalize_column_names([" A", "A "])


def test_numeric_cleaning_counts_missing_infinite_and_union():
    frame = pd.DataFrame(
        {"a": [1.0, np.nan, np.inf, "bad"], "b": [2.0, 3.0, -np.inf, 4.0]}
    )
    numeric, valid, report = numeric_validity(frame)
    assert valid.tolist() == [True, False, False, False]
    assert report == {
        "missing_rows": 2,
        "missing_cells": 2,
        "infinite_rows": 1,
        "infinite_cells": 2,
        "invalid_union_rows": 3,
    }
    assert np.isfinite(numeric.to_numpy()[valid]).all()


def test_scaler_is_fit_on_training_only():
    train = np.array([[0.0], [10.0]])
    validation = np.array([[20.0]])
    scaler = MinMaxScaler().fit(train)
    assert scaler.transform(train).max() == 1.0
    assert scaler.transform(validation).item() == 2.0


def test_label_encoder_is_fit_once_and_rejects_unseen_test_label():
    encoder, train, validation, test = fit_training_label_encoder(
        np.array(["normal", "attack", "normal"]),
        np.array(["attack"]),
        np.array(["normal"]),
    )
    assert encoder.classes_.tolist() == ["attack", "normal"]
    assert validation.tolist() == [0]
    assert test.tolist() == [1]
    with pytest.raises(ValueError, match="previously unseen"):
        encoder.transform(["unknown"])
