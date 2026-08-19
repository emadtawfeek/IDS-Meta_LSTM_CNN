from pathlib import Path

import numpy as np
import pytest

from ids_repro.data import (
    deterministic_stratified_subset,
    subset_manifest,
)
from ids_repro.selection import (
    selection_fitness,
    validation_selection_callback,
)


def test_stratified_cap_is_deterministic_and_contains_every_class():
    labels = np.repeat(np.arange(4), [80, 15, 4, 1])
    first = deterministic_stratified_subset(labels, 12, seed=42)
    second = deterministic_stratified_subset(labels, 12, seed=42)
    assert np.array_equal(first, second)
    assert set(labels[first]) == {0, 1, 2, 3}
    assert subset_manifest(first, labels) == subset_manifest(second, labels)


def test_stratified_cap_fails_when_every_class_cannot_fit():
    with pytest.raises(ValueError, match="cannot include all"):
        deterministic_stratified_subset(np.arange(5), 4, seed=42)


def test_cost_sensitive_is_binary_only_and_coefficients_are_validated():
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 0, 1])
    assert selection_fitness(
        truth,
        prediction,
        task="binary",
        fitness_name="cost_sensitive",
        false_positive_cost=1,
        false_negative_cost=3,
    ) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="at least one positive"):
        selection_fitness(
            truth,
            prediction,
            task="binary",
            fitness_name="cost_sensitive",
            false_positive_cost=0,
            false_negative_cost=0,
        )
    with pytest.raises(ValueError, match="binary-only"):
        selection_fitness(
            truth,
            prediction,
            task="multiclass",
            fitness_name="cost_sensitive",
        )


def test_epoch_selector_restores_requested_fitness_best_weights():
    probabilities = iter(
        [
            np.array([[0.1], [0.9], [0.9], [0.1]]),  # accuracy 0.5
            np.array([[0.1], [0.9], [0.1], [0.9]]),  # accuracy 1.0
            np.array([[0.9], [0.1], [0.9], [0.1]]),  # accuracy 0.0
        ]
    )

    class FakeModel:
        def __init__(self):
            self.epoch = 0
            self.restored = None
            self.stop_training = False

        def predict(self, *_args, **_kwargs):
            value = next(probabilities)
            self.epoch += 1
            return value

        def get_weights(self):
            return [np.array([self.epoch])]

        def set_weights(self, weights):
            self.restored = int(weights[0][0])

    callback = validation_selection_callback(
        np.zeros((4, 1)),
        np.array([0, 1, 0, 1]),
        task="binary",
        fitness_name="accuracy",
        batch_size=4,
        patience=1,
        min_delta=0.0,
    )
    model = FakeModel()
    callback.set_model(model)
    for epoch in range(3):
        callback.on_epoch_end(epoch, {})
    callback.on_train_end()
    outcome = callback.outcome()
    assert outcome.best_epoch == 2
    assert outcome.best_score == 1.0
    assert len(outcome.history) == 3
    assert model.stop_training
    assert model.restored == 2
