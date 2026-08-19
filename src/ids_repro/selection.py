from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import FitnessName, Task


def validate_fitness_configuration(
    task: Task,
    fitness_name: FitnessName,
    false_positive_cost: float,
    false_negative_cost: float,
) -> None:
    costs = (false_positive_cost, false_negative_cost)
    if not all(math.isfinite(value) and value >= 0 for value in costs):
        raise ValueError("misclassification costs must be finite and non-negative")
    if fitness_name == "cost_sensitive":
        if task != "binary":
            raise ValueError(
                "cost_sensitive fitness is binary-only until a multiclass cost "
                "matrix is supplied"
            )
        if not any(costs):
            raise ValueError("cost_sensitive fitness requires at least one positive cost")


def predictions_from_probabilities(probabilities: np.ndarray, task: Task) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if task == "binary":
        return (probabilities.reshape(-1) >= 0.5).astype(np.uint8)
    return np.argmax(probabilities, axis=1).astype(np.uint8)


def selection_fitness(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    task: Task,
    fitness_name: FitnessName,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> float:
    """Compute the single maximized objective used by every selection path."""

    validate_fitness_configuration(
        task, fitness_name, false_positive_cost, false_negative_cost
    )
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth and prediction must be aligned rank-1 arrays")
    if fitness_name == "accuracy":
        return float(accuracy_score(truth, prediction))
    if fitness_name == "macro_f1":
        return float(f1_score(truth, prediction, average="macro", zero_division=0))
    if fitness_name == "cost_sensitive":
        false_positives = int(np.sum((truth == 0) & (prediction == 1)))
        false_negatives = int(np.sum((truth == 1) & (prediction == 0)))
        true_negatives = int(np.sum((truth == 0) & (prediction == 0)))
        true_positives = int(np.sum((truth == 1) & (prediction == 1)))
        false_alarm_rate = false_positives / max(1, false_positives + true_negatives)
        false_negative_rate = false_negatives / max(
            1, false_negatives + true_positives
        )
        normalized_cost = (
            false_positive_cost * false_alarm_rate
            + false_negative_cost * false_negative_rate
        ) / (false_positive_cost + false_negative_cost)
        return -float(normalized_cost)
    raise ValueError(f"Unknown fitness: {fitness_name}")


def cost_equation(false_positive_cost: float, false_negative_cost: float) -> dict:
    return {
        "objective": "maximize -(c_fp * FAR + c_fn * FNR) / (c_fp + c_fn)",
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost": float(false_negative_cost),
        "scope": "binary classification",
    }


@dataclass
class SelectionOutcome:
    best_score: float
    best_epoch: int
    history: list[dict]
    epochs_completed: int


def validation_selection_callback(
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    task: Task,
    fitness_name: FitnessName,
    batch_size: int,
    patience: int,
    min_delta: float,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    restore_best_weights: bool = True,
    early_stopping: bool = True,
):
    """Create a Keras callback that selects and restores by the requested fitness."""

    import tensorflow as tf

    if patience < 0:
        raise ValueError("patience cannot be negative")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")
    validate_fitness_configuration(
        task, fitness_name, false_positive_cost, false_negative_cost
    )

    class ValidationSelection(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_score = -np.inf
            self.best_epoch = 0
            self.best_weights = None
            self.wait = 0
            self.fitness_history: list[dict] = []

        def on_epoch_end(self, epoch, logs=None):
            probabilities = self.model.predict(
                validation_x, batch_size=batch_size, verbose=0
            )
            prediction = predictions_from_probabilities(probabilities, task)
            score = selection_fitness(
                validation_y,
                prediction,
                task=task,
                fitness_name=fitness_name,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            if not math.isfinite(score):
                raise FloatingPointError(
                    f"Non-finite validation {fitness_name} at epoch {epoch + 1}"
                )
            self.fitness_history.append(
                {
                    "epoch": epoch + 1,
                    "fitness_name": fitness_name,
                    "fitness": score,
                }
            )
            if logs is not None:
                logs[f"val_selection_{fitness_name}"] = score
            if score > self.best_score + min_delta:
                self.best_score = score
                self.best_epoch = epoch + 1
                self.best_weights = (
                    self.model.get_weights() if restore_best_weights else None
                )
                self.wait = 0
            else:
                self.wait += 1
                if early_stopping and self.wait >= patience and patience >= 0:
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

        def outcome(self) -> SelectionOutcome:
            return SelectionOutcome(
                best_score=float(self.best_score),
                best_epoch=int(self.best_epoch),
                history=list(self.fitness_history),
                epochs_completed=len(self.fitness_history),
            )

    return ValidationSelection()
