from __future__ import annotations

from .config import HyperParameters, ModelName, ModelingMode, Task


def build_model(
    model_name: ModelName,
    task: Task,
    params: HyperParameters,
    *,
    feature_count: int,
    class_count: int,
    modeling_mode: ModelingMode = "feature_axis_replication",
    time_steps: int | None = None,
):
    """Build a model with explicit semantics for its sequence axis."""

    import tensorflow as tf

    layers = tf.keras.layers
    model = tf.keras.Sequential(name=f"{model_name}-{task}")
    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    if modeling_mode == "temporal_window":
        if time_steps is None or time_steps < 2:
            raise ValueError("temporal_window mode requires time_steps >= 2")
        sequence_shape = (time_steps, feature_count)
    elif modeling_mode == "feature_axis_replication":
        sequence_shape = (feature_count, 1)
    else:
        raise ValueError(f"Unknown modeling mode: {modeling_mode}")

    if model_name in {"cnn", "cnn-lstm"}:
        if params.num_filters is None or params.kernel_size is None or params.pooling_size is None:
            raise ValueError(f"CNN hyperparameters are required for {model_name}")
        sequence_length = sequence_shape[0]
        if params.kernel_size > sequence_length:
            raise ValueError(
                f"kernel_size={params.kernel_size} exceeds sequence length={sequence_length}"
            )
        convolution_length = sequence_length - params.kernel_size + 1
        if params.pooling_size > convolution_length:
            raise ValueError(
                f"pooling_size={params.pooling_size} exceeds convolution length={convolution_length}"
            )
        model.add(layers.Input(shape=sequence_shape))
        model.add(
            layers.Conv1D(
                filters=params.num_filters,
                kernel_size=params.kernel_size,
                activation="relu",
            )
        )
        model.add(layers.MaxPooling1D(pool_size=params.pooling_size))
        if model_name == "cnn":
            model.add(layers.Dropout(params.dropout_rate))
            model.add(layers.Flatten())
        else:
            if params.lstm_units is None:
                raise ValueError("lstm_units is required for cnn-lstm")
            model.add(layers.LSTM(params.lstm_units, return_sequences=False))
    elif model_name == "lstm":
        if params.lstm_units is None:
            raise ValueError("lstm_units is required for lstm")
        lstm_shape = (
            (1, feature_count)
            if modeling_mode == "feature_axis_replication"
            else sequence_shape
        )
        model.add(layers.Input(shape=lstm_shape))
        model.add(layers.LSTM(params.lstm_units, return_sequences=False))
        model.add(layers.Dropout(params.dropout_rate))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    for _ in range(params.num_dense_layers):
        model.add(layers.Dense(params.dense_units, activation="relu"))
        model.add(layers.Dropout(params.dropout_rate))

    if task == "binary":
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    elif task == "multiclass":
        model.add(layers.Dense(class_count, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    else:
        raise ValueError(f"Unknown task: {task}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model
