from pathlib import Path

import numpy as np
import pytest

from ids_repro.config import get_paper_preset
from ids_repro.data import PreparedDataset
from ids_repro.training import train_and_evaluate


def test_trainer_rejects_rigorous_label_on_paper_split_cache(tmp_path):
    features = np.zeros((4, 3), dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    dataset = PreparedDataset(
        cache_dir=Path(tmp_path),
        train_x=features,
        train_y_multiclass=labels,
        val_x=features,
        val_y_multiclass=labels,
        test_x=features,
        test_y_multiclass=labels,
        class_names=("normal", "attack"),
        feature_names=("a", "b", "c"),
        metadata={"dataset": "synthetic", "paper_split_match": True},
    )
    with pytest.raises(ValueError, match="cache was prepared as paper_replication"):
        train_and_evaluate(
            dataset,
            task="binary",
            model_name="cnn-lstm",
            swarm_name="ssa",
            params=get_paper_preset("binary", "cnn-lstm", "ssa"),
            output_dir=tmp_path / "result",
            protocol="rigorous_evaluation",
            epochs_override=1,
            verbose=0,
        )
