from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from .config import (
    PAPER_CIC_MULTICLASS_NAMES,
    PAPER_CIC_TEST_COUNTS,
    ModelingMode,
    ProtocolName,
)
from .protocols import official_train_validation_split, row_level_split


@dataclass(frozen=True)
class PreparedDataset:
    cache_dir: Path
    train_x: np.ndarray
    train_y_multiclass: np.ndarray
    val_x: np.ndarray
    val_y_multiclass: np.ndarray
    test_x: np.ndarray
    test_y_multiclass: np.ndarray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    metadata: dict

    def assert_integrity(self) -> None:
        for split in ("train", "val", "test"):
            features = getattr(self, f"{split}_x")
            labels = getattr(self, f"{split}_y_multiclass")
            if len(features) != len(labels):
                raise AssertionError(f"{split} feature/label lengths differ")
            if features.ndim not in (2, 3):
                raise AssertionError(f"{split} features must have rank 2 or 3")
            if not np.issubdtype(features.dtype, np.floating):
                raise AssertionError(f"{split} features must be floating point")
            if len(labels) and (labels.min() < 0 or labels.max() >= len(self.class_names)):
                raise AssertionError(f"{split} contains an invalid class code")

    def labels(self, split: str, task: str) -> np.ndarray:
        y = getattr(self, f"{split}_y_multiclass")
        if task == "binary":
            return (y != 0).astype(np.uint8, copy=False)
        if task == "multiclass":
            return y
        raise ValueError(f"Unknown task: {task}")


CIC_DROP_FEATURES = {
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}

EXPECTED_CIC_FILE_NAMES = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
}


def normalize_column_names(columns: list[object] | tuple[object, ...]) -> list[str]:
    """Trim and collapse whitespace while refusing ambiguous duplicate names."""

    normalized = [" ".join(str(column).strip().split()) for column in columns]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Column normalization produced duplicates: {duplicates}")
    return normalized


def numeric_validity(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Coerce features to numeric and count invalid values before removal."""

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    missing = numeric.isna().to_numpy()
    infinite = np.isinf(values)
    valid = ~(missing.any(axis=1) | infinite.any(axis=1))
    return numeric, valid, {
        "missing_rows": int(missing.any(axis=1).sum()),
        "missing_cells": int(missing.sum()),
        "infinite_rows": int(infinite.any(axis=1).sum()),
        "infinite_cells": int(infinite.sum()),
        "invalid_union_rows": int((~valid).sum()),
    }


def fit_training_label_encoder(
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    """Fit once on training labels and transform both held-out partitions."""

    encoder = LabelEncoder().fit(train_labels)
    return (
        encoder,
        encoder.transform(train_labels),
        encoder.transform(validation_labels),
        encoder.transform(test_labels),
    )


def _write_preprocessing_artifacts(
    cache_dir: Path,
    *,
    feature_names: list[str],
    label_mapping: dict[str, int],
    report: dict,
    mapping_report: dict,
) -> None:
    (cache_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    (cache_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2, sort_keys=True), encoding="utf-8"
    )
    (cache_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (cache_dir / "dataset_mapping_report.json").write_text(
        json.dumps(mapping_report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_existing_for_protocol(
    cache_dir: Path, requested: ProtocolName
) -> PreparedDataset:
    dataset = load_prepared(cache_dir)
    existing = dataset.metadata.get("protocol")
    if existing is None:
        existing = (
            "paper_replication"
            if dataset.metadata.get("paper_split_match")
            else "legacy_unspecified"
        )
    requested_rigorous = requested in {"rigorous", "rigorous_evaluation"}
    existing_rigorous = existing in {"rigorous", "rigorous_evaluation"}
    if (requested_rigorous and not existing_rigorous) or (
        requested == "paper_replication" and existing != "paper_replication"
    ):
        raise ValueError(
            f"Cache {cache_dir} already exists with protocol={existing}; requested={requested}. "
            "Use a separate cache directory for each protocol."
        )
    return dataset


def load_prepared(cache_dir: Path | str) -> PreparedDataset:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = PreparedDataset(
        cache_dir=cache_dir,
        train_x=np.load(cache_dir / "train_x.npy", mmap_mode="r"),
        train_y_multiclass=np.load(cache_dir / "train_y.npy", mmap_mode="r"),
        val_x=np.load(cache_dir / "val_x.npy", mmap_mode="r"),
        val_y_multiclass=np.load(cache_dir / "val_y.npy", mmap_mode="r"),
        test_x=np.load(cache_dir / "test_x.npy", mmap_mode="r"),
        test_y_multiclass=np.load(cache_dir / "test_y.npy", mmap_mode="r"),
        class_names=tuple(metadata["class_names"]),
        feature_names=tuple(metadata["feature_names"]),
        metadata=metadata,
    )
    dataset.assert_integrity()
    return dataset


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _find_cic_files(data_dir: Path) -> list[Path]:
    files = sorted(
        (p for p in data_dir.rglob("*.csv") if p.name in EXPECTED_CIC_FILE_NAMES),
        key=lambda p: p.name,
    )
    names = {p.name for p in files}
    if names != EXPECTED_CIC_FILE_NAMES:
        missing = sorted(EXPECTED_CIC_FILE_NAMES - names)
        extra = sorted(names - EXPECTED_CIC_FILE_NAMES)
        raise FileNotFoundError(
            f"Expected the eight CIC-IDS2017 MachineLearningCVE CSV files. "
            f"Missing={missing}; unexpected matching files={extra}; searched={data_dir}"
        )
    return files


def _cic_group_label(raw: object) -> int:
    label = str(raw).strip()
    if label == "BENIGN":
        return 0
    if label == "Bot":
        return 1
    if label in {"FTP-Patator", "SSH-Patator"}:
        return 2
    if label == "DDoS":
        return 3
    if label.startswith("DoS "):
        return 4
    if label == "Heartbleed":
        return 5
    if label == "Infiltration":
        return 6
    if label == "PortScan":
        return 7
    # The downloaded CSVs may decode the dash in these labels as U+FFFD.
    if label.startswith("Web Attack"):
        return 8
    raise ValueError(f"Unexpected CIC-IDS2017 label: {label!r}")


def _fit_minmax_in_chunks(
    features: np.ndarray, train_indices: np.ndarray, chunk_size: int
) -> MinMaxScaler:
    scaler = MinMaxScaler(copy=True)
    for start in range(0, len(train_indices), chunk_size):
        scaler.partial_fit(features[train_indices[start : start + chunk_size]])
    return scaler


def _write_scaled_split(
    path: Path,
    features: np.ndarray,
    indices: np.ndarray,
    scaler: MinMaxScaler,
    chunk_size: int,
) -> None:
    output = open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(indices), features.shape[1]),
    )
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        output[start:stop] = scaler.transform(features[indices[start:stop]])
    output.flush()
    del output


def prepare_cicids2017(
    data_dir: Path | str,
    cache_dir: Path | str,
    *,
    force: bool = False,
    chunk_size: int = 100_000,
    protocol: ProtocolName = "paper_replication",
) -> PreparedDataset:
    """Build the exact cleaned 70/10/20 split implied by the paper's matrices.

    The paper omits duplicate removal, but the published matrices prove it occurred:
    invalid-row removal followed by exact deduplication leaves 2,520,798 rows and a
    504,160-row test set. Filename order is also material to the random split.
    """

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if (cache_dir / "metadata.json").exists() and not force:
        return _load_existing_for_protocol(cache_dir, protocol)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _find_cic_files(data_dir)

    seen_hashes: set[int] = set()
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    raw_rows = invalid_rows = duplicate_rows = 0
    missing_rows = missing_cells = infinite_rows = infinite_cells = 0
    feature_names: list[str] | None = None
    retained_positions: list[int] | None = None
    raw_label_counts: Counter[str] = Counter()

    for path in files:
        for frame in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            raw_rows += len(frame)
            stripped_columns = normalize_column_names(list(frame.columns))
            label_position = next(
                (i for i, name in enumerate(stripped_columns) if name.lower() == "label"),
                None,
            )
            if label_position is None:
                raise ValueError(f"No Label column in {path}")
            if feature_names is None:
                retained_positions = [
                    i
                    for i, name in enumerate(stripped_columns)
                    if i != label_position and name not in CIC_DROP_FEATURES
                ]
                feature_names = [stripped_columns[i] for i in retained_positions]
            assert retained_positions is not None

            numeric, good, validity = numeric_validity(
                frame.drop(frame.columns[label_position], axis=1)
            )
            numeric_values = numeric.to_numpy(dtype=np.float64, copy=False)
            missing_rows += validity["missing_rows"]
            missing_cells += validity["missing_cells"]
            infinite_rows += validity["infinite_rows"]
            infinite_cells += validity["infinite_cells"]
            invalid_rows += int((~good).sum())
            good_positions = np.flatnonzero(good)
            if not len(good_positions):
                continue

            valid_raw = frame.iloc[good_positions]
            valid_hashes = pd.util.hash_pandas_object(valid_raw, index=False).to_numpy(
                dtype=np.uint64
            )
            unique_within_valid = np.empty(len(valid_hashes), dtype=bool)
            for i, row_hash in enumerate(valid_hashes):
                key = int(row_hash)
                unique_within_valid[i] = key not in seen_hashes
                if unique_within_valid[i]:
                    seen_hashes.add(key)
                else:
                    duplicate_rows += 1
            selected_positions = good_positions[unique_within_valid]
            if not len(selected_positions):
                continue

            raw_labels = frame.iloc[selected_positions, label_position]
            raw_label_counts.update(raw_labels.astype(str).str.strip().value_counts().to_dict())
            label_chunks.append(
                np.fromiter(
                    (_cic_group_label(label) for label in raw_labels),
                    dtype=np.uint8,
                    count=len(raw_labels),
                )
            )
            # retained_positions refer to the original frame. All retained columns are
            # before Label in the official files, so they also index numeric_values.
            feature_chunks.append(
                numeric_values[selected_positions][:, retained_positions].astype(
                    np.float32, copy=False
                )
            )

    if feature_names is None or not feature_chunks:
        raise ValueError(f"No usable CIC-IDS2017 rows found below {data_dir}")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    if len(features) != len(labels):
        raise AssertionError("Feature/label length mismatch")

    split_indices = row_level_split(labels, protocol=protocol, seed=42)
    train_indices = split_indices.train
    val_indices = split_indices.validation
    test_indices = split_indices.test
    label_encoder = LabelEncoder().fit(labels[train_indices])
    expected_codes = np.arange(len(PAPER_CIC_MULTICLASS_NAMES), dtype=np.uint8)
    if not np.array_equal(label_encoder.classes_, expected_codes):
        raise ValueError(
            "Training split does not contain every CIC class; cannot transform labels consistently"
        )
    labels = label_encoder.transform(labels).astype(np.uint8)
    joblib.dump(label_encoder, cache_dir / "label_encoder.joblib")

    scaler = _fit_minmax_in_chunks(features, train_indices, chunk_size)
    joblib.dump(scaler, cache_dir / "preprocessor.joblib")
    for split, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        _write_scaled_split(
            cache_dir / f"{split}_x.npy", features, indices, scaler, chunk_size
        )
        np.save(cache_dir / f"{split}_y.npy", labels[indices])
        np.save(cache_dir / f"{split}_indices.npy", indices)

    test_counts = np.bincount(labels[test_indices], minlength=9).tolist()
    label_mapping = {
        name: index for index, name in enumerate(PAPER_CIC_MULTICLASS_NAMES)
    }
    preprocessing_report = {
        "original_shape": [raw_rows, 79],
        "normalized_column_names": True,
        "missing_rows": missing_rows,
        "missing_cells": missing_cells,
        "infinite_rows": infinite_rows,
        "infinite_cells": infinite_cells,
        "invalid_union_rows_removed": invalid_rows,
        "exact_duplicate_rows_removed_after_invalid_filter": duplicate_rows,
        "constant_features_removed": sorted(CIC_DROP_FEATURES),
        "constant_feature_count": len(CIC_DROP_FEATURES),
        "final_shape": [int(len(labels)), int(features.shape[1]) + 1],
        "removal_policy": "Invalid and duplicate rows are counted explicitly; first exact duplicate is retained",
        "scaler_fit_split": "training only",
        "label_encoder_fit_split": "training only after documented raw-to-nine-class mapping",
    }
    metadata = {
        "dataset": "cicids2017",
        "protocol": protocol,
        "temporal_window_supported": False,
        "temporal_window_reason": "MachineLearningCVE exports contain no timestamp or session identifier",
        "source_files": [str(path.resolve()) for path in files],
        "source_sha256": {path.name: _sha256(path) for path in files},
        "source_file_order": [path.name for path in files],
        "raw_rows": raw_rows,
        "invalid_rows_removed": invalid_rows,
        "duplicate_rows_removed_after_invalid_filter": duplicate_rows,
        "clean_rows": int(len(labels)),
        "feature_count": int(features.shape[1]),
        "fitted_class_count": int(len(label_encoder.classes_)),
        "feature_names": feature_names,
        "class_names": list(PAPER_CIC_MULTICLASS_NAMES),
        "raw_label_counts_after_cleaning": dict(sorted(raw_label_counts.items())),
        "split": {
            "method": split_indices.method,
            "seed": 42,
            "indices_saved": True,
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(val_indices)),
            "test_rows": int(len(test_indices)),
            "test_class_counts": test_counts,
        },
        "paper_test_class_counts": list(PAPER_CIC_TEST_COUNTS),
        "paper_split_match": protocol == "paper_replication"
        and test_counts == list(PAPER_CIC_TEST_COUNTS),
        "normalization": "MinMaxScaler fitted on training rows only",
        "label_mapping": label_mapping,
        "preprocessing_report": preprocessing_report,
        "deduplication_note": "64-bit pandas row hashes preserve first occurrence; collision probability is negligible",
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_preprocessing_artifacts(
        cache_dir,
        feature_names=feature_names,
        label_mapping=label_mapping,
        report=preprocessing_report,
        mapping_report={
            "dataset": "cicids2017",
            "model_input_policy": "dataset-specific ordered feature list",
            "cross_dataset_feature_mapping": None,
            "reason": "CIC-IDS2017 and NSL-KDD schemas have different semantics; no common feature space is fabricated",
        },
    )
    return load_prepared(cache_dir)


NSL_COLUMNS = (
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
)

NSL_CLASS_NAMES = ("normal", "DoS", "Probe", "R2L", "U2R")
NSL_ATTACK_GROUPS = {
    "DoS": {
        "apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable",
        "smurf", "teardrop", "udpstorm", "worm",
    },
    "Probe": {"ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"},
    "R2L": {
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named",
        "phf", "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient",
        "warezmaster", "xlock", "xsnoop",
    },
    "U2R": {
        "buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm",
    },
}


def _nsl_group_label(label: object) -> int:
    value = str(label).strip().rstrip(".")
    if value == "normal":
        return 0
    for code, name in enumerate(NSL_CLASS_NAMES[1:], start=1):
        if value in NSL_ATTACK_GROUPS[name]:
            return code
    raise ValueError(f"Unmapped NSL-KDD attack label: {value!r}")


def _write_array(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float32)
    np.save(path, array)


def prepare_nsl_kdd(
    data_dir: Path | str,
    cache_dir: Path | str,
    *,
    force: bool = False,
    protocol: ProtocolName = "rigorous_evaluation",
) -> PreparedDataset:
    """Prepare the standard NSL-KDD train/test files as a paper-model extension."""

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if (cache_dir / "metadata.json").exists() and not force:
        return _load_existing_for_protocol(cache_dir, protocol)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = data_dir / "KDDTrain+.txt", data_dir / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"KDDTrain+.txt and KDDTest+.txt are required in {data_dir}")

    train_frame = pd.read_csv(train_path, names=NSL_COLUMNS, header=None)
    test_frame = pd.read_csv(test_path, names=NSL_COLUMNS, header=None)
    train_labels_all = np.fromiter(
        (_nsl_group_label(x) for x in train_frame["label"]),
        dtype=np.uint8,
        count=len(train_frame),
    )
    test_labels = np.fromiter(
        (_nsl_group_label(x) for x in test_frame["label"]),
        dtype=np.uint8,
        count=len(test_frame),
    )
    feature_columns = list(NSL_COLUMNS[:-2])
    categorical = ["protocol_type", "service", "flag"]
    numeric = [name for name in feature_columns if name not in categorical]

    train_indices, val_indices, split_method = official_train_validation_split(
        train_labels_all, protocol=protocol, seed=42
    )
    label_encoder = LabelEncoder().fit(train_labels_all[train_indices])
    expected_codes = np.arange(len(NSL_CLASS_NAMES), dtype=np.uint8)
    if not np.array_equal(label_encoder.classes_, expected_codes):
        raise ValueError(
            "Training split does not contain every NSL-KDD class; cannot transform labels consistently"
        )
    train_labels_all = label_encoder.transform(train_labels_all).astype(np.uint8)
    test_labels = label_encoder.transform(test_labels).astype(np.uint8)
    joblib.dump(label_encoder, cache_dir / "label_encoder.joblib")
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", MinMaxScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train_features_frame = train_frame[feature_columns]
    test_features_frame = test_frame[feature_columns]
    preprocessor.fit(train_features_frame.iloc[train_indices])
    joblib.dump(preprocessor, cache_dir / "preprocessor.joblib")
    transformed_names = preprocessor.get_feature_names_out().tolist()

    _write_array(
        cache_dir / "train_x.npy",
        preprocessor.transform(train_features_frame.iloc[train_indices]),
    )
    _write_array(
        cache_dir / "val_x.npy",
        preprocessor.transform(train_features_frame.iloc[val_indices]),
    )
    _write_array(cache_dir / "test_x.npy", preprocessor.transform(test_features_frame))
    np.save(cache_dir / "train_y.npy", train_labels_all[train_indices])
    np.save(cache_dir / "val_y.npy", train_labels_all[val_indices])
    np.save(cache_dir / "test_y.npy", test_labels)
    np.save(cache_dir / "train_indices.npy", train_indices)
    np.save(cache_dir / "val_indices.npy", val_indices)
    np.save(cache_dir / "test_indices.npy", np.arange(len(test_frame), dtype=np.int64))

    label_mapping = {name: index for index, name in enumerate(NSL_CLASS_NAMES)}
    preprocessing_report = {
        "original_train_shape": [int(len(train_frame)), int(len(NSL_COLUMNS))],
        "original_test_shape": [int(len(test_frame)), int(len(NSL_COLUMNS))],
        "normalized_column_names": True,
        "missing_rows": int(train_frame.isna().any(axis=1).sum() + test_frame.isna().any(axis=1).sum()),
        "infinite_rows": 0,
        "exact_duplicate_rows": int(train_frame.duplicated().sum() + test_frame.duplicated().sum()),
        "removed_rows": 0,
        "constant_training_features": ["num_outbound_cmds"],
        "constant_feature_policy": "retained to preserve the official NSL-KDD schema",
        "numeric_scaler_fit_split": "training only",
        "categorical_encoder_fit_split": "training only with unknown-category handling",
        "label_encoder_fit_split": "training only",
    }
    metadata = {
        "dataset": "nsl-kdd",
        "status": "extension; NSL-KDD was not evaluated in the attached paper",
        "protocol": protocol,
        "temporal_window_supported": False,
        "temporal_window_reason": "NSL-KDD contains neither timestamps nor stable session identifiers",
        "source_files": [str(train_path.resolve()), str(test_path.resolve())],
        "source_sha256": {
            train_path.name: _sha256(train_path),
            test_path.name: _sha256(test_path),
        },
        "official_train_rows": int(len(train_frame)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(val_indices)),
        "test_rows": int(len(test_frame)),
        "feature_count": len(transformed_names),
        "fitted_class_count": int(len(label_encoder.classes_)),
        "feature_names": transformed_names,
        "class_names": list(NSL_CLASS_NAMES),
        "train_class_counts": np.bincount(train_labels_all[train_indices], minlength=5).tolist(),
        "validation_class_counts": np.bincount(train_labels_all[val_indices], minlength=5).tolist(),
        "test_class_counts": np.bincount(test_labels, minlength=5).tolist(),
        "split": {"method": split_method, "indices_saved": True, "seed": 42},
        "normalization": "train-fitted MinMaxScaler for numeric columns; train-fitted one-hot encoding for categorical columns",
        "label_mapping": label_mapping,
        "preprocessing_report": preprocessing_report,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_preprocessing_artifacts(
        cache_dir,
        feature_names=transformed_names,
        label_mapping=label_mapping,
        report=preprocessing_report,
        mapping_report={
            "dataset": "nsl-kdd",
            "paper_status": "extension; not used in the attached paper",
            "input_schema": {
                "raw_features": len(feature_columns),
                "categorical": categorical,
                "numeric_count": len(numeric),
                "transformed_features": len(transformed_names),
            },
            "cross_dataset_feature_mapping": None,
            "reason": "NSL-KDD connection features do not form a defensible one-to-one mapping to CIC-IDS2017 flow statistics",
            "action": "use a dataset-specific fitted transformer and model input dimension",
        },
    )
    return load_prepared(cache_dir)


def reshape_for_model(
    features: np.ndarray,
    model_name: str,
    modeling_mode: ModelingMode = "feature_axis_replication",
) -> np.ndarray:
    """Create the explicit rank-3 tensor required by Conv1D/LSTM."""

    features = np.asarray(features)
    if modeling_mode == "temporal_window":
        if features.ndim != 3:
            raise ValueError(
                "temporal_window mode requires (samples, time_steps, features) input"
            )
        if features.shape[1] < 2:
            raise ValueError("temporal_window mode requires at least two time steps")
        return features
    if modeling_mode != "feature_axis_replication":
        raise ValueError(f"Unknown modeling mode: {modeling_mode}")
    if features.ndim != 2:
        raise ValueError(
            "feature_axis_replication requires (samples, features) input before reshape"
        )

    if model_name in {"cnn", "cnn-lstm"}:
        return features[..., np.newaxis]
    if model_name == "lstm":
        return features[:, np.newaxis, :]
    raise ValueError(f"Unknown model: {model_name}")


def deterministic_subset(length: int, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or maximum >= length:
        return np.arange(length, dtype=np.int64)
    if maximum < 1:
        raise ValueError("Sample limit must be positive")
    return np.random.RandomState(seed).permutation(length)[:maximum]
