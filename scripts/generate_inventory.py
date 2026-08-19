"""Generate a checksum/role inventory without changing any source artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def role(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "s11227-025-07802-w.pdf":
        return "source paper"
    if suffix == ".py":
        return "Python source/test/script"
    if suffix in {".yaml", ".yml", ".toml"}:
        return "configuration/environment specification"
    if name == "metadata.json" or "manifest" in name:
        return "dataset/run provenance metadata"
    if "preprocessor" in name or "scaler" in name or "label_encoder" in name:
        return "fitted preprocessing object"
    if name == "model.keras":
        return "saved trained model"
    if name in {"predictions.npy", "probabilities.npy", "y_true.npy"}:
        return "saved evaluation array"
    if "indices" in name and suffix == ".npy":
        return "saved split/subset indices"
    if "confusion_matrix" in name or "classification_report" in name or "metrics" in name:
        return "evaluation artifact"
    if "history" in name or "trace" in name or "convergence" in name:
        return "training/optimization trace"
    if suffix in {".csv", ".txt", ".arff"} and "kdd" in str(path).lower():
        return "NSL-KDD source dataset"
    if suffix == ".csv" and "machinelearning" in str(path).lower():
        return "CIC-IDS2017 source dataset"
    if suffix in {".zip", ".rar"}:
        return "preserved dataset archive"
    if suffix == ".md":
        return "documentation/report"
    return "supporting or generated project artifact"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    output_resolved = args.output.resolve()
    for root in args.root:
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.resolve() == output_resolved
                or any(part in EXCLUDED_PARTS for part in path.parts)
            ):
                continue
            entries.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": role(path),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"file_count": len(entries), "files": entries}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
