"""Summarize successful metric files and record failures explicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ids_repro.statistics import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    values = []
    failed = []
    individual = []
    for path in args.paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = float(payload["metrics"][args.metric])
            values.append(value)
            individual.append({"path": str(path), "value": value})
        except Exception as error:
            failed.append({"path": str(path), "error": str(error)})
    payload = {
        "metric": args.metric,
        "individual_runs": individual,
        "successful_runs": len(values),
        "failed_runs": len(failed),
        "failures": failed,
        "summary": summarize(values) if len(values) >= 2 else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
