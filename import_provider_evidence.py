#!/usr/bin/env python3
"""Validate QPU/service traces for later CAPO calibration and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "job_id", "provider", "backend", "submitted_at", "started_at",
    "completed_at", "status", "shots", "compiled_depth", "compiled_twoq",
    "calibration_time", "twoq_error", "readout_error",
}


def validate(path: Path) -> dict:
    data = pd.read_csv(path)
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError(f"missing provider-evidence columns: {missing}")
    if data.empty or data["job_id"].isna().any():
        raise ValueError("provider evidence must contain non-null job IDs")
    for column in ("submitted_at", "started_at", "completed_at", "calibration_time"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    if (data["started_at"] < data["submitted_at"]).any() or \
       (data["completed_at"] < data["started_at"]).any():
        raise ValueError("provider timestamps are not monotone")
    return {"evidence_label": "measured_provider", "rows": len(data),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "columns": list(data.columns)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = validate(args.csv)
    output = json.dumps(manifest, indent=2)
    if args.manifest:
        args.manifest.write_text(output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
