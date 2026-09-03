#!/usr/bin/env python3
"""Compare the primary and replayed evaluations at numeric-output level."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
COMMON_FILES = [
    "per_replica.csv",
    "per_case.csv",
    "summary_by_model.csv",
    "metrics_long.csv",
    "frame_metrics.csv",
    "motion_curves.csv",
    "failures.csv",
]
PROFILE_FILES = {
    "quick": COMMON_FILES,
    "full": COMMON_FILES + ["window_metrics.csv"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_csv(left: Path, right: Path) -> dict:
    result = {
        "file": left.name,
        "primary_sha256": digest(left),
        "replay_sha256": digest(right),
        "byte_identical": left.read_bytes() == right.read_bytes(),
    }
    a = pd.read_csv(left)
    b = pd.read_csv(right)
    result["same_shape"] = a.shape == b.shape
    result["same_columns"] = list(a.columns) == list(b.columns)
    if not (result["same_shape"] and result["same_columns"]):
        result["numeric_identical"] = False
        result["content_identical"] = False
        return result

    numeric_cols = list(a.select_dtypes(include=[np.number]).columns)
    other_cols = [column for column in a.columns if column not in numeric_cols]
    if numeric_cols:
        av = a[numeric_cols].to_numpy(dtype=float)
        bv = b[numeric_cols].to_numpy(dtype=float)
        finite = np.isfinite(av) & np.isfinite(bv)
        result["max_abs_numeric_difference"] = (
            float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0
        )
        numeric_equal = bool(np.array_equal(av, bv, equal_nan=True))
    else:
        result["max_abs_numeric_difference"] = 0.0
        numeric_equal = True

    other_equal = all(
        a[column].fillna("<NA>").astype(str).equals(
            b[column].fillna("<NA>").astype(str)
        )
        for column in other_cols
    )
    result["numeric_identical"] = numeric_equal
    result["non_numeric_identical"] = other_equal
    result["content_identical"] = numeric_equal and other_equal
    return result


audit: dict[str, object] = {"profiles": {}}
all_content_identical = True
for profile, filenames in PROFILE_FILES.items():
    comparisons = [
        compare_csv(ROOT / profile / filename, ROOT / "replay" / profile / filename)
        for filename in filenames
    ]
    profile_identical = all(item["content_identical"] for item in comparisons)
    all_content_identical &= profile_identical
    audit["profiles"][profile] = {
        "all_compared_outputs_identical": profile_identical,
        "files": comparisons,
    }

audit["all_compared_outputs_identical"] = all_content_identical
(ROOT / "reproducibility.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
if not all_content_identical:
    raise SystemExit("Replay differs from the primary evaluation.")
print("All compared numeric evaluation outputs are identical.")
