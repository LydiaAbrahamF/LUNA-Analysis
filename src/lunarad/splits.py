"""Patient-level split: a sealed test set plus cross-validation folds.

The split is made ONCE (pipeline/02_make_split.py) and then locked. Patients, not
nodules, are the unit, so no patient ever appears in two splits or two folds.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_INNER, N_OUTER, SEED

REQUIRED = ["voi_name", "patient_id", "malignant", "thickness"]


def build_split(nodules, seed=SEED, n_outer=N_OUTER, n_inner=N_INNER):
    """Return a table [voi_name, patient_id, split, fold] in the same row order as nodules.

    split is "trainval" or "test". fold is 0..n_inner-1 for trainval and -1 for test.
    Strata combine the label and the slice-thickness group, so both stay balanced.
    """
    missing = [c for c in REQUIRED if c not in nodules.columns]
    if missing:
        raise ValueError(f"nodules table is missing columns: {missing}")
    nodules = nodules.reset_index(drop=True)
    if nodules[REQUIRED].isna().any().any():
        raise ValueError("missing values in voi_name, patient_id, malignant or thickness")

    strata = nodules["malignant"].astype(str) + "_" + nodules["thickness"].astype(str)
    groups = nodules["patient_id"]

    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(nodules, strata, groups))
    split = np.full(len(nodules), "trainval", dtype=object)
    split[test_idx] = "test"
    fold = np.full(len(nodules), -1, dtype=int)

    trainval = nodules.iloc[trainval_idx]
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(inner.split(trainval, strata.iloc[trainval_idx],
                                                 trainval["patient_id"])):
        fold[trainval_idx[val_idx]] = k

    result = pd.DataFrame({
        "voi_name": nodules["voi_name"],
        "patient_id": nodules["patient_id"],
        "split": split,
        "fold": fold,
    })
    validate_split(nodules, result, n_inner)
    return result


def validate_split(nodules, splits, n_inner=N_INNER):
    """Raise ValueError if the split breaks any rule. Call it before every use."""
    if nodules["voi_name"].duplicated().any() or splits["voi_name"].duplicated().any():
        raise ValueError("duplicate voi_name")
    left = nodules.set_index("voi_name").sort_index()
    right = splits.set_index("voi_name").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("nodules and splits do not contain the same voi_name values")
    if not left["patient_id"].astype(str).equals(right["patient_id"].astype(str)):
        raise ValueError("nodules and splits disagree on patient_id")
    if not set(right["split"]) <= {"trainval", "test"} or "test" not in set(right["split"]):
        raise ValueError("split must contain 'trainval' and 'test' only")
    if (right.groupby("patient_id")["split"].nunique() > 1).any():
        raise ValueError("a patient appears in both trainval and test")
    test = right[right["split"] == "test"]
    trainval = right[right["split"] == "trainval"]
    if trainval.empty or (test["fold"] != -1).any():
        raise ValueError("test rows must have fold == -1 and trainval must not be empty")
    if set(trainval["fold"]) != set(range(n_inner)):
        raise ValueError(f"trainval folds must be exactly 0..{n_inner - 1}")
    if (trainval.groupby("patient_id")["fold"].nunique() > 1).any():
        raise ValueError("a patient spans several trainval folds")


def summarize_split(nodules, splits):
    """Table of size, malignant share and patient count per split and fold."""
    merged = splits.merge(nodules[["voi_name", "malignant"]], on="voi_name")
    return (merged.groupby(["split", "fold"])
            .agg(nodules=("voi_name", "size"),
                 malignant_share=("malignant", "mean"),
                 patients=("patient_id", "nunique"))
            .round(3))


def file_hash(path):
    """SHA-256 of a file; record it when you lock the split."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
