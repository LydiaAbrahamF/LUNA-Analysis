"""Describe trainval-only univariate feature-ranking stability across shared folds.

This is a screening summary, not a model-performance comparison. Each fold's
ranking uses only the other four trainval folds. The held-out test set is never
read by this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer


ROOT = Path(".")
FEATURES_FILE = ROOT / "results/radiomics_trainval/features_trainval.csv"
NODULES_FILE = ROOT / "data/processed/nodules.csv"
SPLITS_FILE = ROOT / "data/processed/splits.csv"
OUTPUT_FILE = ROOT / "results/radiomics_trainval/feature_selection_stability.csv"
PROTOCOLS = ("original_spacing", "isotropic_1mm")
FOLDS = (0, 1, 2, 3, 4)
TOP_K = 10
META = {
    "voi_name", "patient_id", "malignant", "fold", "protocol",
    "original_spacing_x", "original_spacing_y", "original_spacing_z",
}


def main() -> None:
    features = pd.read_csv(FEATURES_FILE)
    nodules = pd.read_csv(NODULES_FILE).set_index("voi_name")
    splits = pd.read_csv(SPLITS_FILE).set_index("voi_name")
    feature_columns = [column for column in features.columns if column not in META]
    if len(feature_columns) != 107:
        raise ValueError(f"Expected 107 Radiomics features; found {len(feature_columns)}")

    expected_ids = set(splits.index[splits["split"] == "trainval"])
    if set(features["voi_name"]) != expected_ids:
        raise ValueError("Feature table IDs do not match the shared trainval split")
    if "test" in set(splits.loc[features["voi_name"], "split"]):
        raise ValueError("Test rows found in the feature table")

    features["malignant"] = features["voi_name"].map(nodules["malignant"]).astype(int)
    features["patient_id"] = features["voi_name"].map(splits["patient_id"])
    features["fold"] = features["voi_name"].map(splits["fold"]).astype(int)
    if features.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ValueError("A patient appears in multiple trainval folds")

    rows = []
    for protocol in PROTOCOLS:
        table = features.loc[features["protocol"] == protocol].copy()
        if table["voi_name"].nunique() != len(expected_ids):
            raise ValueError(f"Incomplete paired feature rows for {protocol}")
        for fold in FOLDS:
            fit = table.loc[table["fold"] != fold]
            x = SimpleImputer(strategy="median").fit_transform(fit[feature_columns])
            scores, _ = f_classif(x, fit["malignant"].to_numpy())
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            order = np.argsort(-scores, kind="stable")[:TOP_K]
            for rank, index in enumerate(order, start=1):
                rows.append({
                    "protocol": protocol,
                    "validation_fold": fold,
                    "fit_nodules": len(fit),
                    "rank": rank,
                    "feature": feature_columns[index],
                    "f_score": float(scores[index]),
                })

    ranked = pd.DataFrame(rows)
    stability = (
        ranked.groupby(["protocol", "feature"], as_index=False)
        .agg(
            folds_in_top10=("validation_fold", "nunique"),
            mean_rank_when_selected=("rank", "mean"),
            mean_f_score=("f_score", "mean"),
        )
        .sort_values(["protocol", "folds_in_top10", "mean_rank_when_selected"],
                     ascending=[True, False, True])
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    stability.to_csv(OUTPUT_FILE, index=False)
    print(stability.groupby("protocol").head(15).to_string(index=False))
    print(f"\nSaved trainval-only ranking stability to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
