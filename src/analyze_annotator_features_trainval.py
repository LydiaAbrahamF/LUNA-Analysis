"""Summarise radiologist feature categories and inter-reader agreement on trainval only.

Annotation rows are matched to the final nodule table by nearest 3D centre within
the same patient. The match is accepted only within 5 mm and the mapped number
of ratings must exactly equal the shared nodule table's n_raters count.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
METADATA_FILE = Path.home() / "Desktop" / "LUNA_Dataset" / "MetadatabyAnnotation.xlsx"
NODULES_FILE = ROOT / "data/processed/nodules.csv"
SPLITS_FILE = ROOT / "data/processed/splits.csv"
OUTPUT_DIR = ROOT / "results/eda"
MATCH_TOLERANCE_MM = 5.0
FEATURE_COLUMNS = {
    "Calcification": "Calcification",
    "InternalStructure": "InternalStructure",
    "Lobulation": "Lobulation",
    "Margin": "Margin",
    "Sphericity": "Sphericity",
    "Spiculation": "Spiculation",
    "Subtlety": "Subtlety",
    "Texture": "Texture",
}


def match_trainval_annotations(annotations: pd.DataFrame,
                               nodules: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    for patient_id, patient_annotations in annotations.groupby("patient_id", sort=False):
        candidates = nodules.loc[nodules["patient_id"] == patient_id]
        if candidates.empty:
            rows.extend({"annotation_row": idx, "voi_name": None, "distance_mm": np.inf}
                        for idx in patient_annotations.index)
            continue
        candidate_coords = candidates[["coordX", "coordY", "coordZ"]].to_numpy(float)
        candidate_ids = candidates["voi_name"].to_numpy()
        for idx, ann in patient_annotations.iterrows():
            point = ann[["coordX", "coordY", "coordZ"]].to_numpy(dtype=float)
            distances = np.linalg.norm(candidate_coords - point, axis=1)
            nearest = int(np.argmin(distances))
            rows.append({
                "annotation_row": idx,
                "voi_name": candidate_ids[nearest],
                "distance_mm": float(distances[nearest]),
            })
    matched = pd.DataFrame(rows).set_index("annotation_row")
    valid = matched.loc[matched["distance_mm"] <= MATCH_TOLERANCE_MM].copy()
    return valid, int(len(matched) - len(valid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_FILE,
        help="Path to MetadatabyAnnotation.xlsx (the file is not tracked in Git).",
    )
    args = parser.parse_args()

    splits = pd.read_csv(SPLITS_FILE)
    trainval_ids = set(splits.loc[splits["split"] == "trainval", "voi_name"])
    trainval_nodules = pd.read_csv(NODULES_FILE)
    trainval_nodules = trainval_nodules.loc[
        trainval_nodules["voi_name"].isin(trainval_ids)
    ].copy()
    if set(trainval_nodules["voi_name"]) != trainval_ids:
        raise ValueError("Nodule table IDs do not exactly match shared trainval IDs")
    if trainval_nodules["voi_name"].duplicated().any():
        raise ValueError("Duplicate trainval VOI IDs")

    # Only annotation rows from trainval patients are loaded into the analysis.
    trainval_patients = set(trainval_nodules["patient_id"])
    annotations = pd.read_excel(args.metadata, sheet_name="Metadata_byAnnotations")
    annotations = annotations.loc[annotations["patient_id"].isin(trainval_patients)].copy()
    annotation_features = list(FEATURE_COLUMNS)
    required = {"patient_id", "coordX", "coordY", "coordZ", *FEATURE_COLUMNS.values()}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    matches, unmatched_count = match_trainval_annotations(annotations, trainval_nodules)
    matched_annotations = annotations.loc[matches.index].copy()
    matched_annotations["voi_name"] = matches["voi_name"]
    matched_annotations["distance_mm"] = matches["distance_mm"]
    expected_counts = trainval_nodules.set_index("voi_name")["n_raters"].astype(int)
    observed_counts = matched_annotations.groupby("voi_name").size()
    count_difference = (
        observed_counts.reindex(expected_counts.index, fill_value=0) - expected_counts
    )
    if (count_difference != 0).any():
        raise ValueError(
            "Annotation-to-nodule matching failed the shared n_raters check for "
            f"{int((count_difference != 0).sum())} trainval nodules"
        )
    if len(matched_annotations) != int(expected_counts.sum()):
        raise ValueError("Matched annotation count does not equal expected trainval ratings")

    label_map = trainval_nodules.set_index("voi_name")["malignant"].astype(int)
    matched_annotations["malignant"] = matched_annotations["voi_name"].map(label_map)
    per_nodule_rows = []
    category_rows = []
    agreement_rows = []

    for voi_name, group in matched_annotations.groupby("voi_name", sort=False):
        label = int(group["malignant"].iloc[0])
        for feature_name, source_column in FEATURE_COLUMNS.items():
            ratings = group[source_column].dropna().astype(str).str.strip()
            if ratings.empty:
                continue
            proportions = ratings.value_counts(normalize=True)
            per_nodule_rows.append({
                "voi_name": voi_name,
                "patient_id": group["patient_id"].iloc[0],
                "malignant": label,
                "feature": feature_name,
                "n_ratings": len(ratings),
                "n_categories": int(ratings.nunique()),
                "unanimous": int(ratings.nunique() == 1),
                "majority_agreement": float(proportions.max()),
            })
            for category in sorted(ratings.unique()):
                category_rows.append({
                    "voi_name": voi_name,
                    "malignant": label,
                    "feature": feature_name,
                    "category": category,
                    "rating_share": float(proportions[category]),
                })
            agreement_rows.append({
                "voi_name": voi_name,
                "malignant": label,
                "feature": feature_name,
                "n_categories": int(ratings.nunique()),
                "unanimous": int(ratings.nunique() == 1),
                "majority_agreement": float(proportions.max()),
            })

    per_nodule = pd.DataFrame(per_nodule_rows)
    category_per_nodule = pd.DataFrame(category_rows)
    agreement_per_nodule = pd.DataFrame(agreement_rows)

    # Equal-weight nodules so nodules with more annotations do not dominate.
    class_sizes = trainval_nodules.groupby("malignant")["voi_name"].nunique()
    category_summary = (
        category_per_nodule.groupby(["feature", "malignant", "category"], as_index=False)
        .agg(
            nodules_with_category=("voi_name", "nunique"),
            sum_nodule_rating_share=("rating_share", "sum"),
        )
    )
    category_summary["class_nodules"] = category_summary["malignant"].map(class_sizes).astype(int)
    category_summary["label"] = category_summary["malignant"].map({0: "benign", 1: "malignant"})
    category_summary["mean_nodule_rating_share"] = (
        category_summary["sum_nodule_rating_share"] / category_summary["class_nodules"]
    )
    category_summary = category_summary.drop(columns="sum_nodule_rating_share")
    category_summary["fraction_nodules_with_category"] = (
        category_summary["nodules_with_category"] / category_summary["class_nodules"]
    )

    agreement_summary = (
        agreement_per_nodule.groupby(["feature", "malignant"], as_index=False)
        .agg(
            n_nodules=("voi_name", "nunique"),
            fraction_unanimous=("unanimous", "mean"),
            median_majority_agreement=("majority_agreement", "median"),
            median_number_of_categories=("n_categories", "median"),
        )
    )
    agreement_summary["label"] = agreement_summary["malignant"].map(
        {0: "benign", 1: "malignant"}
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    category_summary.to_csv(
        OUTPUT_DIR / "annotator_feature_category_distribution_trainval.csv", index=False
    )
    agreement_summary.to_csv(
        OUTPUT_DIR / "annotator_feature_agreement_trainval.csv", index=False
    )
    per_nodule.to_csv(
        OUTPUT_DIR / "annotator_feature_per_nodule_trainval.csv", index=False
    )

    print(f"Trainval nodules: {len(trainval_nodules)}")
    print(f"Trainval annotations matched: {len(matched_annotations)}")
    print(f"Trainval annotation rows outside the 5 mm match tolerance: {unmatched_count}")
    print("Mapped annotation counts exactly match n_raters for every trainval nodule.")
    print("\nInter-reader agreement summary:")
    print(agreement_summary.to_string(index=False))
    print("\nTop category proportions by malignant label:")
    print(category_summary.sort_values(
        ["feature", "malignant", "mean_nodule_rating_share"],
        ascending=[True, True, False],
    ).groupby(["feature", "malignant"]).head(2).to_string(index=False))
    print("\nAggregate CSVs are suitable for sharing; per-nodule output should remain local.")


if __name__ == "__main__":
    main()
