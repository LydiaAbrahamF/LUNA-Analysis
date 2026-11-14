"""Summarise HU intensities inside nodule masks using trainval VOIs only.

Each nodule contributes equally to the class-level histogram so large masks do
not dominate. This is descriptive EDA; it does not compare or tune models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(".")
DATA_ROOT = Path.home() / "Desktop" / "LUNA_Dataset"
IMAGE_DIR = DATA_ROOT / "VOIs" / "image"
MASK_DIR = DATA_ROOT / "VOIs" / "nodule_mask"
NODULES_FILE = ROOT / "data/processed/nodules.csv"
SPLITS_FILE = ROOT / "data/processed/splits.csv"
SUMMARY_DIR = ROOT / "results/eda"
PER_NODULE_OUTPUT = SUMMARY_DIR / "hu_per_nodule_trainval.csv"
SUMMARY_OUTPUT = SUMMARY_DIR / "hu_class_summary_trainval.csv"
HISTOGRAM_OUTPUT = SUMMARY_DIR / "hu_histogram_trainval.csv"
HISTOGRAM_EDGES = np.arange(-1500, 3101, 50, dtype=float)


def main() -> None:
    splits = pd.read_csv(SPLITS_FILE)
    trainval_splits = splits.loc[splits["split"] == "trainval"].copy()
    expected_ids = set(trainval_splits["voi_name"])
    if trainval_splits["voi_name"].duplicated().any():
        raise ValueError("Duplicate VOI IDs in the trainval split")

    nodules = pd.read_csv(NODULES_FILE)
    nodules = nodules.loc[nodules["voi_name"].isin(expected_ids)].copy()
    if set(nodules["voi_name"]) != expected_ids or nodules["voi_name"].duplicated().any():
        raise ValueError("Nodule metadata does not exactly match the trainval split")
    if not nodules["malignant"].isin([0, 1]).all():
        raise ValueError("Expected binary malignant labels")

    split_map = trainval_splits.set_index("voi_name")
    nodules = nodules.set_index("voi_name")
    records = []
    per_nodule_histograms = {0: [], 1: []}
    failures = []

    for voi_name in sorted(expected_ids):
        image_path = IMAGE_DIR / voi_name
        mask_path = MASK_DIR / voi_name
        try:
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            if (image.GetSize() != mask.GetSize()
                    or image.GetSpacing() != mask.GetSpacing()
                    or image.GetOrigin() != mask.GetOrigin()
                    or image.GetDirection() != mask.GetDirection()):
                raise ValueError("Image and mask geometry differ")

            hu = sitk.GetArrayViewFromImage(image)
            mask_array = sitk.GetArrayViewFromImage(mask)
            values = np.asarray(hu[mask_array == 1], dtype=np.float64)
            if values.size == 0:
                raise ValueError("Mask contains no label-1 voxels")

            label = int(nodules.loc[voi_name, "malignant"])
            counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
            hist = counts.astype(np.float64) / (counts.sum() * np.diff(HISTOGRAM_EDGES))
            per_nodule_histograms[label].append(hist)

            records.append({
                "voi_name": voi_name,
                "patient_id": split_map.loc[voi_name, "patient_id"],
                "fold": int(split_map.loc[voi_name, "fold"]),
                "malignant": label,
                "mask_voxels": int(values.size),
                "mean_hu": float(np.mean(values)),
                "median_hu": float(np.median(values)),
                "q10_hu": float(np.percentile(values, 10)),
                "q90_hu": float(np.percentile(values, 90)),
                "fraction_hu_gt_0": float(np.mean(values > 0)),
                "fraction_hu_gt_400": float(np.mean(values > 400)),
            })
        except Exception as exc:
            failures.append({"voi_name": voi_name, "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        failure_file = SUMMARY_DIR / "hu_analysis_failures_trainval.csv"
        SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_file, index=False)
        raise RuntimeError(f"{len(failures)} VOIs failed; details saved to {failure_file}")
    if len(records) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} nodules, got {len(records)}")

    per_nodule = pd.DataFrame(records)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    per_nodule.to_csv(PER_NODULE_OUTPUT, index=False)

    summary_rows = []
    for label, label_name in ((0, "benign"), (1, "malignant")):
        group = per_nodule.loc[per_nodule["malignant"] == label]
        summary_rows.append({
            "label": label_name,
            "n_nodules": len(group),
            "median_nodule_median_hu": float(group["median_hu"].median()),
            "q25_nodule_median_hu": float(group["median_hu"].quantile(0.25)),
            "q75_nodule_median_hu": float(group["median_hu"].quantile(0.75)),
            "median_nodule_q10_hu": float(group["q10_hu"].median()),
            "median_nodule_q90_hu": float(group["q90_hu"].median()),
            "median_fraction_hu_gt_0": float(group["fraction_hu_gt_0"].median()),
            "median_fraction_hu_gt_400": float(group["fraction_hu_gt_400"].median()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    centers = (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:]) / 2
    histogram_rows = []
    for label in (0, 1):
        hists = np.vstack(per_nodule_histograms[label])
        label_name = "malignant" if label else "benign"
        for midpoint, density in zip(centers, hists.mean(axis=0)):
            histogram_rows.append({
                "label": label_name,
                "hu_bin_midpoint": float(midpoint),
                "mean_nodule_normalized_density": float(density),
            })
    pd.DataFrame(histogram_rows).to_csv(HISTOGRAM_OUTPUT, index=False)

    print(summary.to_string(index=False))
    print(f"\nProcessed {len(per_nodule)} trainval nodules; failures: {len(failures)}")
    print(f"Aggregate table: {SUMMARY_OUTPUT}")
    print(f"Class histogram data: {HISTOGRAM_OUTPUT}")
    print("Per-nodule HU summary is local and should not be uploaded.")


if __name__ == "__main__":
    main()
