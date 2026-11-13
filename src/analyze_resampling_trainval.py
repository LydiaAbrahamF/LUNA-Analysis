"""Summarize paired resampling changes using trainval features only."""

from pathlib import Path

import pandas as pd


INPUT = Path("results/radiomics_trainval/features_trainval.csv")
OUTPUT = Path("results/radiomics_trainval/feature_change_summary.csv")
META = {
    "voi_name", "patient_id", "malignant", "fold", "protocol",
    "original_spacing_x", "original_spacing_y", "original_spacing_z",
}


def main() -> None:
    df = pd.read_csv(INPUT)
    features = [c for c in df.columns if c not in META]
    a = df.loc[df.protocol == "original_spacing"].set_index("voi_name").sort_index()
    b = df.loc[df.protocol == "isotropic_1mm"].set_index("voi_name").sort_index()
    if not a.index.equals(b.index) or not a.index.is_unique:
        raise ValueError("Expected one paired original/1 mm feature row per trainval nodule")

    difference = (b[features] - a[features]).abs()
    baseline = a[features].abs()
    relative = difference.where(baseline > 1e-10) / baseline.where(baseline > 1e-10) * 100
    summary = pd.DataFrame({
        "feature": features,
        "mean_absolute_change": difference.mean().to_numpy(),
        "median_absolute_change": difference.median().to_numpy(),
        "max_absolute_change": difference.max().to_numpy(),
        "fraction_nodules_changed": (difference > 1e-12).mean().to_numpy(),
        "n_valid_relative_change": relative.count().to_numpy(),
        "median_relative_change_percent": relative.median().to_numpy(),
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(f"Summarized {len(features)} features across {len(a)} paired trainval nodules")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
