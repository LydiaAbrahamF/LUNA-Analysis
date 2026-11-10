
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path(
    "results/resampling_comparison/features_comparison.csv"
)

OUTPUT_DIR = Path("results/resampling_comparison")
df = pd.read_csv(INPUT_FILE)

feature_cols = [
    col for col in df.columns
    if col.startswith("original_")
    and not col.startswith("original_spacing_")
]

results = []

for nodule_id, group in df.groupby("nodule_id"):

    original = group[
        group["protocol"] == "original_spacing"
    ].iloc[0]

    resampled = group[
        group["protocol"] == "isotropic_1mm"
    ].iloc[0]

    for feature in feature_cols:

        a = original[feature]
        b = resampled[feature]

        abs_change = abs(b - a)

        # Relative change is undefined when baseline is zero
        if abs(a) > 1e-10:
            relative_change = abs_change / abs(a) * 100
        else:
            relative_change = np.nan

        results.append({
            "nodule_id": nodule_id,
            "feature": feature,
            "original_value": a,
            "resampled_value": b,
            "absolute_change": abs_change,
            "relative_change_percent": relative_change
        })

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "feature_changes.csv",
    index=False
)

# Summary across the three samples
summary = (
    comparison.groupby("feature")
    .agg(
        mean_absolute_change=("absolute_change", "mean"),
        max_absolute_change=("absolute_change", "max"),
        mean_relative_change=(
            "relative_change_percent", "mean"
        )
    )
    .sort_values(
        "mean_relative_change",
        ascending=False
    )
)

summary.to_csv(
    OUTPUT_DIR / "feature_change_summary.csv"
)

print("\nTop 15 features by mean relative change:")
print(summary.head(15).to_string())

print("\nTop 15 features by mean absolute change:")
print(
    summary.sort_values(
        "mean_absolute_change",
        ascending=False
    ).head(15).to_string()
)

print("\nFiles saved:")
print(OUTPUT_DIR / "feature_changes.csv")
print(OUTPUT_DIR / "feature_change_summary.csv")
