"""Stage 5: how fragile are the radiomics features?

1. Ranking stability: which features are top-ranked in every CV fold (trainval only).
2. Resampling change: how much each feature moves between original spacing and 1 mm.
3. Mask volume change: how much the nodule mask volume changes when resampled to 1 mm.

Writes three tables to results/tables/. Never reads the sealed test labels.
"""
import argparse
import math

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import imageoperations
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from lunarad import config as cfg
from lunarad.features import feature_columns, load_feature_table


def ranking_stability(features, nodules, splits):
    meta = nodules[["voi_name", "malignant"]].merge(splits[["voi_name", "split", "fold"]],
                                                    on="voi_name")
    trainval = features.merge(meta, on="voi_name")
    trainval = trainval[trainval["split"] == "trainval"]
    columns = feature_columns(trainval)
    rows = []
    for protocol in cfg.PROTOCOLS:
        table = trainval[trainval["protocol"] == protocol]
        for fold in sorted(table["fold"].unique()):
            fit = table[table["fold"] != fold]
            x = SimpleImputer(strategy="median").fit_transform(fit[columns])
            scores, _ = f_classif(x, fit["malignant"].to_numpy())
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            for rank, index in enumerate(np.argsort(-scores, kind="stable")[:cfg.TOP_K], start=1):
                rows.append({"protocol": protocol, "validation_fold": fold, "rank": rank,
                             "feature": columns[index], "f_score": float(scores[index])})
    ranked = pd.DataFrame(rows)
    return (ranked.groupby(["protocol", "feature"], as_index=False)
            .agg(folds_in_top=("validation_fold", "nunique"),
                 mean_rank_when_selected=("rank", "mean"),
                 mean_f_score=("f_score", "mean"))
            .sort_values(["protocol", "folds_in_top", "mean_rank_when_selected"],
                         ascending=[True, False, True]))


def resampling_change(features):
    """Paired change per feature between the two protocols. Label-free, so all nodules."""
    columns = feature_columns(features)
    a = features[features["protocol"] == "original_spacing"].set_index("voi_name").sort_index()
    b = features[features["protocol"] == "isotropic_1mm"].set_index("voi_name").sort_index()
    if not a.index.equals(b.index):
        raise ValueError("expected one row per nodule in each protocol")
    difference = (b[columns] - a[columns]).abs()
    baseline = a[columns].abs()
    relative = difference.where(baseline > 1e-10) / baseline.where(baseline > 1e-10) * 100
    return pd.DataFrame({
        "feature": columns,
        "mean_absolute_change": difference.mean().to_numpy(),
        "median_absolute_change": difference.median().to_numpy(),
        "fraction_nodules_changed": (difference > 1e-12).mean().to_numpy(),
        "median_relative_change_percent": relative.median().to_numpy(),
    })


def mask_volume_change(nodules, n, seed=cfg.SEED):
    """Mask volume before and after PyRadiomics' own 1 mm resampling, for a random sample."""
    settings = {"label": 1, "resampledPixelSpacing": (1.0, 1.0, 1.0),
                "interpolator": sitk.sitkBSpline, "padDistance": 5}

    def stats(mask):
        voxels = int((sitk.GetArrayViewFromImage(mask) == 1).sum())
        return voxels, voxels * math.prod(mask.GetSpacing())

    rows = []
    for name in nodules.sample(min(n, len(nodules)), random_state=seed)["voi_name"]:
        image = sitk.ReadImage(str(cfg.VOI_IMAGE / name))
        mask = sitk.ReadImage(str(cfg.VOI_MASK / name))
        _, corrected = imageoperations.checkMask(image, mask, **settings)
        if corrected is not None:
            mask = corrected
        voxels_before, volume_before = stats(mask)
        _, resampled = imageoperations.resampleImage(image, mask, **settings)
        voxels_after, volume_after = stats(resampled)
        rows.append({"voi_name": name, "voxels_before": voxels_before,
                     "voxels_after": voxels_after, "volume_before_mm3": volume_before,
                     "volume_after_mm3": volume_after,
                     "volume_change_percent": 100 * (volume_after - volume_before) / volume_before})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=int, default=30,
                        help="nodules sampled for the mask volume check (0 to skip)")
    args = parser.parse_args()

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    features = load_feature_table(cfg.FEATURES_CSV)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    stability = ranking_stability(features, nodules, splits)
    stability.to_csv(cfg.TABLES / "feature_ranking_stability.csv", index=False)
    print(stability.groupby("protocol").head(10).to_string(index=False))

    resampling_change(features).to_csv(cfg.TABLES / "resampling_feature_change.csv", index=False)
    if args.masks:
        volumes = mask_volume_change(nodules, args.masks)
        volumes.to_csv(cfg.TABLES / "mask_volume_change.csv", index=False)
        print(volumes["volume_change_percent"].describe().round(2))
    print(f"wrote tables to {cfg.TABLES}")


if __name__ == "__main__":
    main()
