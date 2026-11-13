"""Validate the shared tables and extract A/B Radiomics features on one split.

Defaults to trainval. Use --split test only for a pre-specified final evaluation.
This script performs feature extraction and QC only; it does not train models
or select features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


PROTOCOLS = {
    "original_spacing": None,
    "isotropic_1mm": (1.0, 1.0, 1.0),
}
TABLE_COLUMNS = {"voi_name", "patient_id", "split", "fold"}
NODULE_COLUMNS = TABLE_COLUMNS | {"malignant", "n_high"}


def validate_tables(nodules: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    missing_n = NODULE_COLUMNS - set(nodules.columns)
    missing_s = TABLE_COLUMNS - set(splits.columns)
    if missing_n or missing_s:
        raise ValueError(f"Missing columns; nodules={sorted(missing_n)}, splits={sorted(missing_s)}")
    if nodules["voi_name"].duplicated().any() or splits["voi_name"].duplicated().any():
        raise ValueError("Duplicate voi_name found in nodules.csv or splits.csv")

    left = nodules.set_index("voi_name").sort_index()
    right = splits.set_index("voi_name").sort_index()
    if not left.index.equals(right.index):
        only_nodules = left.index.difference(right.index).tolist()[:5]
        only_splits = right.index.difference(left.index).tolist()[:5]
        raise ValueError(f"VOI ID mismatch; only nodules={only_nodules}, only splits={only_splits}")
    for col in ("patient_id", "split", "fold"):
        if not left[col].astype(str).equals(right[col].astype(str)):
            raise ValueError(f"nodules.csv and splits.csv disagree on {col}")

    patient_splits = splits.groupby("patient_id")["split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError("At least one patient occurs in more than one split")
    trainval = nodules.loc[nodules["split"] == "trainval"].copy()
    test = nodules.loc[nodules["split"] == "test"]
    if trainval.empty or test.empty:
        raise ValueError("Expected non-empty trainval and test splits")
    if set(trainval["fold"].astype(str)) - {"0", "1", "2", "3", "4"}:
        raise ValueError("Expected non-empty trainval rows with folds 0–4")
    if set(test["fold"].astype(str)) != {"-1"}:
        raise ValueError("Expected test rows to use fold=-1")
    if (trainval.groupby("patient_id")["fold"].nunique() > 1).any():
        raise ValueError("At least one patient spans multiple trainval folds")

    malignant = pd.to_numeric(nodules["malignant"], errors="raise").astype(int)
    n_high = pd.to_numeric(nodules["n_high"], errors="raise").astype(int)
    if not malignant.isin([0, 1]).all():
        raise ValueError("malignant must be binary")
    if not (malignant == (n_high >= 2).astype(int)).all():
        raise ValueError("malignant does not match the confirmed rule n_high >= 2")
    return nodules.copy()


def extract_one(image_path: Path, mask_path: Path, protocol: str) -> dict:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if (image.GetSize() != mask.GetSize()
            or image.GetSpacing() != mask.GetSpacing()
            or image.GetOrigin() != mask.GetOrigin()
            or image.GetDirection() != mask.GetDirection()):
        raise ValueError("Image/mask geometry mismatch")
    if 1 not in set(sitk.GetArrayViewFromImage(mask).ravel()):
        raise ValueError("Label 1 is absent from the mask")

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        normalize=False,
        resampledPixelSpacing=PROTOCOLS[protocol],
        interpolator=sitk.sitkBSpline,
        force2D=False,
        label=1,
    )
    result = extractor.execute(image, mask, label=1)
    features = {k: float(v) for k, v in result.items() if k.startswith("original_")}
    return {
        "original_spacing_x": image.GetSpacing()[0],
        "original_spacing_y": image.GetSpacing()[1],
        "original_spacing_z": image.GetSpacing()[2],
        **features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Desktop" / "LUNA_Dataset")
    parser.add_argument("--nodules", type=Path, default=Path("data/processed/nodules.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--split", choices=("trainval", "test"), default="trainval")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocols", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS))
    args = parser.parse_args()

    nodules = pd.read_csv(args.nodules)
    splits = pd.read_csv(args.splits)
    validated = validate_tables(nodules, splits)
    selected = validated.loc[validated["split"] == args.split].copy()
    if args.output_dir is None:
        args.output_dir = Path(f"results/radiomics_{args.split}")
    image_dir = args.data_dir / "VOIs" / "image"
    mask_dir = args.data_dir / "VOIs" / "nodule_mask"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for row in selected.itertuples(index=False):
        image_path = image_dir / row.voi_name
        mask_path = mask_dir / row.voi_name
        for protocol in args.protocols:
            try:
                values = extract_one(image_path, mask_path, protocol)
                records.append({
                    "voi_name": row.voi_name,
                    "patient_id": row.patient_id,
                    "malignant": int(row.malignant),
                    "fold": int(row.fold),
                    "protocol": protocol,
                    **values,
                })
            except Exception as exc:  # keep batch progress and write an auditable error log
                failures.append({
                    "voi_name": row.voi_name,
                    "protocol": protocol,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    pd.DataFrame(records).to_csv(args.output_dir / f"features_{args.split}.csv", index=False)
    pd.DataFrame(failures, columns=["voi_name", "protocol", "error_type", "error"]).to_csv(
        args.output_dir / "extraction_failures.csv", index=False
    )
    print(f"{args.split} nodules: {len(selected)}")
    print(f"successful extraction rows: {len(records)}")
    print(f"failed extraction rows: {len(failures)}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
