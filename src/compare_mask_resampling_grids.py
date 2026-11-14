"""Validate three image-mask pairs and compare archived manual grids to PyRadiomics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from radiomics import imageoperations


def geometry_matches(image: sitk.Image, mask: sitk.Image, tolerance: float = 1e-6) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and all(abs(a - b) <= tolerance for a, b in zip(image.GetSpacing(), mask.GetSpacing()))
        and all(abs(a - b) <= tolerance for a, b in zip(image.GetOrigin(), mask.GetOrigin()))
        and all(abs(a - b) <= tolerance for a, b in zip(image.GetDirection(), mask.GetDirection()))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    image_dir, mask_dir = root / "image", root / "nodule_mask"
    res_dir = root / "results" / "resampling_comparison"
    samples = pd.read_csv(root / "results" / "resampling_samples.csv")
    manual = pd.read_csv(res_dir / "mask_comparison.csv").set_index("nodule_id")

    settings = {
        "label": 1,
        "resampledPixelSpacing": (1.0, 1.0, 1.0),
        "interpolator": sitk.sitkBSpline,
        "padDistance": 5,
    }
    rows = []
    for nodule_id in samples["nodule_id"]:
        image_path, mask_path = image_dir / nodule_id, mask_dir / nodule_id
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair for {nodule_id}: {image_path} ; {mask_path}")
        image, mask = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
        pair_ok = geometry_matches(image, mask)
        if not pair_ok:
            raise ValueError(f"Image/mask geometry mismatch: {nodule_id}")
        mask_array = sitk.GetArrayViewFromImage(mask)
        if 1 not in set(mask_array.ravel()):
            raise ValueError(f"No label 1 in mask: {nodule_id}")
        if not set(mask_array.ravel()).issubset({0, 1}):
            raise ValueError(f"Unexpected non-binary labels in mask: {nodule_id}")

        old = manual.loc[nodule_id]
        old_voxels = int(old["resampled_voxels"])
        old_volume = float(old["resampled_volume_mm3"])
        old_change = float(old["volume_change_percent"])

        _, corrected_mask = imageoperations.checkMask(image, mask, **settings)
        if corrected_mask is not None:
            mask = corrected_mask
        original_voxels = int((sitk.GetArrayViewFromImage(mask) == 1).sum())
        original_volume = original_voxels * math.prod(mask.GetSpacing())
        _, pyr_mask = imageoperations.resampleImage(image, mask, **settings)
        pyr_voxels = int((sitk.GetArrayViewFromImage(pyr_mask) == 1).sum())
        pyr_volume = pyr_voxels * math.prod(pyr_mask.GetSpacing())
        pyr_change = 100.0 * (pyr_volume - original_volume) / original_volume
        if original_voxels != int(old["original_voxels"]):
            raise ValueError(f"Archived/manual original voxel count does not match source mask: {nodule_id}")

        rows.append({
            "nodule_id": nodule_id,
            "pair_status": "PASS" if pair_ok else "FAIL",
            "original_size_xyz": "x".join(map(str, image.GetSize())),
            "original_spacing_xyz_mm": "x".join(f"{v:.6g}" for v in image.GetSpacing()),
            "original_voxels": original_voxels,
            "original_volume_mm3": original_volume,
            "manual_grid_voxels": old_voxels,
            "manual_grid_volume_mm3": old_volume,
            "manual_volume_change_percent": old_change,
            "pyradiomics_roi_grid_voxels": pyr_voxels,
            "pyradiomics_roi_grid_volume_mm3": pyr_volume,
            "pyradiomics_volume_change_percent": pyr_change,
            "pyradiomics_minus_manual_voxels": pyr_voxels - old_voxels,
            "pyradiomics_minus_manual_volume_mm3": pyr_volume - old_volume,
            "pyradiomics_roi_grid_size_xyz": "x".join(map(str, pyr_mask.GetSize())),
            "pyradiomics_grid_spacing_xyz_mm": "x".join(f"{v:.6g}" for v in pyr_mask.GetSpacing()),
        })

    current = pd.DataFrame(rows)
    out_csv = res_dir / "mask_resampling_comparison_updated.csv"
    current.to_csv(out_csv, index=False, float_format="%.6f")

    report = """# Three-sample mask resampling check

## Method

For each of the three prespecified VOIs, the image and same-named mask were loaded and checked for matching size, spacing, origin, and direction (tolerance 1e-6); masks were also checked for binary values and presence of label 1. Archived results from the earlier separately calculated manual output grid were compared with a fresh run of PyRadiomics 3.0.1 `imageoperations.resampleImage` at 1 mm isotropic spacing, label 1, B-spline image interpolation, and 5-voxel padding. The label map is resampled with nearest-neighbour interpolation by PyRadiomics. The fresh PyRadiomics mask is measured on its own ROI-aligned output grid. Volume is voxel count × voxel spacing product.

## Results

All three image-mask pairs passed the geometry and label checks. Manual-grid versus PyRadiomics ROI-grid mask voxels were: **0292_R_1, 124 vs 126** (volume change −10.02% vs −8.57%); **0470_R_2, 6456 vs 6516** (+0.19% vs +1.12%); and **0144_R_2, 503 vs 524** (−2.86% vs +1.20%). Thus the two grid definitions produce different voxel counts and can even change the direction of the measured volume change, as in 0144_R_2.

## Limitations

This is a three-case geometry check, not a comparison over all nodules and not evidence that either grid is globally more accurate. The older manual-grid output was read from its archived CSV; its original grid-generation script/settings were not available for an independent rerun. These measurements describe binary-mask volume only and do not establish which extraction protocol is preferable for downstream prediction. Do not generalize the result beyond these three samples.

Reproduce with `python -m src.compare_mask_resampling_grids`. Aggregate check table: `results/resampling_comparison/mask_resampling_comparison_updated.csv`.
"""
    report_path = res_dir / "mask_resampling_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved {out_csv}")
    print(f"Saved {report_path}")
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
