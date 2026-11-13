"""Compare mask volume on the exact ROI-aligned grid used by PyRadiomics."""

from pathlib import Path
import argparse
import math

import pandas as pd
import SimpleITK as sitk
from radiomics import imageoperations


def mask_stats(mask: sitk.Image, label: int) -> tuple[int, float]:
    voxels = int((sitk.GetArrayViewFromImage(mask) == label).sum())
    volume_mm3 = voxels * math.prod(mask.GetSpacing())
    return voxels, float(volume_mm3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Desktop" / "LUNA_Dataset",
        help="Dataset root containing VOIs/image and VOIs/nodule_mask",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("results/resampling_samples.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/resampling_comparison/mask_comparison.csv"))
    args = parser.parse_args()

    if (args.data_dir / "VOIs" / "image").is_dir():
        image_dir = args.data_dir / "VOIs" / "image"
        mask_dir = args.data_dir / "VOIs" / "nodule_mask"
    else:
        image_dir = args.data_dir / "image"
        mask_dir = args.data_dir / "nodule_mask"
    samples = pd.read_csv(args.samples)

    settings = {
        "label": 1,
        "resampledPixelSpacing": (1.0, 1.0, 1.0),
        "interpolator": sitk.sitkBSpline,
        "padDistance": 5,
    }
    records = []

    for nodule_id in samples["nodule_id"]:
        image_path = image_dir / nodule_id
        mask_path = mask_dir / nodule_id
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing paired image or mask for {nodule_id}: "
                f"{image_path} ; {mask_path}"
            )

        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        _, corrected_mask = imageoperations.checkMask(image, mask, **settings)
        if corrected_mask is not None:
            mask = corrected_mask

        original_voxels, original_volume = mask_stats(mask, settings["label"])
        # PyRadiomics derives a cropped grid from the ROI. It uses the selected
        # interpolator for the image and nearest-neighbor for the label map.
        resampled_image, resampled_mask = imageoperations.resampleImage(
            image, mask, **settings
        )
        resampled_voxels, resampled_volume = mask_stats(
            resampled_mask, settings["label"]
        )

        records.append(
            {
                "nodule_id": nodule_id,
                "original_voxels": original_voxels,
                "resampled_voxels": resampled_voxels,
                "original_volume_mm3": original_volume,
                "resampled_volume_mm3": resampled_volume,
                "volume_change_percent": 100 * (resampled_volume - original_volume) / original_volume,
                "resampled_size_xyz": "x".join(map(str, resampled_mask.GetSize())),
                "resampled_spacing_xyz": "x".join(map(str, resampled_mask.GetSpacing())),
                "resampled_origin_xyz": "x".join(map(str, resampled_mask.GetOrigin())),
                "resampled_direction": ";".join(map(str, resampled_mask.GetDirection())),
                "resampled_image_size_xyz": "x".join(map(str, resampled_image.GetSize())),
            }
        )

    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
