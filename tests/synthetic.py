"""Tiny fake LUNA16 dataset in the real folder layout, for tests. Not real medical data."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def make_dataset(root, n_patients=60, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    image_dir = root / "VOIs" / "VOIs" / "image"
    mask_dir = root / "VOIs" / "VOIs" / "nodule_mask"
    for folder in (image_dir, mask_dir, root / "CT" / "CT" / "image"):
        folder.mkdir(parents=True, exist_ok=True)

    rows, annotation = [], 0
    for patient in range(1, n_patients + 1):
        patient_id = f"LIDC-IDRI-{patient:04d}"
        sx = round(float(rng.uniform(0.55, 0.95)), 3)
        sz = float(rng.choice([0.625, 1.25, 2.5]))
        for nodule_id in range(1, int(rng.integers(1, 3)) + 1):
            diameter = float(np.clip(rng.lognormal(2.0, 0.45), 3.3, 30.0))
            coords = rng.uniform(-100, 100, 3)
            for _ in range(int(rng.choice([3, 4]))):
                annotation += 1
                rating = int(np.clip(round(1 + diameter / 9 + rng.normal(0, 1.0)), 1, 5))
                rows.append({
                    "patient_id": patient_id, "nodule_id": nodule_id,
                    "coordX": coords[0], "coordY": coords[1], "coordZ": coords[2],
                    "diameter_mm": diameter, "bboxLowX": 0.0, "bboxHighX": diameter / sx,
                    "bboxLowZ": 0.0, "bboxHighZ": diameter / sz,
                    "Malignancy_value": rating, "anotation_id": annotation})

            nx, ny, nz = 28, 28, 10
            grid = np.indices((nx, ny, nz)).astype(float)
            rx = float(np.clip(diameter / (2 * sx), 2.0, 11.0))
            rz = float(np.clip(diameter / (2 * sz), 1.5, 4.0))
            ellipsoid = (((grid[0] - nx / 2) / rx) ** 2 + ((grid[1] - ny / 2) / rx) ** 2
                         + ((grid[2] - nz / 2) / rz) ** 2) <= 1.0
            volume = rng.normal(-650, 60, (nx, ny, nz)).astype("float32")
            volume[ellipsoid] += rng.normal(500, 80, int(ellipsoid.sum())).astype("float32")
            affine = np.diag([sx, sx, sz, 1.0])
            name = f"{patient_id}_R_{nodule_id}.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine), image_dir / name)
            nib.save(nib.Nifti1Image(ellipsoid.astype("uint8"), affine), mask_dir / name)

    pd.DataFrame(rows).to_excel(root / "MetadatabyAnnotation.xlsx", index=False)
    return root
