"""Compare the planned 32 mm and 40 mm fields of view on trainval VOIs only.

The 40 mm view uses 80 x 80 pixels to preserve the v1 physical pixel size
(0.5 mm/pixel). This is a geometry/padding audit, not a classifier comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.preprocess import PreprocessConfig, preprocess_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--voi-root", type=Path,
                    default=Path.home() / "Desktop" / "LUNA_Dataset" / "VOIs")
    args = ap.parse_args()
    root = args.project_root.resolve()
    proc = root / "data" / "processed"
    out = root / "results" / "preprocessing_window_comparison_trainval.csv"

    index = pd.read_csv(proc / "patches_v1_trainval_index.csv")
    splits = pd.read_csv(proc / "splits.csv")
    if set(index["split"]) != {"trainval"}:
        raise ValueError("Only trainval patch rows are allowed")
    expected = set(splits.loc[splits["split"] == "trainval", "voi_name"])
    if set(index["voi_name"]) != expected or index["voi_name"].duplicated().any():
        raise ValueError("v1 index does not match the shared trainval split exactly")
    X32 = np.load(proc / "patches_v1_trainval_X.npy", mmap_mode="r")
    M32 = np.load(proc / "patches_v1_trainval_mask.npy", mmap_mode="r")
    if X32.shape != (len(index), 1, 64, 64) or M32.shape != (len(index), 64, 64):
        raise ValueError("Unexpected v1 patch array shape")
    meta = pd.read_csv(proc / "nodules.csv")
    index = index.merge(meta[["voi_name", "diameter_mm", "sx"]], on="voi_name",
                        validate="one_to_one")
    cfg40 = PreprocessConfig(name="window40", window_mm=40.0, out_px=80)
    image_dir, mask_dir = args.voi_root / "image", args.voi_root / "nodule_mask"
    padding40 = np.zeros(len(index), dtype=float)
    edge40 = np.zeros(len(index), dtype=bool)
    added_mask_pixels = np.zeros(len(index), dtype=int)
    centre_images_equal = np.zeros(len(index), dtype=bool)
    centre_masks_equal = np.zeros(len(index), dtype=bool)

    for i, name in enumerate(index["voi_name"]):
        img = nib.load(str(image_dir / name))
        mask_img = nib.load(str(mask_dir / name))
        patch = preprocess_images(img, mask_img, cfg40)
        mask = patch.mask
        padding40[i] = patch.pad_fraction
        edge40[i] = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
        centre_mask = mask[8:72, 8:72]
        added_mask_pixels[i] = int(mask.sum() - centre_mask.sum())
        centre_masks_equal[i] = np.array_equal(centre_mask, M32[i])
        centre_images_equal[i] = np.array_equal(patch.x[:, 8:72, 8:72], X32[i])
        if (i + 1) % 100 == 0 or i + 1 == len(index):
            print(f"{i + 1}/{len(index)} trainval VOIs")

    padding32 = index["pad_fraction"].to_numpy(dtype=float)
    y = index["malignant"].to_numpy(dtype=int)
    edge32 = np.any(np.concatenate([M32[:, 0, :], M32[:, -1, :],
                                    M32[:, :, 0], M32[:, :, -1]], axis=1), axis=1)
    rows = []
    for name, window, size, pad, edge in (
        ("v1_32mm", 32.0, "64x64", padding32, edge32),
        ("window40", 40.0, "80x80", padding40, edge40),
    ):
        rows.append({
            "protocol": name,
            "n_trainval": len(index),
            "window_mm": window,
            "patch_size": size,
            "pixel_spacing_mm": 0.5,
            "any_padding_fraction": float(np.mean(pad > 0)),
            "median_padding_fraction": float(np.median(pad)),
            "q90_padding_fraction": float(np.quantile(pad, 0.9)),
            "max_padding_fraction": float(pad.max()),
            "pad_auc_malignant_raw_direction": float(roc_auc_score(y, pad)),
            "spearman_pad_vs_sx": float(spearmanr(pad, index["sx"]).statistic),
            "spearman_pad_vs_diameter_mm": float(spearmanr(pad, index["diameter_mm"]).statistic),
            "mask_edge_fraction": float(np.mean(edge)),
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"40 mm central image crop pixel-identical to v1: {centre_images_equal.mean():.1%}")
    print(f"40 mm central mask crop identical to v1: {centre_masks_equal.mean():.1%}")
    print(f"40 mm masks with pixels outside the central 32 mm: {np.count_nonzero(added_mask_pixels)} / {len(index)}")
    print(f"Saved aggregate comparison: {out}")


if __name__ == "__main__":
    main()
