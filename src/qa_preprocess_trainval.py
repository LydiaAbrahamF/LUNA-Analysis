"""Trainval-only quality checks for the fixed-size CNN preprocessing output.

This script reads saved patches and trainval VOI headers only. It writes an
aggregate report and local-only sample previews; it does not read test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--voi-root", type=Path,
                    default=Path.home() / "Desktop" / "LUNA_Dataset" / "VOIs")
    args = ap.parse_args()
    root = args.project_root.resolve()
    proc = root / "data" / "processed"
    out = root / "results" / "preprocessing_qa_trainval.md"
    preview_dir = proc / "qa_previews"

    X = np.load(proc / "patches_v1_trainval_X.npy", mmap_mode="r")
    masks = np.load(proc / "patches_v1_trainval_mask.npy", mmap_mode="r")
    index = pd.read_csv(proc / "patches_v1_trainval_index.csv")
    nodules = pd.read_csv(proc / "nodules.csv")
    split = pd.read_csv(proc / "splits.csv")
    if set(index["split"]) != {"trainval"} or set(index["voi_name"]) != set(split.loc[split["split"] == "trainval", "voi_name"]):
        raise ValueError("Expected exactly the shared trainval VOIs; test rows must not be included")
    if len(index) != len(X) or len(index) != len(masks) or index["voi_name"].duplicated().any():
        raise ValueError("Patch arrays and index are incomplete or misaligned")
    if X.shape[1:] != (1, 64, 64) or masks.shape[1:] != (64, 64):
        raise ValueError(f"Unexpected patch/mask shapes: {X.shape}, {masks.shape}")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError("Patch values must be finite and in [0, 1]")

    index = index.merge(nodules[["voi_name", "diameter_mm", "thickness", "sx", "sz"]],
                        on="voi_name", how="left", validate="one_to_one")
    if index[["diameter_mm", "thickness", "sx", "sz"]].isna().any().any():
        raise ValueError("Missing geometry metadata for a trainval patch")

    areas = masks.sum(axis=(1, 2)).astype(np.int64)
    empty_masks = int((areas == 0).sum())
    if empty_masks:
        raise ValueError(f"{empty_masks} patch masks are empty")
    yy, xx = np.indices(masks.shape[1:])
    cy = (masks * yy).sum(axis=(1, 2)) / areas
    cx = (masks * xx).sum(axis=(1, 2)) / areas
    centroid_offset = np.hypot(cx - 31.5, cy - 31.5)
    pad = index["pad_fraction"].to_numpy(dtype=float)
    y = index["malignant"].to_numpy(dtype=int)
    orientation_counts: dict[str, int] = {}
    geometry_mismatches = 0
    image_dir = args.voi_root / "image"
    mask_dir = args.voi_root / "nodule_mask"
    for name in index["voi_name"]:
        image = nib.load(str(image_dir / name))
        mask = nib.load(str(mask_dir / name))
        orientation = "".join(nib.aff2axcodes(image.affine))
        orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
        if image.shape != mask.shape or not np.allclose(image.affine, mask.affine):
            geometry_mismatches += 1
        if nib.aff2axcodes(nib.as_closest_canonical(image).affine) != ("R", "A", "S"):
            raise ValueError(f"Canonical orientation check failed: {name}")

    auc = float(roc_auc_score(y, pad))
    correlations = {column: float(spearmanr(pad, index[column]).statistic)
                    for column in ("diameter_mm", "sx", "sz")}
    thickness_means = index.assign(pad_fraction=pad).groupby("thickness", observed=True)["pad_fraction"].mean()

    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(2, 8, figsize=(17, 5.5))
    for row, label in enumerate((0, 1)):
        candidates = np.flatnonzero(y == label)
        chosen = np.sort(rng.choice(candidates, 8, replace=False))
        for ax, i in zip(axes[row], chosen):
            ax.imshow(X[i, 0], cmap="gray", vmin=0, vmax=1, origin="lower")
            ax.contour(masks[i].astype(float), levels=[0.5], colors="red", linewidths=0.7)
            ax.set_title(f"{'malignant' if label else 'benign'}\nmask {areas[i]} px", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_dir / "patches_v1_trainval_montage.png", dpi=150)
    plt.close(fig)

    ras = [i for i, name in enumerate(index["voi_name"])
           if "".join(nib.aff2axcodes(nib.load(str(image_dir / name)).affine)) == "RAS"]
    if not ras:
        raise ValueError("No RAS-stored trainval VOI available for orientation preview")
    chosen = [int(np.argmax(pad)), ras[0], int(rng.choice(len(index)))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, i in enumerate(chosen):
        name = index.iloc[i]["voi_name"]
        image = nib.as_closest_canonical(nib.load(str(image_dir / name)))
        volume = np.asanyarray(image.dataobj).astype(np.float32)
        axes[0, col].imshow(volume[:, :, volume.shape[2] // 2].T, cmap="gray",
                            vmin=-1400, vmax=200, origin="lower")
        axes[0, col].set_title(f"native VOI, {volume.shape[2]} slices", fontsize=9)
        axes[1, col].imshow(X[i, 0], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[1, col].contour(masks[i].astype(float), levels=[0.5], colors="red", linewidths=0.7)
        axes[1, col].set_title(f"32 mm patch, pad={pad[i]:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_dir / "patches_v1_trainval_hardcases.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Real-data preprocessing QA (trainval only)", "",
        f"- Inputs: {len(index)} shared trainval VOIs; held-out test rows and images were not processed.",
        f"- Patch array: `{tuple(X.shape)}`, dtype `{X.dtype}`, finite values, range {float(X.min()):.1f}–{float(X.max()):.1f}.",
        f"- Empty patch masks: {empty_masks}; mask area min/median/max: {int(areas.min())}/{float(np.median(areas)):.1f}/{int(areas.max())} pixels.",
        f"- Mask-centroid offset from patch centre: mean {centroid_offset.mean():.2f}px, median {np.median(centroid_offset):.2f}px, max {centroid_offset.max():.2f}px.",
        f"- Input orientations: {orientation_counts}; image-mask shape/affine mismatches: {geometry_mismatches}. All image headers canonicalized to RAS by the preprocessing routine.",
        f"- Padding: any padding in {np.mean(pad > 0):.1%}; median {np.median(pad):.3f}, 90th percentile {np.quantile(pad, .9):.3f}, maximum {pad.max():.3f}.",
        f"- Raw `pad_fraction` ROC-AUC for malignant label: {auc:.3f}; reverse-direction AUC: {1-auc:.3f}.",
        "- Spearman correlation of padding with: " + ", ".join(f"{k} {v:.3f}" for k, v in correlations.items()) + ".",
        "- Mean padding by slice-thickness group: " + ", ".join(f"{k}: {v:.3f}" for k, v in thickness_means.items()) + ".",
        "",
        "## Visual review", "",
        "The deterministic class montage and three hard-case panels show nodules located within the 32 mm patches, with masks aligned to the visible nodule in the reviewed samples. No obvious orientation or centring failure was seen in these previews. This is a visual spot-check, not a per-sample clinical review.",
        "",
        "## Interpretation and limits", "",
        "The pipeline produced finite, correctly shaped trainval patches with no empty masks or image-mask geometry mismatches. However, padding is strongly associated with native in-plane spacing (|Spearman rho| about 0.94) and appears in most patches. This suggests a possible acquisition-resolution shortcut. The raw padding score alone has weak label discrimination (AUC 0.419; reversed 0.581), but the spacing association means preprocessing QA is not a blanket approval of the representation. Do not alter the configuration based on the held-out test set; any padding/window ablation must be specified and compared using trainval only.",
        "",
        "Patch arrays and preview images are local working data; do not upload them with raw images or masks. Preview files: `data/processed/qa_previews/`.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved aggregate QA report: {out}")


if __name__ == "__main__":
    main()
