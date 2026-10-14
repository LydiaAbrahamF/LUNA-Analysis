"""Phase 3 preprocessing: one fixed-size, size-neutral patch per LUNA16 nodule VOI.

Pipeline per nodule:
  1. load image and mask, reorient both to RAS
  2. centre on the mask centroid (the mask is used only for centring, never as model input)
  3. sample a fixed physical window (default 32 x 32 mm) at a fixed pixel size (default 0.5 mm)
     at fixed slice offsets around the nodule centre, interpolating in physical space
  4. clip HU to [hu_min, hu_max] and scale to [0, 1]
Anything outside the VOI is filled with air and counted in pad_fraction.
"""
import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PreprocessConfig:
    name: str = "v1"
    window_mm: float = 32.0
    out_px: int = 64
    offsets_mm: tuple = (0.0,)
    hu_min: float = -1000.0
    hu_max: float = 400.0
    pad_hu: float = -1000.0


@dataclass
class Patch:
    x: np.ndarray
    mask: np.ndarray
    pad_fraction: float
    center_world: np.ndarray


def find_root():
    return next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())


def hu_to_unit(hu, cfg):
    scaled = (np.asarray(hu, dtype=np.float32) - cfg.hu_min) / (cfg.hu_max - cfg.hu_min)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def sample_patch(vol, affine, center_world, cfg, cval):
    n = cfg.out_px
    coords = (np.arange(n) + 0.5) * (cfg.window_mm / n) - cfg.window_mm / 2
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    inv = np.linalg.inv(affine)
    limit = (np.array(vol.shape, dtype=float) - 1)[:, None, None]
    channels, inside = [], np.ones((n, n), dtype=bool)
    for dz in cfg.offsets_mm:
        world = np.stack([xx + center_world[0], yy + center_world[1],
                          np.full_like(xx, center_world[2] + dz), np.ones_like(xx)])
        vox = np.tensordot(inv, world, axes=(1, 0))[:3]
        channels.append(map_coordinates(vol, vox, order=1, mode="constant", cval=cval))
        inside &= np.all((vox >= 0) & (vox <= limit), axis=0)
    return np.stack(channels), inside


def preprocess_images(img, mask_img, cfg=PreprocessConfig()):
    img = nib.as_closest_canonical(img)
    mask_img = nib.as_closest_canonical(mask_img)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    mask = np.asanyarray(mask_img.dataobj) > 0
    if vol.shape != mask.shape:
        raise ValueError(f"image {vol.shape} and mask {mask.shape} differ in shape")
    if not mask.any():
        raise ValueError("empty mask")
    centre = (img.affine @ np.append(np.argwhere(mask).mean(axis=0), 1.0))[:3]

    hu, inside = sample_patch(vol, img.affine, centre, cfg, cval=cfg.pad_hu)
    mask_patch, _ = sample_patch(mask.astype(np.float32), img.affine, centre,
                                 replace(cfg, offsets_mm=(0.0,)), cval=0.0)
    return Patch(x=hu_to_unit(hu, cfg), mask=mask_patch[0] > 0.5,
                 pad_fraction=float(1.0 - inside.mean()), center_world=centre)


def preprocess_files(image_path, mask_path, cfg=PreprocessConfig()):
    return preprocess_images(nib.load(image_path), nib.load(mask_path), cfg)


def build_dataset(cfg=PreprocessConfig(), root=None):
    root = Path(root) if root else find_root()
    proc = root / "data" / "processed"
    table = pd.read_csv(proc / "nodules.csv")
    need = ["voi_name", "patient_id", "malignant", "split", "fold"]
    missing = [c for c in need if c not in table.columns]
    if missing:
        raise SystemExit(f"nodules.csv is missing {missing}; run the last cells of 01_eda first")

    n = len(table)
    X = np.zeros((n, len(cfg.offsets_mm), cfg.out_px, cfg.out_px), dtype=np.float32)
    M = np.zeros((n, cfg.out_px, cfg.out_px), dtype=bool)
    pad = np.zeros(n)
    voi_dir = root / "data" / "VOIs" / "VOIs"
    for i, name in enumerate(table["voi_name"]):
        p = preprocess_files(voi_dir / "image" / name, voi_dir / "nodule_mask" / name, cfg)
        X[i], M[i], pad[i] = p.x, p.mask, p.pad_fraction
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"{i + 1}/{n}")

    tag = f"patches_{cfg.name}"
    np.save(proc / f"{tag}_X.npy", X)
    np.save(proc / f"{tag}_mask.npy", M)
    index = table[need].assign(pad_fraction=pad)
    index.to_csv(proc / f"{tag}_index.csv", index=False)
    (proc / f"{tag}_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print("saved", tag, X.shape)
    return X, M, index


def main():
    ap = argparse.ArgumentParser(description="Build the patch dataset from the VOIs")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--window-mm", type=float, default=32.0)
    ap.add_argument("--out-px", type=int, default=64)
    ap.add_argument("--offsets-mm", type=float, nargs="+", default=[0.0])
    ap.add_argument("--hu-max", type=float, default=400.0)
    a = ap.parse_args()
    build_dataset(PreprocessConfig(name=a.name, window_mm=a.window_mm, out_px=a.out_px,
                                   offsets_mm=tuple(a.offsets_mm), hu_max=a.hu_max))


if __name__ == "__main__":
    main()
