"""Stage 1: build the nodule table (labels + image geometry) from the raw LUNA16 files.

Reads   data/MetadatabyAnnotation.xlsx, data/CT, data/VOIs
Writes  data/processed/image_inventory.csv, nodules.csv and label_audit.csv

nodules.csv is the modelling table: ONE binary label column (malignant) plus image geometry.
label_audit.csv holds the ratings and sensitivity labels for analysis. The split lives in
splits.csv (stage 2).
Run from the repo root with PYTHONPATH=src.
"""
import argparse
import sys

import nibabel as nib
import numpy as np
import pandas as pd

from lunarad import config as cfg
from lunarad.labels import add_label_columns

KNOWN_N, KNOWN_MALIGNANT, KNOWN_TIES = 994, 550, 146   # full dataset, mean >= 3 rule

# Everything about the raters. Kept out of nodules.csv and written to label_audit.csv.
RATING_COLUMNS = ["ratings", "n_annotations", "n_raters", "n_high", "rating_mean", "is_tie",
                  "label_mean_gt3", "label_dataset_rule", "label_majority"]


def describe_file(path, kind):
    header = nib.load(path)
    sx, sy, sz = header.header.get_zooms()[:3]
    nx, ny, nz = header.shape
    return {"kind": kind, "file": path.name, "nx": nx, "ny": ny, "nz": nz,
            "sx": round(sx, 3), "sy": round(sy, 3), "sz": round(sz, 3),
            "orientation": "".join(nib.aff2axcodes(header.affine))}


def offset_vox(img, x, y, z):
    """Voxel offset of a metadata coordinate from the VOI centre.
    Metadata uses the DICOM LPS convention, so x and y are negated (validated in the EDA)."""
    ijk = (np.linalg.inv(img.affine) @ np.array([-x, -y, z, 1]))[:3]
    return ijk - (np.array(img.shape) - 1) / 2


def build_inventory(scan_files, voi_files, mask_files):
    sources = [("full_ct", scan_files), ("voi_image", voi_files), ("voi_mask", mask_files)]
    return pd.DataFrame([describe_file(p, kind) for kind, files in sources for p in files])


def aggregate_nodules(metadata):
    """One row per nodule. Coordinates are part of the key so that two different nodules
    sharing an ID (LIDC-IDRI-0332) stay separate. One metadata row is one annotation."""
    metadata = metadata.assign(bbox_x=metadata["bboxHighX"] - metadata["bboxLowX"],
                               bbox_z=metadata["bboxHighZ"] - metadata["bboxLowZ"])
    key = ["patient_id", "nodule_id", "coordX", "coordY", "coordZ"]
    nodules = (metadata.groupby(key, as_index=False)
               .agg(ratings=("Malignancy_value", list),
                    n_annotations=("anotation_id", "nunique"),
                    diameter_mm=("diameter_mm", "first"),
                    bbox_x=("bbox_x", "first"),
                    bbox_z=("bbox_z", "first")))
    nodules["part"] = nodules.groupby(["patient_id", "nodule_id"]).cumcount()
    nodules["voi_name"] = (nodules["patient_id"] + "_R_"
                           + nodules["nodule_id"].astype(str) + ".nii.gz")
    return nodules


def resolve_shared_ids(nodules, voi_dir):
    """When two nodules share (patient, nodule_id), the VOI file belongs to the one whose
    coordinates are closest to the VOI centre; the other is marked as having no image."""
    nodules = nodules.copy()
    for (_, _), group in nodules[nodules["has_voi"]].groupby(["patient_id", "nodule_id"]):
        if len(group) < 2:
            continue
        voi = nib.load(voi_dir / group["voi_name"].iloc[0])
        distance = {idx: np.linalg.norm(offset_vox(voi, r["coordX"], r["coordY"], r["coordZ"]))
                    for idx, r in group.iterrows()}
        owner = min(distance, key=distance.get)
        for idx in group.index:
            if idx != owner:
                nodules.loc[idx, "has_voi"] = False
        print(f"shared ID {group['voi_name'].iloc[0]}: image belongs to part "
              f"{nodules.loc[owner, 'part']}")
    return nodules


def crop_check():
    """Prove that a VOI is an exact crop of its CT (validates the coordinate conversion)."""
    ct_path = cfg.CT_DIR / "LIDC-IDRI-0001.nii.gz"
    voi_path = cfg.VOI_IMAGE / "LIDC-IDRI-0001_R_1.nii.gz"
    if not (ct_path.is_file() and voi_path.is_file()):
        print("crop check skipped (LIDC-IDRI-0001 CT or VOI not found)")
        return
    ct, voi = nib.load(ct_path), nib.load(voi_path)
    corner = (np.linalg.inv(ct.affine) @ voi.affine @ np.array([0, 0, 0, 1]))[:3]
    i, j, k = np.round(corner).astype(int)
    nx, ny, nz = voi.shape
    assert np.allclose(ct.affine[:3, :3], voi.affine[:3, :3])
    assert np.allclose(np.asanyarray(ct.dataobj)[i:i + nx, j:j + ny, k:k + nz],
                       np.asanyarray(voi.dataobj))
    print("crop check passed: VOI is an exact crop of its CT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="also run the VOI-vs-CT crop check")
    args = parser.parse_args()

    scan_files = sorted(cfg.CT_DIR.glob("*.nii.gz"))
    voi_files = sorted(cfg.VOI_IMAGE.glob("*.nii.gz"))
    mask_files = sorted(cfg.VOI_MASK.glob("*.nii.gz"))
    print(f"CT scans: {len(scan_files)}, VOIs: {len(voi_files)}, masks: {len(mask_files)}")
    if not voi_files:
        sys.exit(f"no VOI files found in {cfg.VOI_IMAGE}; check LUNA_DATA / the data folder")

    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(scan_files, voi_files, mask_files)
    inventory.to_csv(cfg.INVENTORY_CSV, index=False)

    metadata = pd.read_excel(cfg.METADATA_XLSX)
    nodules = add_label_columns(aggregate_nodules(metadata))
    nodules["has_voi"] = nodules["voi_name"].isin({p.name for p in voi_files})
    nodules = resolve_shared_ids(nodules, cfg.VOI_IMAGE)

    vois = inventory[inventory["kind"] == "voi_image"].assign(
        mm_x=lambda d: d["nx"] * d["sx"], mm_z=lambda d: d["nz"] * d["sz"])
    props = (vois[["file", "nx", "ny", "nz", "sx", "sz", "orientation", "mm_x", "mm_z"]]
             .rename(columns={"file": "voi_name"}))
    final = nodules[nodules["has_voi"]].merge(props, on="voi_name", how="left")
    final["thickness"] = pd.cut(final["sz"], [0, 1.0, 1.5, 3.0],
                                labels=["thin (<=1.0)", "1.25", "thick (>=1.8)"])
    final["pixel_size"] = pd.qcut(final["sx"], 3, labels=["small", "medium", "large"])
    final["size_bin"] = pd.qcut(final["diameter_mm"], 3, labels=["small", "medium", "large"])

    assert final["voi_name"].is_unique
    assert final["nx"].notna().all(), "a nodule has no matching row in the image inventory"

    n, malignant, ties = len(final), int(final["malignant"].sum()), int(final["is_tie"].sum())
    print(f"nodules with image: {n} | malignant {malignant} | benign {n - malignant} | "
          f"ties at the threshold {ties}")
    if n == KNOWN_N and cfg.LABEL_THRESHOLD == 3.0:       # known-answer check, full dataset only
        assert (malignant, ties) == (KNOWN_MALIGNANT, KNOWN_TIES), \
            f"expected {KNOWN_MALIGNANT} malignant and {KNOWN_TIES} ties on the full dataset"

    # The modelling table keeps exactly one label column. Ratings, their mean and the
    # sensitivity labels go to a separate audit file: a column like rating_mean in the
    # modelling table is the label before thresholding, and a leak waiting to happen.
    audit = final[["voi_name", "patient_id", "malignant", *RATING_COLUMNS]]
    audit.to_csv(cfg.LABEL_AUDIT_CSV, index=False)
    final.drop(columns=RATING_COLUMNS).to_csv(cfg.NODULES_CSV, index=False)
    print(f"wrote {cfg.NODULES_CSV} (one label column: malignant)")
    print(f"wrote {cfg.LABEL_AUDIT_CSV} (ratings and sensitivity labels, analysis only)")
    if args.qa:
        crop_check()


if __name__ == "__main__":
    main()