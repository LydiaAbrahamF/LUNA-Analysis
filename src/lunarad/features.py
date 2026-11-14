"""PyRadiomics feature extraction. Nothing in this module knows about labels or splits:
features are a function of the image and mask only, so they are computed once for all nodules.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from .config import BIN_WIDTH, FEATURE_PREFIX, N_FEATURES, PROTOCOLS


def make_extractor(protocol):
    """One extractor per protocol. CT is already in HU, so intensities are not normalised."""
    return featureextractor.RadiomicsFeatureExtractor(
        binWidth=BIN_WIDTH,
        normalize=False,
        resampledPixelSpacing=PROTOCOLS[protocol],
        interpolator=sitk.sitkBSpline,
        force2D=False,
        label=1,
    )


def extract_one(image_path, mask_path, extractor):
    """Return spacing and the PyRadiomics features of one nodule as a dict."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if (image.GetSize() != mask.GetSize()
            or image.GetSpacing() != mask.GetSpacing()
            or image.GetOrigin() != mask.GetOrigin()
            or image.GetDirection() != mask.GetDirection()):
        raise ValueError("image/mask geometry mismatch")
    if 1 not in set(sitk.GetArrayViewFromImage(mask).ravel()):
        raise ValueError("label 1 is absent from the mask")
    result = extractor.execute(image, mask, label=1)
    features = {k: float(v) for k, v in result.items() if k.startswith(FEATURE_PREFIX)}
    spacing = image.GetSpacing()
    return {"spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2], **features}


def extract_all(nodules, image_dir, mask_dir, protocols=None, verbose=True):
    """Extract every protocol for every nodule. Returns (features, failures) tables.

    A failure on one nodule is logged and skipped so that a long run is not lost.
    """
    protocols = list(protocols or PROTOCOLS)
    extractors = {p: make_extractor(p) for p in protocols}
    records, failures = [], []
    for i, row in enumerate(nodules.itertuples(index=False), start=1):
        for protocol in protocols:
            try:
                values = extract_one(Path(image_dir) / row.voi_name,
                                     Path(mask_dir) / row.voi_name, extractors[protocol])
                records.append({"voi_name": row.voi_name, "patient_id": row.patient_id,
                                "protocol": protocol, **values})
            except Exception as exc:  # keep going and keep an auditable log
                failures.append({"voi_name": row.voi_name, "protocol": protocol,
                                 "error_type": type(exc).__name__, "error": str(exc)})
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(nodules)} nodules done", flush=True)
    columns = ["voi_name", "protocol", "error_type", "error"]
    return pd.DataFrame(records), pd.DataFrame(failures, columns=columns)


def feature_columns(df, expected=N_FEATURES):
    """Names of the PyRadiomics feature columns. Raises if the count is not as expected."""
    columns = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if expected is not None and len(columns) != expected:
        raise ValueError(f"expected {expected} feature columns, found {len(columns)}")
    return columns


def load_feature_table(path):
    table = pd.read_csv(path)
    if table.duplicated(["voi_name", "protocol"]).any():
        raise ValueError("duplicate (voi_name, protocol) rows in the feature table")
    feature_columns(table)
    return table
