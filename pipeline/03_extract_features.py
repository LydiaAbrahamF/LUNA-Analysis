"""Stage 3: extract PyRadiomics features for every nodule, once.

Reads   data/processed/nodules.csv, the VOI images and masks
Writes  data/processed/features.csv and extraction_failures.csv

Features depend only on image and mask, never on labels or splits, so a change of
label or split never requires re-extraction.
"""
import argparse
import sys

import pandas as pd

from lunarad import config as cfg
from lunarad.features import extract_all, feature_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N nodules (quick test)")
    args = parser.parse_args()

    nodules = pd.read_csv(cfg.NODULES_CSV)
    if args.limit:
        nodules = nodules.head(args.limit)
    print(f"extracting {len(nodules)} nodules x {len(cfg.PROTOCOLS)} protocols")

    features, failures = extract_all(nodules, cfg.VOI_IMAGE, cfg.VOI_MASK)
    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_csv(cfg.FEATURES_CSV, index=False)
    failures.to_csv(cfg.FAILURES_CSV, index=False)

    expected = len(nodules) * len(cfg.PROTOCOLS)
    print(f"rows: {len(features)} (expected {expected}) | failures: {len(failures)}")
    print(f"feature columns: {len(feature_columns(features))}")
    if len(failures) or len(features) != expected:
        print(failures.head(), file=sys.stderr)
        sys.exit("extraction incomplete; see " + str(cfg.FAILURES_CSV))


if __name__ == "__main__":
    main()
