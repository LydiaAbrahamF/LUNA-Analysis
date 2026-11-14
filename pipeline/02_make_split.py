"""Stage 2: make the patient-level split ONCE, then lock it.

Reads   data/processed/nodules.csv
Writes  data/processed/splits.csv  (refuses to overwrite unless --force)

After running, record the printed hash (docs/decisions.md). Regenerating the split
changes the sealed test set, so it must be a deliberate, documented act.
"""
import argparse

import pandas as pd

from lunarad import config as cfg
from lunarad.splits import build_split, file_hash, summarize_split, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    parser.add_argument("--check", action="store_true", help="validate the existing split only")
    args = parser.parse_args()

    nodules = pd.read_csv(cfg.NODULES_CSV)

    if cfg.SPLITS_CSV.exists() and not args.force:
        existing = pd.read_csv(cfg.SPLITS_CSV)
        validate_split(nodules, existing)
        print(f"split is locked and valid: {cfg.SPLITS_CSV}")
        print(f"sha256 {file_hash(cfg.SPLITS_CSV)}")
        return
    if args.check:
        raise SystemExit(f"no split found at {cfg.SPLITS_CSV}")

    splits = build_split(nodules)
    print(summarize_split(nodules, splits))
    splits.to_csv(cfg.SPLITS_CSV, index=False)
    print(f"wrote {cfg.SPLITS_CSV}")
    print(f"sha256 {file_hash(cfg.SPLITS_CSV)}   <- record this in docs/decisions.md")


if __name__ == "__main__":
    main()
