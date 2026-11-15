"""Compare two feature tables value by value (for example two independent extractions).

Usage: python scripts/compare_features.py data/processed/features.csv data/processed/features_second.csv
"""
import sys

import numpy as np
import pandas as pd

a, b = (pd.read_csv(p).sort_values(["voi_name", "protocol"]).reset_index(drop=True) for p in sys.argv[1:3])
print(f"rows: {len(a)} vs {len(b)}")
if len(a) != len(b) or not (a[["voi_name", "protocol"]] == b[["voi_name", "protocol"]]).all().all():
    sys.exit("the two tables do not contain the same nodules and protocols")
numeric = [c for c in a.columns if c.startswith("original_")]
difference = np.abs(a[numeric].to_numpy(float) - b[numeric].to_numpy(float))
scale = np.maximum(np.abs(a[numeric].to_numpy(float)), 1e-12)
print(f"feature columns compared: {len(numeric)}")
print(f"largest absolute difference: {np.nanmax(difference):.3g}")
print(f"largest relative difference: {np.nanmax(difference / scale):.3g}")
print(f"values that differ by more than 1e-9 (relative): {int((difference / scale > 1e-9).sum())} of {difference.size}")
