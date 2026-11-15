"""Compare two results folders (for example local run vs Docker run) number by number.

    python scripts/compare_results.py results results_docker --tolerance 0.005

For every CSV that exists in both folders it compares all numeric columns row by row and prints the
largest absolute difference. Exit code 0 if every difference is within the tolerance, 1 otherwise.
Non-numeric columns must match exactly. Patient-level prediction files are skipped by default.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKIP = {"oof_predictions.csv", "final_test_predictions.csv"}


def compare(a: Path, b: Path, tolerance: float):
    left, right = pd.read_csv(a), pd.read_csv(b)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return None, f"shape or columns differ: {left.shape} vs {right.shape}"
    worst, worst_where = 0.0, ""
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            diff = np.nanmax(np.abs(left[column].to_numpy(float) - right[column].to_numpy(float)),
                             initial=0.0)
            if diff > worst:
                worst, worst_where = float(diff), column
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return None, f"text column '{column}' differs"
    return worst, worst_where


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    first, second = Path(args.first), Path(args.second)
    names = sorted(p.relative_to(first).as_posix() for p in first.rglob("*.csv")
                   if p.name not in SKIP and (second / p.relative_to(first)).exists())
    if not names:
        sys.exit("no CSV file exists in both folders")
    bad = 0
    for name in names:
        worst, where = compare(first / name, second / name, args.tolerance)
        if worst is None:
            status, detail = "DIFFERENT", where
            bad += 1
        elif worst > args.tolerance:
            status, detail = "OUT OF TOLERANCE", f"max |diff| {worst:.4f} in '{where}'"
            bad += 1
        else:
            status, detail = "ok", f"max |diff| {worst:.4f}"
        print(f"{status:17s} {name:45s} {detail}")
    only_first = sorted(p.name for p in first.rglob("*.csv")
                        if p.name not in SKIP and not (second / p.relative_to(first)).exists())
    if only_first:
        print(f"\nonly in {first}: {', '.join(only_first)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
