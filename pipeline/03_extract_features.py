"""Stage 3: extract PyRadiomics features for every nodule, once.

Reads   data/processed/nodules.csv, the VOI images and masks
Writes  data/processed/features.csv and extraction_failures.csv

Features depend only on image and mask, never on labels or splits, so a change of
label or split never requires re-extraction.

Crash isolation. The compiled part of PyRadiomics can occasionally crash the whole Python
process (on Windows: exit code -1073741819, an access violation) at a different nodule on
every run. To make the extraction reliable, nodules are processed in small chunks, each in
its own worker process. A finished chunk is saved under data/processed/feature_chunks/, so
a crash loses only that chunk: it is retried, and if it keeps crashing its nodules are run
one at a time. Rerunning the script resumes from the saved chunks. Use --fresh to start over.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from lunarad import config as cfg
from lunarad.features import extract_all, feature_columns

CHUNK_DIR = cfg.PROCESSED / "feature_chunks"
FAILURE_COLUMNS = ["voi_name", "protocol", "error_type", "error"]


def chunk_paths(start, stop):
    stem = f"{start:05d}_{stop:05d}"
    return CHUNK_DIR / f"features_{stem}.csv", CHUNK_DIR / f"failures_{stem}.csv"


def run_worker(start, stop, trace):
    """Extract nodules[start:stop] in this process and save the chunk (failures file last = done marker)."""
    marker = os.environ.get("LUNARAD_TEST_CRASH_ONCE")      # used only by the tests, to exercise the retry
    if marker and not Path(marker).exists():
        Path(marker).write_text("crashed once")
        os._exit(139)
    nodules = pd.read_csv(cfg.NODULES_CSV).iloc[start:stop]
    features, failures = extract_all(nodules, cfg.VOI_IMAGE, cfg.VOI_MASK, verbose=False, trace=trace)
    features_path, failures_path = chunk_paths(start, stop)
    for table, path in ((features, features_path), (failures, failures_path)):
        temporary = path.with_suffix(".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)


def spawn(start, stop, trace):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(start), str(stop)]
    return subprocess.run(command + (["--trace"] if trace else [])).returncode


def extract_range(start, stop, retries, trace, crashes):
    """Run one chunk in a worker; on repeated crashes fall back to one nodule per worker."""
    if chunk_paths(start, stop)[1].exists():
        return
    for attempt in range(1, retries + 1):
        code = spawn(start, stop, trace)
        if code == 0:
            return
        crashes.append({"start": start, "stop": stop, "attempt": attempt, "exit_code": code})
        print(f"  worker for nodules {start}-{stop - 1} crashed (exit code {code}), attempt {attempt}/{retries}",
              flush=True)
    if stop - start > 1:
        for single in range(start, stop):
            extract_range(single, single + 1, retries, trace, crashes)
        return
    name = pd.read_csv(cfg.NODULES_CSV)["voi_name"].iloc[start]            # one nodule, crashed every time
    pd.DataFrame(columns=["voi_name"]).to_csv(chunk_paths(start, stop)[0], index=False)
    pd.DataFrame([{"voi_name": name, "protocol": "all", "error_type": "ProcessCrash",
                   "error": f"worker crashed {retries} times"}]).to_csv(chunk_paths(start, stop)[1], index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="quick test on the first N nodules; writes NOTHING to disk")
    parser.add_argument("--chunk-size", type=int, default=25, help="nodules per worker process")
    parser.add_argument("--retries", type=int, default=3, help="attempts per chunk before splitting it")
    parser.add_argument("--fresh", action="store_true", help="discard saved chunks and start over")
    parser.add_argument("--trace", action="store_true", help="print every nodule before extracting it")
    parser.add_argument("--worker", nargs=2, type=int, metavar=("START", "STOP"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(*args.worker, trace=args.trace)
        return

    nodules = pd.read_csv(cfg.NODULES_CSV)
    if args.limit:
        features, failures = extract_all(nodules.head(args.limit), cfg.VOI_IMAGE, cfg.VOI_MASK, trace=args.trace)
        print(f"quick test: rows {len(features)} | failures {len(failures)} | nothing written")
        return

    if args.fresh and CHUNK_DIR.exists():
        shutil.rmtree(CHUNK_DIR)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    (CHUNK_DIR / "nodules_used.txt").write_text("\n".join(nodules["voi_name"]))
    n, crashes = len(nodules), []
    print(f"extracting {n} nodules x {len(cfg.PROTOCOLS)} protocols in chunks of {args.chunk_size}", flush=True)
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        extract_range(start, stop, args.retries, args.trace, crashes)
        if stop % 100 < args.chunk_size or stop == n:
            print(f"  {stop}/{n} nodules done", flush=True)

    feature_files = sorted(CHUNK_DIR.glob("features_*.csv"))
    features = pd.concat([pd.read_csv(f) for f in feature_files if f.stat().st_size > 12], ignore_index=True)
    failures = pd.concat([pd.read_csv(f) for f in sorted(CHUNK_DIR.glob("failures_*.csv"))] +
                         [pd.DataFrame(columns=FAILURE_COLUMNS)], ignore_index=True)
    order = {name: i for i, name in enumerate(nodules["voi_name"])}
    features = features.assign(_order=features["voi_name"].map(order)).sort_values(
        ["_order", "protocol"], kind="stable").drop(columns="_order").reset_index(drop=True)
    assert not features.duplicated(["voi_name", "protocol"]).any(), "a nodule was extracted twice; rerun with --fresh"

    features.to_csv(cfg.FEATURES_CSV, index=False)
    failures.to_csv(cfg.FAILURES_CSV, index=False)
    pd.DataFrame(crashes, columns=["start", "stop", "attempt", "exit_code"]).to_csv(
        cfg.PROCESSED / "extraction_crashes.csv", index=False)

    expected = n * len(cfg.PROTOCOLS)
    print(f"rows: {len(features)} (expected {expected}) | failures: {len(failures)}")
    print(f"feature columns: {len(feature_columns(features))}")
    print(f"worker crashes that were retried: {len(crashes)} (see extraction_crashes.csv)")
    if len(failures) or len(features) != expected:
        print(failures.head().to_string(index=False), file=sys.stderr)
        sys.exit("extraction incomplete; see " + str(cfg.FAILURES_CSV))


if __name__ == "__main__":
    main()
