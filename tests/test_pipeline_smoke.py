"""Run pipeline stages 1-5 end to end on a tiny synthetic dataset (not real data)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("radiomics")
pytest.importorskip("nibabel")

from synthetic import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("luna")
    data = make_dataset(base / "data")
    env = dict(os.environ, LUNA_DATA=str(data), LUNA_RESULTS=str(base / "results"),
               PYTHONPATH=str(ROOT / "src"))
    return base, env


def run(env, script, *args, expect_ok=True):
    result = subprocess.run([sys.executable, str(ROOT / "pipeline" / script), *args],
                            env=env, capture_output=True, text=True, cwd=ROOT)
    if expect_ok:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_full_pipeline(workspace):
    base, env = workspace
    processed, tables = base / "data" / "processed", base / "results" / "tables"

    run(env, "01_build_dataset.py", "--qa")
    nodules = pd.read_csv(processed / "nodules.csv")
    assert {"malignant", "is_tie", "thickness", "diameter_mm"} <= set(nodules.columns)
    assert "split" not in nodules.columns and "fold" not in nodules.columns

    run(env, "02_make_split.py")
    locked = (processed / "splits.csv").read_bytes()
    run(env, "02_make_split.py")                                  # second run must not rewrite
    assert (processed / "splits.csv").read_bytes() == locked

    run(env, "03_extract_features.py")
    features = pd.read_csv(processed / "features.csv")
    assert len(features) == 2 * len(nodules)
    assert len([c for c in features.columns if c.startswith("original_")]) == 107

    run(env, "04_run_models.py")
    cv = pd.read_csv(tables / "cv_metrics.csv")
    assert {"diameter", "scanner", "shape_only", "radiomics", "combined"} <= set(cv["input"])
    assert not (tables / "test_metrics.csv").exists()             # cross-validation only

    run(env, "05_robustness.py", "--masks", "5")
    assert (tables / "feature_ranking_stability.csv").exists()

    run(env, "04_run_models.py", "--final")
    assert (tables / "test_metrics.csv").exists()
    again = run(env, "04_run_models.py", "--final", expect_ok=False)
    assert again.returncode != 0                                   # the test set is sealed
    run(env, "04_run_models.py", "--final", "--force-final")
