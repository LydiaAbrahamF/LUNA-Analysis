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
    assert {"malignant", "thickness", "diameter_mm"} <= set(nodules.columns)
    assert "split" not in nodules.columns and "fold" not in nodules.columns
    for leaky in ["ratings", "rating_mean", "n_high", "n_raters", "is_tie", "label_mean_gt3",
                  "label_dataset_rule", "label_majority", "n_annotations"]:
        assert leaky not in nodules.columns                     # one label column only
    audit = pd.read_csv(processed / "label_audit.csv")
    assert {"ratings", "rating_mean", "is_tie"} <= set(audit.columns)
    assert (audit["malignant"] == nodules["malignant"]).all()

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
    assert not (tables / "final_test_metrics.csv").exists()       # cross-validation only

    run(env, "05_robustness.py", "--masks", "5")
    assert (tables / "feature_ranking_stability.csv").exists()

    assert {"cv_roc_auc_ci_low", "cv_roc_auc_ci_high"} <= set(cv.columns)
    assert (cv["cv_roc_auc_ci_low"] <= cv["cv_roc_auc"]).all()
    assert (cv["cv_roc_auc"] <= cv["cv_roc_auc_ci_high"]).all()

    run(env, "06_repeated_cv.py", "--repeats", "3", "--bootstrap", "50")
    rep = pd.read_csv(tables / "cv_repeats_summary.csv")
    assert (rep["n_repeats"] == 3).all() and (rep["roc_auc_min"] <= rep["roc_auc_max"]).all()
    assert not (tables / "final_test_metrics.csv").exists()       # still sealed

    run(env, "07_compare.py", "--repeats", "2", "--bootstrap", "30")
    paired = pd.read_csv(tables / "paired_differences.csv")
    assert len(paired) == 11 and (paired["ci_low"] <= paired["ci_high"]).all()
    assert (tables / "feature_size_correlation.csv").exists()
    assert len(pd.read_csv(tables / "size_stratified_auc.csv")) == 9
    assert not (tables / "final_test_metrics.csv").exists()       # still sealed

    run(env, "08_model_comparison.py", "--repeats", "1", "--bootstrap", "30",
        "--models", "logreg", "logreg_l2", "hist_gb", "--inputs", "radiomics",
        "--ensemble-members", "logreg_l2", "hist_gb")
    cmp_ = pd.read_csv(tables / "model_comparison.csv")
    assert set(cmp_["model"]) == {"logreg", "logreg_l2", "hist_gb", "ensemble_avg"}
    assert len(pd.read_csv(tables / "model_comparison_paired.csv")) == 5
    oof = pd.read_csv(tables / "oof_predictions.csv")
    assert oof["probability_malignant"].between(0, 1).all() and not oof["voi_name"].isna().any()
    run(env, "09_threshold_calibration.py", "--primary", "radiomics", "logreg_l2",
        "--no-secondary", "--bootstrap", "30")
    thr = pd.read_csv(tables / "threshold_analysis.csv")
    assert len(thr) == 7 and (thr["sensitivity_ci_low"] <= thr["sensitivity_ci_high"]).all()
    recall_rows = thr[thr["rule"].str.startswith("recall")]
    assert (recall_rows["sensitivity"].to_numpy() >= [0.90, 0.95, 0.98]).all()
    assert (base / "results" / "figures" / "calibration.png").exists()
    assert not (tables / "final_test_metrics.csv").exists()       # still sealed

    old_final = run(env, "04_run_models.py", "--final", expect_ok=False)
    assert old_final.returncode != 0                               # the old test path is gone
    assert not (tables / "final_test_metrics.csv").exists()

    stage9 = float(thr.loc[(thr["rule"] == "recall >= 0.95"), "threshold"].iloc[0])
    final_args = ["--primary", "radiomics", "logreg_l2", "--secondary", "radiomics", "hist_gb",
                  "--primary-threshold", str(round(stage9, 4)), "--secondary-threshold", "0.5",
                  "--bootstrap", "30"]
    wrong = run(env, "10_final_test.py", *final_args[:3], "--primary-threshold", "0.9", "--confirm",
                expect_ok=False)
    assert wrong.returncode != 0 and "mismatch" in wrong.stderr   # threshold must match stage 9
    assert not (tables / "final_test_metrics.csv").exists()
    refused = run(env, "10_final_test.py", *final_args, expect_ok=False)
    assert refused.returncode != 0 and not (tables / "final_test_metrics.csv").exists()  # no --confirm
    bad_hash = run(env, "10_final_test.py", *final_args, "--confirm", "--expect-sha256", "0" * 64,
                   expect_ok=False)
    assert bad_hash.returncode != 0 and not (tables / "final_test_metrics.csv").exists()

    run(env, "10_final_test.py", *final_args, "--confirm")
    final = pd.read_csv(tables / "final_test_metrics.csv")
    assert list(final["role"]) == ["primary", "secondary", "reference"]
    assert (final["roc_auc_ci_low"] <= final["roc_auc"]).all()
    assert len(pd.read_csv(tables / "final_test_threshold_metrics.csv")) == 2
    assert len(pd.read_csv(tables / "final_test_paired.csv")) == 3
    again = run(env, "10_final_test.py", *final_args, "--confirm", expect_ok=False)
    assert again.returncode != 0                                   # the test set is sealed
    run(env, "10_final_test.py", *final_args, "--confirm", "--force-final")
    assert bool(pd.read_csv(tables / "final_test_metrics.csv")["forced_second_pass"].all())
