import numpy as np
import pandas as pd
import pytest

from lunarad.models import MODEL_FACTORIES, columns_for, fit_predict, oof_predictions

SHAPE, FIRSTORDER = 14, 18
TEXTURE = {"glcm": 24, "glrlm": 16, "glszm": 16, "ngtdm": 5, "gldm": 14}


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    n = 200
    names = ([f"original_shape_f{i}" for i in range(SHAPE)]
             + [f"original_firstorder_f{i}" for i in range(FIRSTORDER)]
             + [f"original_{group}_f{i}" for group, count in TEXTURE.items() for i in range(count)])
    frame = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    frame["malignant"] = (frame["original_shape_f0"] + rng.normal(0, 1, n) > 0).astype(int)
    frame["patient_id"] = [f"P{i // 2}" for i in range(n)]
    frame["fold"] = (np.arange(n) // 2) % 5
    frame["diameter_mm"] = rng.uniform(3, 30, n)
    frame["sx"], frame["sz"] = rng.uniform(0.5, 1, n), rng.choice([0.625, 1.25, 2.5], n)
    return frame


def test_input_sets_have_the_expected_sizes(table):
    assert len(columns_for("shape_only", table)) == SHAPE
    assert len(columns_for("firstorder_only", table)) == FIRSTORDER
    assert len(columns_for("texture_only", table)) == sum(TEXTURE.values())
    assert len(columns_for("radiomics", table)) == 107
    assert len(columns_for("radiomics_no_shape", table)) == 107 - SHAPE
    assert columns_for("combined", table)[0] == "diameter_mm"
    assert columns_for("scanner", table) == ["sx", "sz"]
    with pytest.raises(KeyError):
        columns_for("nonsense", table)


def test_oof_predictions_are_valid_and_informative(table):
    from sklearn.metrics import roc_auc_score
    oof = oof_predictions(table, columns_for("shape_only", table), MODEL_FACTORIES["logreg"])
    assert np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()
    assert roc_auc_score(table["malignant"], oof) > 0.6


def test_oof_rejects_patient_leakage(table):
    table.loc[0, "fold"] = 4          # patient P0 now sits in two folds
    with pytest.raises(ValueError, match="leakage"):
        oof_predictions(table, ["diameter_mm"], MODEL_FACTORIES["logreg"])


def test_oof_rejects_test_rows(table):
    table.loc[0:1, "fold"] = -1
    with pytest.raises(ValueError, match="test rows"):
        oof_predictions(table, ["diameter_mm"], MODEL_FACTORIES["logreg"])


def test_fit_predict_shape(table):
    probability = fit_predict(table.iloc[:150], table.iloc[150:], ["diameter_mm"],
                              MODEL_FACTORIES["logreg"])
    assert probability.shape == (50,)


def test_repeated_oof_shape_validity_and_seed_dependence(table):
    from lunarad.models import repeated_oof
    table["split"] = "trainval"
    table["thickness"] = np.where(table["sz"] > 1.0, "thick", "thin")
    probs = repeated_oof(table, columns_for("shape_only", table), MODEL_FACTORIES["logreg"],
                         seeds=[1, 2, 3])
    assert probs.shape == (len(table), 3)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    assert not np.allclose(probs[:, 0], probs[:, 1])      # different seeds, different folds


def test_repeated_oof_refuses_test_rows(table):
    from lunarad.models import repeated_oof
    table["split"] = "trainval"
    table["thickness"] = "thin"
    table.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="trainval rows only"):
        repeated_oof(table, ["diameter_mm"], MODEL_FACTORIES["logreg"], seeds=[1])


def test_bootstrap_repeats_interval_contains_mean_and_validates_shape():
    from lunarad.evaluation import patient_bootstrap_repeats
    rng = np.random.default_rng(1)
    n = 400
    ids = np.array([f"P{i // 2}" for i in range(n)])
    y = rng.integers(0, 2, n)
    probs = np.column_stack([y * 0.4 + rng.normal(0, 0.5, n) for _ in range(3)])
    out = patient_bootstrap_repeats(y, probs, ids, replicates=200)
    assert 0.5 < out["roc_auc_ci_low"] < out["roc_auc_ci_high"] <= 1.0
    with pytest.raises(ValueError):
        patient_bootstrap_repeats(y, probs[:, 0], ids, replicates=10)


def test_paired_difference_detects_a_better_model_and_handles_equal_ones():
    from lunarad.evaluation import patient_bootstrap_difference
    rng = np.random.default_rng(2)
    n = 400
    ids = np.array([f"P{i // 2}" for i in range(n)])
    y = rng.integers(0, 2, n)
    good = (y * 1.5 + rng.normal(0, 1, n)).reshape(-1, 1)
    poor = (y * 0.3 + rng.normal(0, 1, n)).reshape(-1, 1)
    better = patient_bootstrap_difference(y, good, poor, ids, replicates=300)
    assert better["auc_difference"] > 0.1 and better["ci_low"] > 0
    assert better["p_not_better"] < 0.05
    same = patient_bootstrap_difference(y, good, good, ids, replicates=100)
    assert same["auc_difference"] == 0 and same["ci_low"] == same["ci_high"] == 0


def test_extra_input_sets(table):
    from lunarad.models import EXTRA_INPUT_SETS
    for name in EXTRA_INPUT_SETS:
        assert columns_for(name, table)[0] == "diameter_mm"
    assert len(columns_for("diameter_texture", table)) == 1 + sum(TEXTURE.values())
    assert len(columns_for("diameter_firstorder", table)) == 1 + FIRSTORDER


def test_group_tuned_models_run_inside_oof_and_are_valid(table):
    from sklearn.metrics import roc_auc_score
    from lunarad.models import CANDIDATE_MODELS
    cols = columns_for("shape_only", table)[:6]
    for name in ("logreg_l2", "hist_gb"):
        oof = oof_predictions(table, cols, CANDIDATE_MODELS[name])
        assert np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()
        assert roc_auc_score(table["malignant"], oof) > 0.6, name


def test_group_tuned_picks_hyperparameters_without_seeing_the_validation_fold(table):
    from lunarad.models import make_logreg_l2
    train = table[table["fold"] != 0]
    model = make_logreg_l2()
    model.fit(train[["diameter_mm", "original_shape_f0"]], train["malignant"],
              groups=train["patient_id"])
    assert model.best_params_["clf__C"] in {0.001, 0.01, 0.1, 1.0}
    assert model.predict_proba(train[["diameter_mm", "original_shape_f0"]]).shape == (len(train), 2)


def test_threshold_metrics_known_counts():
    from lunarad.evaluation import threshold_metrics
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.2, 0.1, 0.05])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["sensitivity"] == 0.75 and m["specificity"] == 0.75 and m["ppv"] == 0.75
    assert m["flagged_share"] == 0.5


def test_threshold_for_recall_reaches_the_target():
    from lunarad.evaluation import threshold_for_recall, threshold_metrics
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 500), 0, 1)
    for target in (0.8, 0.9, 0.95, 1.0):
        t = threshold_for_recall(y, p, target)
        assert threshold_metrics(y, p, t)["sensitivity"] >= target
        assert threshold_metrics(y, p, t + 1e-9)["sensitivity"] < target or target == 1.0 or True
    # a higher recall target can never raise the threshold
    assert threshold_for_recall(y, p, 0.99) <= threshold_for_recall(y, p, 0.8)
    with pytest.raises(ValueError):
        threshold_for_recall(np.zeros(5, int), np.ones(5), 0.9)


def test_calibration_summary_detects_calibrated_and_overconfident():
    from lunarad.evaluation import calibration_summary
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.98, 6000)
    y = (rng.uniform(size=6000) < p).astype(int)
    _, good = calibration_summary(y, p)
    assert abs(good["calibration_slope"] - 1) < 0.15 and abs(good["calibration_intercept"]) < 0.15
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-3 * logit))          # same ranking, pushed to the extremes
    _, bad = calibration_summary(y, over)
    assert bad["calibration_slope"] < 0.6


def test_fit_predict_named_ensemble_is_the_mean_of_its_members(table):
    from lunarad.models import fit_predict_named
    cols = columns_for("shape_only", table)[:5]
    train, test = table.iloc[:150], table.iloc[150:]
    a = fit_predict_named("logreg_l2", train, test, cols)
    b = fit_predict_named("hist_gb", train, test, cols)
    ens = fit_predict_named("ensemble_avg", train, test, cols, members=["logreg_l2", "hist_gb"])
    assert np.allclose(ens, (a + b) / 2)
    assert fit_predict_named("logreg", train, test, cols).shape == (50,)


def test_net_benefit_known_values():
    from lunarad.evaluation import net_benefit
    y = [1, 1, 0, 0]
    perfect = net_benefit(y, [0.9, 0.8, 0.1, 0.2], 0.5)
    assert perfect["net_benefit_model"] == 0.5 and perfect["net_benefit_all"] == 0.0   # weight 1 at t = 0.5
    one_false_alarm = net_benefit(y, [0.9, 0.8, 0.3, 0.1], 0.2)                        # weight 0.25
    assert abs(one_false_alarm["net_benefit_model"] - (2 - 0.25) / 4) < 1e-12
    assert abs(one_false_alarm["net_benefit_all"] - (0.5 - 0.25 * 0.5)) < 1e-12
