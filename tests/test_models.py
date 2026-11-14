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
