"""Model factories, input sets, and patient-grouped cross-validation."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_PREFIX, LOGREG_C, SCANNER_COLUMNS


def make_logreg():
    """Median imputation, standardisation, logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=LOGREG_C, max_iter=5000, solver="lbfgs"),
    )


# Add your stage-3 candidates here. Every factory returns a fresh, unfitted estimator.
MODEL_FACTORIES = {"logreg": make_logreg}

_SHAPE = FEATURE_PREFIX + "shape_"
_FIRSTORDER = FEATURE_PREFIX + "firstorder_"
_TEXTURE = tuple(f"{FEATURE_PREFIX}{name}_" for name in ("glcm", "glrlm", "glszm", "ngtdm", "gldm"))

INPUT_SETS = ("diameter", "scanner", "shape_only", "firstorder_only", "texture_only",
              "radiomics_no_shape", "radiomics", "combined")


def columns_for(name, table):
    """Column list for a named input set. 'combined' is diameter plus all radiomics."""
    features = [c for c in table.columns if c.startswith(FEATURE_PREFIX)]
    sets = {
        "diameter": ["diameter_mm"],
        "scanner": list(SCANNER_COLUMNS),
        "shape_only": [c for c in features if c.startswith(_SHAPE)],
        "firstorder_only": [c for c in features if c.startswith(_FIRSTORDER)],
        "texture_only": [c for c in features if c.startswith(_TEXTURE)],
        "radiomics_no_shape": [c for c in features if not c.startswith(_SHAPE)],
        "radiomics": features,
        "combined": ["diameter_mm"] + features,
    }
    if name not in sets:
        raise KeyError(f"unknown input set {name!r}; choose from {sorted(sets)}")
    columns = sets[name]
    if not columns:
        raise ValueError(f"input set {name!r} has no columns in this table")
    return columns


def oof_predictions(table, columns, make_model):
    """Out-of-fold probabilities using the 'fold' column (each fold predicted by a model
    trained on the other folds). The table must contain trainval rows only."""
    needed = {"fold", "patient_id", "malignant", *columns}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    if (table["fold"] < 0).any():
        raise ValueError("test rows (fold -1) must not be in the cross-validation table")
    oof = np.full(len(table), np.nan)
    for fold in sorted(table["fold"].unique()):
        validation = (table["fold"] == fold).to_numpy()
        if set(table.loc[validation, "patient_id"]) & set(table.loc[~validation, "patient_id"]):
            raise ValueError(f"patient leakage in fold {fold}")
        model = make_model()
        model.fit(table.loc[~validation, columns], table.loc[~validation, "malignant"])
        oof[validation] = model.predict_proba(table.loc[validation, columns])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("missing out-of-fold predictions")
    return oof


def fit_predict(train, test, columns, make_model):
    """Fit on all of train, return probabilities for test."""
    model = make_model()
    model.fit(train[columns], train["malignant"])
    return model.predict_proba(test[columns])[:, 1]
