"""Model factories, input sets, and patient-grouped cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ENSEMBLE_MEMBERS, FEATURE_PREFIX, LOGREG_C, N_INNER, N_TUNING_FOLDS, SCANNER_COLUMNS, SEED
from .splits import refold


def make_logreg():
    """Median imputation, standardisation, logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=LOGREG_C, max_iter=5000, solver="lbfgs"),
    )


class GroupTuned:
    """Hyperparameter search that happens INSIDE one training set, on patient-grouped splits.

    Used so that tuning never sees the validation fold: the outer loop (oof_predictions)
    hands over only its training rows and their patient ids, and this class cross-validates
    among those rows to pick hyperparameters, then refits on all of them.
    """
    needs_groups = True

    def __init__(self, estimator, grid, n_splits=N_TUNING_FOLDS, seed=SEED):
        self.estimator, self.grid, self.n_splits, self.seed = estimator, grid, n_splits, seed

    def fit(self, X, y, groups):
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
        splits = list(cv.split(X, y, groups))
        self.search_ = GridSearchCV(self.estimator, self.grid, cv=splits, scoring="roc_auc",
                                    refit=True, error_score="raise")
        self.search_.fit(X, y)
        self.best_params_ = self.search_.best_params_
        return self

    def predict_proba(self, X):
        return self.search_.predict_proba(X)


def _pipeline(classifier, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("clf", classifier)])


# Small grids on purpose: 828 nodules cannot support a large search.
def make_logreg_l2():
    return GroupTuned(_pipeline(LogisticRegression(max_iter=5000)),
                      {"clf__C": [0.001, 0.01, 0.1, 1.0]})


def make_logreg_enet():
    base = LogisticRegression(solver="saga", max_iter=5000)     # l1_ratio set => elastic net
    return GroupTuned(_pipeline(base), {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.2, 0.5, 0.8]})


def make_random_forest():
    base = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    return GroupTuned(_pipeline(base, scale=False),
                      {"clf__min_samples_leaf": [1, 3, 8], "clf__max_features": ["sqrt", 0.3]})


def make_hist_gradient_boosting():
    base = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    return GroupTuned(_pipeline(base, scale=False),
                      {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [4, 8],
                       "clf__l2_regularization": [0.0, 1.0]})


def make_svm_rbf():
    base = SVC(kernel="rbf", probability=True, random_state=SEED)
    return GroupTuned(_pipeline(base), {"clf__C": [0.3, 1.0, 3.0, 10.0], "clf__gamma": ["scale", 0.003]})


# Stages 4 and 6 use MODEL_FACTORIES; stage 8 compares CANDIDATE_MODELS against it.
MODEL_FACTORIES = {"logreg": make_logreg}
CANDIDATE_MODELS = {"logreg_l2": make_logreg_l2, "logreg_enet": make_logreg_enet,
                    "random_forest": make_random_forest, "hist_gb": make_hist_gradient_boosting,
                    "svm_rbf": make_svm_rbf}
ALL_MODELS = {**MODEL_FACTORIES, **CANDIDATE_MODELS}

_SHAPE = FEATURE_PREFIX + "shape_"
_FIRSTORDER = FEATURE_PREFIX + "firstorder_"
_TEXTURE = tuple(f"{FEATURE_PREFIX}{name}_" for name in ("glcm", "glrlm", "glszm", "ngtdm", "gldm"))

INPUT_SETS = ("diameter", "scanner", "shape_only", "firstorder_only", "texture_only",
              "radiomics_no_shape", "radiomics", "combined")


# Extra input sets used by pipeline/07_compare.py (not run by the default stages 4 and 6).
EXTRA_INPUT_SETS = ("diameter_firstorder", "diameter_texture")


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
        "diameter_firstorder": ["diameter_mm"] + [c for c in features if c.startswith(_FIRSTORDER)],
        "diameter_texture": ["diameter_mm"] + [c for c in features if c.startswith(_TEXTURE)],
    }
    if name not in sets:
        raise KeyError(f"unknown input set {name!r}; choose from {sorted(sets)}")
    columns = sets[name]
    if not columns:
        raise ValueError(f"input set {name!r} has no columns in this table")
    return columns


def _fit(model, rows, columns):
    """Fit; estimators that tune hyperparameters also receive the patient ids of these rows."""
    if getattr(model, "needs_groups", False):
        model.fit(rows[columns], rows["malignant"], groups=rows["patient_id"])
    else:
        model.fit(rows[columns], rows["malignant"])


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
        _fit(model, table.loc[~validation], columns)
        oof[validation] = model.predict_proba(table.loc[validation, columns])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("missing out-of-fold predictions")
    return oof


def fit_predict(train, test, columns, make_model):
    """Fit on all of train, return probabilities for test."""
    model = make_model()
    _fit(model, train, columns)
    return model.predict_proba(test[columns])[:, 1]


def build_table(protocol, features, nodules, splits):
    """Features of one protocol joined to labels, diameter, scanner settings and the split."""
    table = features[features["protocol"] == protocol]
    table = table.merge(nodules[["voi_name", "malignant", "diameter_mm", "sx", "sz", "thickness"]],
                        on="voi_name", how="left", validate="one_to_one")
    table = table.merge(splits[["voi_name", "split", "fold"]],
                        on="voi_name", how="left", validate="one_to_one")
    if table[["malignant", "split", "fold"]].isna().any().any():
        raise ValueError("feature rows without a label or split; rerun stages 1-3")
    return table


def repeated_oof(trainval, columns, make_model, seeds, n_inner=N_INNER):
    """Out-of-fold probabilities under several different fold assignments.

    trainval rows only (the sealed test set is never involved). Returns an array of shape
    (len(trainval), len(seeds)): column j holds the predictions made with the folds of seeds[j].
    """
    if (trainval["split"] != "trainval").any():
        raise ValueError("repeated CV must use trainval rows only")
    trainval = trainval.reset_index(drop=True)
    out = np.empty((len(trainval), len(seeds)))
    for j, seed in enumerate(seeds):
        table = trainval.assign(fold=refold(trainval, seed, n_inner))
        out[:, j] = oof_predictions(table, columns, make_model)
    return out


def fit_predict_named(name, train, test, columns, members=ENSEMBLE_MEMBERS):
    """fit_predict by model name; 'ensemble_avg' fits every member and averages them."""
    if name == "ensemble_avg":
        return np.mean([fit_predict(train, test, columns, ALL_MODELS[m]) for m in members], axis=0)
    return fit_predict(train, test, columns, ALL_MODELS[name])
