"""Nested, patient-grouped selected-feature comparison using trainval only.

The script never opens test feature tables, predictions, or metrics. Feature
count selection happens inside each outer training fold using grouped inner CV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
FEATURES_FILE = ROOT / "results/radiomics_trainval/features_trainval.csv"
NODULES_FILE = ROOT / "data/processed/nodules.csv"
SPLITS_FILE = ROOT / "data/processed/splits.csv"
OUTPUT_DIR = ROOT / "results/radiomics"
PROTOCOLS = ("original_spacing", "isotropic_1mm")
OUTER_FOLDS = (0, 1, 2, 3, 4)
INNER_FOLDS = 4
K_CANDIDATES = (5, 10, 20, 50)
RANDOM_SEED = 42
META = {
    "voi_name", "patient_id", "malignant", "fold", "protocol",
    "original_spacing_x", "original_spacing_y", "original_spacing_z",
}


def make_estimator(feature_columns: list[str], model_input: str) -> Pipeline:
    if model_input == "diameter":
        transform = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        columns = ["diameter_mm"]
        transformers = [("diameter", transform, columns)]
    else:
        selected = model_input in {"radiomics_selected", "combined_selected"}
        radiomics_steps = [("imputer", SimpleImputer(strategy="median"))]
        if selected:
            radiomics_steps.append(("select", SelectKBest(score_func=f_classif)))
        radiomics_transform = Pipeline(radiomics_steps)
        transformers = [("radiomics", radiomics_transform, feature_columns)]
        if model_input in {"combined_all", "combined_selected"}:
            transformers.insert(0, (
                "diameter",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["diameter_mm"],
            ))

    features = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("features", features),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
    ])


def selected_feature_names(estimator: Pipeline, feature_columns: list[str], model_input: str) -> list[str]:
    if model_input not in {"radiomics_selected", "combined_selected"}:
        return []
    radiomics_pipeline = estimator.named_steps["features"].named_transformers_["radiomics"]
    selector = radiomics_pipeline.named_steps["select"]
    return [feature_columns[index] for index in selector.get_support(indices=True)]


def main() -> None:
    # Deliberately load only the trainval feature table. Do not add a test-file path here.
    features = pd.read_csv(FEATURES_FILE)
    splits = pd.read_csv(SPLITS_FILE).set_index("voi_name")
    feature_columns = [column for column in features.columns if column not in META]
    if len(feature_columns) != 107:
        raise ValueError(f"Expected 107 Radiomics features; found {len(feature_columns)}")

    expected_ids = set(splits.index[splits["split"] == "trainval"])
    nodules = pd.read_csv(NODULES_FILE)
    nodules = nodules.loc[nodules["voi_name"].isin(expected_ids)].set_index("voi_name")
    if set(nodules.index) != expected_ids:
        raise ValueError("Nodule metadata IDs do not exactly match the shared trainval split")
    if set(features["voi_name"]) != expected_ids:
        raise ValueError("Feature IDs do not exactly match the shared trainval split")
    if not features["protocol"].isin(PROTOCOLS).all():
        raise ValueError("Unexpected extraction protocol")
    if features.duplicated(["voi_name", "protocol"]).any():
        raise ValueError("Duplicate nodule/protocol feature rows")

    features["malignant"] = features["voi_name"].map(nodules["malignant"]).astype(int)
    features["diameter_mm"] = pd.to_numeric(
        features["voi_name"].map(nodules["diameter_mm"]), errors="raise"
    )
    features["patient_id"] = features["voi_name"].map(splits["patient_id"])
    features["fold"] = features["voi_name"].map(splits["fold"]).astype(int)
    if not (features.groupby("patient_id")["fold"].nunique() == 1).all():
        raise ValueError("A patient occurs in multiple trainval folds")
    if set(features["fold"]) != set(OUTER_FOLDS):
        raise ValueError("Expected shared outer folds 0–4")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_inputs = (
        "diameter", "radiomics_all", "combined_all",
        "radiomics_selected", "combined_selected",
    )
    oof: dict[tuple[str, str], np.ndarray] = {}
    fold_rows = []
    feature_rows = []

    for protocol in PROTOCOLS:
        table = (
            features.loc[features["protocol"] == protocol]
            .sort_values("voi_name")
            .reset_index(drop=True)
        )
        if table["voi_name"].nunique() != len(expected_ids):
            raise ValueError(f"Incomplete protocol rows for {protocol}")
        y = table["malignant"].to_numpy(dtype=int)
        groups = table["patient_id"].to_numpy()

        for model_input in model_inputs:
            key = (protocol, model_input)
            oof[key] = np.full(len(table), np.nan)

        for outer_fold in OUTER_FOLDS:
            valid_mask = table["fold"].to_numpy() == outer_fold
            fit_mask = ~valid_mask
            fit = table.loc[fit_mask]
            valid = table.loc[valid_mask]
            if set(fit["patient_id"]) & set(valid["patient_id"]):
                raise ValueError(f"Patient leakage in outer fold {outer_fold}")

            inner_cv = StratifiedGroupKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED
            )
            for model_input in model_inputs:
                selected = model_input.endswith("_selected")
                estimator = make_estimator(feature_columns, model_input)

                if selected:
                    parameter_name = "features__radiomics__select__k"
                    search = GridSearchCV(
                        estimator,
                        param_grid={parameter_name: list(K_CANDIDATES)},
                        scoring="roc_auc",
                        cv=inner_cv,
                        refit=True,
                        n_jobs=1,
                        error_score="raise",
                    )
                    search.fit(
                        fit, fit["malignant"], groups=fit["patient_id"]
                    )
                    fitted = search.best_estimator_
                    best_k = int(search.best_params_[parameter_name])
                    best_inner_auc = float(search.best_score_)
                else:
                    fitted = estimator.fit(fit, fit["malignant"])
                    best_k = 107 if "radiomics" in model_input else 0
                    best_inner_auc = np.nan

                probability = fitted.predict_proba(valid)[:, 1]
                oof[(protocol, model_input)][np.flatnonzero(valid_mask)] = probability
                fold_rows.append({
                    "protocol": protocol,
                    "model_input": model_input,
                    "outer_fold": outer_fold,
                    "selected_k": best_k,
                    "inner_best_roc_auc": best_inner_auc,
                })
                for name in selected_feature_names(fitted, feature_columns, model_input):
                    feature_rows.append({
                        "protocol": protocol,
                        "model_input": model_input,
                        "outer_fold": outer_fold,
                        "selected_k": best_k,
                        "feature": name,
                    })

    metric_rows = []
    for (protocol, model_input), probability in oof.items():
        table = features.loc[features["protocol"] == protocol].sort_values("voi_name")
        y = table["voi_name"].map(nodules["malignant"]).to_numpy(dtype=int)
        if not np.isfinite(probability).all():
            raise ValueError(f"Missing OOF predictions for {protocol}, {model_input}")
        metric_rows.append({
            "protocol": protocol,
            "model_input": model_input,
            "n_trainval": len(y),
            "outer_oof_roc_auc": float(roc_auc_score(y, probability)),
            "outer_oof_average_precision": float(average_precision_score(y, probability)),
        })

    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    selected_features = pd.DataFrame(
        feature_rows,
        columns=["protocol", "model_input", "outer_fold", "selected_k", "feature"],
    )
    metrics.to_csv(OUTPUT_DIR / "selected_feature_metrics_trainval.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "selected_feature_k_by_fold_trainval.csv", index=False)
    selected_features.to_csv(
        OUTPUT_DIR / "selected_features_by_fold_trainval.csv", index=False
    )
    print(metrics.to_string(index=False))
    print(f"\nSaved trainval-only metrics and feature selections to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
