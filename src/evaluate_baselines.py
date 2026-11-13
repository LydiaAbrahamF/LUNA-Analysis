"""Evaluate pre-specified diameter/Radiomics baselines on the shared patient split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
TRAIN_FILE = ROOT / "results/radiomics_trainval/features_trainval.csv"
TEST_FILE = ROOT / "results/radiomics_test/features_test.csv"
NODULES_FILE = ROOT / "data/processed/nodules.csv"
SPLITS_FILE = ROOT / "data/processed/splits.csv"
OUTPUT = ROOT / "results/baseline_evaluation"
PROTOCOLS = ("original_spacing", "isotropic_1mm")
MODEL_INPUTS = ("diameter", "radiomics", "combined")
META = {
    "voi_name", "patient_id", "malignant", "fold", "protocol",
    "original_spacing_x", "original_spacing_y", "original_spacing_z",
}
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 42


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"),
    )


def metric_values(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(y, probability)),
        float(average_precision_score(y, probability)),
    )


def patient_bootstrap(test: pd.DataFrame, probability: np.ndarray) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    patient_ids = test["patient_id"].drop_duplicates().to_numpy()
    indices_by_patient = {
        patient: np.flatnonzero(test["patient_id"].to_numpy() == patient)
        for patient in patient_ids
    }
    y = test["malignant"].to_numpy(dtype=int)
    aucs, aps = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        ix = np.concatenate([indices_by_patient[p] for p in sampled])
        if np.unique(y[ix]).size < 2:
            continue
        auc, ap = metric_values(y[ix], probability[ix])
        aucs.append(auc)
        aps.append(ap)
    return {
        "test_roc_auc_ci_low": float(np.percentile(aucs, 2.5)),
        "test_roc_auc_ci_high": float(np.percentile(aucs, 97.5)),
        "test_average_precision_ci_low": float(np.percentile(aps, 2.5)),
        "test_average_precision_ci_high": float(np.percentile(aps, 97.5)),
        "bootstrap_valid_replicates": len(aucs),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    nodules = pd.read_csv(NODULES_FILE)
    splits = pd.read_csv(SPLITS_FILE)
    feature_columns = [c for c in train.columns if c not in META]

    if len(feature_columns) != 107:
        raise ValueError(f"Expected 107 Radiomics features, got {len(feature_columns)}")
    if train.duplicated(["voi_name", "protocol"]).any() or test.duplicated(["voi_name", "protocol"]).any():
        raise ValueError("Duplicate VOI/protocol rows in an extraction table")

    # Recheck split integrity immediately before evaluation.
    split_map = splits.set_index("voi_name")
    if set(train["voi_name"]) != set(split_map.index[split_map["split"] == "trainval"]):
        raise ValueError("Trainval feature IDs do not match the shared trainval split")
    if set(test["voi_name"]) != set(split_map.index[split_map["split"] == "test"]):
        raise ValueError("Test feature IDs do not match the shared test split")
    if set(train["patient_id"]) & set(test["patient_id"]):
        raise ValueError("Patient leakage between trainval and test")
    if not set(train["fold"].astype(int)).issubset({0, 1, 2, 3, 4}):
        raise ValueError("Unexpected trainval fold values")
    if not (train.groupby("patient_id")["fold"].nunique() == 1).all():
        raise ValueError("A patient occurs in more than one trainval fold")

    nodule_map = nodules.set_index("voi_name")
    for table, expected_split in ((train, "trainval"), (test, "test")):
        labels = table["voi_name"].map(nodule_map["malignant"])
        diameter = table["voi_name"].map(nodule_map["diameter_mm"])
        if labels.isna().any() or diameter.isna().any():
            raise ValueError(f"Missing label or diameter in {expected_split}")
        table["malignant"] = labels.astype(int)
        table["diameter_mm"] = pd.to_numeric(diameter, errors="raise")
        table["patient_id"] = table["voi_name"].map(split_map["patient_id"])
        table["fold"] = table["voi_name"].map(split_map["fold"]).astype(int)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = []
    prediction_rows = []
    for protocol in PROTOCOLS:
        tr = train.loc[train.protocol == protocol].sort_values("voi_name").reset_index(drop=True)
        te = test.loc[test.protocol == protocol].sort_values("voi_name").reset_index(drop=True)
        if len(tr) != train["voi_name"].nunique() or len(te) != test["voi_name"].nunique():
            raise ValueError(f"Unpaired protocol rows for {protocol}")

        for model_input in MODEL_INPUTS:
            columns = {
                "diameter": ["diameter_mm"],
                "radiomics": feature_columns,
                "combined": ["diameter_mm", *feature_columns],
            }[model_input]
            oof_probability = np.full(len(tr), np.nan)
            for fold in (0, 1, 2, 3, 4):
                validation_mask = tr["fold"].to_numpy() == fold
                fit_mask = ~validation_mask
                if set(tr.loc[fit_mask, "patient_id"]) & set(tr.loc[validation_mask, "patient_id"]):
                    raise ValueError(f"Patient leakage in fold {fold}")
                model = make_model()
                model.fit(tr.loc[fit_mask, columns], tr.loc[fit_mask, "malignant"])
                oof_probability[validation_mask] = model.predict_proba(
                    tr.loc[validation_mask, columns]
                )[:, 1]
            if not np.isfinite(oof_probability).all():
                raise ValueError("Missing out-of-fold predictions")
            cv_auc, cv_ap = metric_values(tr["malignant"].to_numpy(int), oof_probability)

            final_model = make_model()
            final_model.fit(tr[columns], tr["malignant"])
            test_probability = final_model.predict_proba(te[columns])[:, 1]
            test_auc, test_ap = metric_values(te["malignant"].to_numpy(int), test_probability)
            intervals = patient_bootstrap(te, test_probability)
            metrics.append({
                "protocol": protocol,
                "model_input": model_input,
                "n_trainval": len(tr),
                "n_test": len(te),
                "cv_roc_auc": cv_auc,
                "cv_average_precision": cv_ap,
                "test_roc_auc": test_auc,
                "test_average_precision": test_ap,
                **intervals,
            })
            prediction_rows.extend({
                "voi_name": voi,
                "protocol": protocol,
                "model_input": model_input,
                "malignant": int(y),
                "probability_malignant": float(prob),
            } for voi, y, prob in zip(te["voi_name"], te["malignant"], test_probability))

    pd.DataFrame(metrics).to_csv(OUTPUT / "baseline_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved metrics and final test predictions to {OUTPUT}")


if __name__ == "__main__":
    main()
