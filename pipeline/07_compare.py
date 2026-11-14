"""Stage 7: do radiomics add anything beyond nodule size, and which protocol is better?

Cross-validation on trainval only (the sealed test set is never used). Three analyses:

 1. paired AUC differences with patient-bootstrap 95% intervals
      -> results/tables/paired_differences.csv
 2. how strongly each radiomics feature tracks nodule diameter (Spearman), per feature family
      -> results/tables/feature_size_correlation.csv
 3. AUC inside diameter tertiles: does the model separate malignant from benign among nodules
    of similar size?  -> results/tables/size_stratified_auc.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from lunarad import config as cfg
from lunarad.evaluation import patient_bootstrap, patient_bootstrap_difference
from lunarad.features import load_feature_table
from lunarad.models import MODEL_FACTORIES, build_table, columns_for, repeated_oof
from lunarad.splits import validate_split

PRIMARY, OTHER = "original_spacing", "isotropic_1mm"
# (name, protocol of A, input of A, protocol of B, input of B): A minus B
COMPARISONS = [
    ("radiomics vs diameter", PRIMARY, "radiomics", PRIMARY, "diameter"),
    ("combined vs diameter", PRIMARY, "combined", PRIMARY, "diameter"),
    ("texture_only vs diameter", PRIMARY, "texture_only", PRIMARY, "diameter"),
    ("firstorder_only vs diameter", PRIMARY, "firstorder_only", PRIMARY, "diameter"),
    ("shape_only vs diameter", PRIMARY, "shape_only", PRIMARY, "diameter"),
    ("diameter+texture vs diameter", PRIMARY, "diameter_texture", PRIMARY, "diameter"),
    ("diameter+firstorder vs diameter", PRIMARY, "diameter_firstorder", PRIMARY, "diameter"),
    ("radiomics vs diameter+texture", PRIMARY, "radiomics", PRIMARY, "diameter_texture"),
    ("radiomics: original vs isotropic", PRIMARY, "radiomics", OTHER, "radiomics"),
    ("firstorder: original vs isotropic", PRIMARY, "firstorder_only", OTHER, "firstorder_only"),
    ("texture: original vs isotropic", PRIMARY, "texture_only", OTHER, "texture_only"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=cfg.N_REPEATS)
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    parser.add_argument("--model", default="logreg", choices=list(MODEL_FACTORIES))
    args = parser.parse_args()
    seeds = [cfg.SEED + i for i in range(args.repeats)]
    factory = MODEL_FACTORIES[args.model]

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    validate_split(nodules, splits)
    features = load_feature_table(cfg.FEATURES_CSV)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    trainval = {}
    for protocol in cfg.PROTOCOLS:
        table = build_table(protocol, features, nodules, splits)
        # same row order in both protocols, so identical seeds give identical folds
        trainval[protocol] = (table[table["split"] == "trainval"]
                              .sort_values("voi_name").reset_index(drop=True))
    reference = trainval[PRIMARY]
    assert (reference["voi_name"] == trainval[OTHER]["voi_name"]).all()
    y, patients = reference["malignant"].to_numpy(), reference["patient_id"]

    cache = {}

    def predictions(protocol, input_name):
        key = (protocol, input_name)
        if key not in cache:
            table = trainval[protocol]
            cache[key] = repeated_oof(table, columns_for(input_name, table), factory, seeds)
        return cache[key]

    # 1. paired differences
    rows = []
    for name, pa, ia, pb, ib in COMPARISONS:
        result = patient_bootstrap_difference(y, predictions(pa, ia), predictions(pb, ib),
                                              patients, replicates=args.bootstrap)
        rows.append({"comparison": name, **result})
        print(f"{name:36s} dAUC {result['auc_difference']:+.3f}  "
              f"95% CI [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]  "
              f"p(not better) {result['p_not_better']:.3f}")
    pd.DataFrame(rows).to_csv(cfg.TABLES / "paired_differences.csv", index=False)

    # 2. feature vs size
    feature_cols = columns_for("radiomics", reference)
    records = []
    for column in feature_cols:
        rho = spearmanr(reference[column], reference["diameter_mm"], nan_policy="omit")[0]
        family = column.split("_")[1]
        try:
            auc = roc_auc_score(y, reference[column].fillna(reference[column].median()))
        except ValueError:
            auc = np.nan
        records.append({"feature": column, "family": family, "spearman_with_diameter": rho,
                        "abs_spearman": abs(rho), "univariate_auc": auc,
                        "univariate_auc_oriented": max(auc, 1 - auc)})
    corr = pd.DataFrame(records).sort_values("abs_spearman", ascending=False)
    corr.to_csv(cfg.TABLES / "feature_size_correlation.csv", index=False)
    summary = corr.groupby("family").agg(
        n=("feature", "size"), median_abs_rho=("abs_spearman", "median"),
        share_above_0_7=("abs_spearman", lambda s: float((s > 0.7).mean())))
    print("\nHow strongly each feature family tracks diameter (|Spearman|):")
    print(summary.round(2).to_string())

    # 3. AUC within diameter tertiles
    tertile = pd.qcut(reference["diameter_mm"], 3, labels=["small", "medium", "large"])
    rows = []
    for input_name in ("diameter", "radiomics", "texture_only"):
        probability = predictions(PRIMARY, input_name).mean(axis=1)
        for group in ("small", "medium", "large"):
            mask = (tertile == group).to_numpy()
            both_classes = np.unique(y[mask]).size == 2
            ci = (patient_bootstrap(y[mask], probability[mask], patients[mask].to_numpy(),
                                    replicates=args.bootstrap) if both_classes else {})
            rows.append({"input": input_name, "diameter_tertile": group, "n": int(mask.sum()),
                         "diameter_range_mm": f"{reference.loc[mask, 'diameter_mm'].min():.1f}-"
                                              f"{reference.loc[mask, 'diameter_mm'].max():.1f}",
                         "malignant_share": float(y[mask].mean()),
                         "roc_auc": roc_auc_score(y[mask], probability[mask]) if both_classes
                         else np.nan,
                         "roc_auc_ci_low": ci.get("roc_auc_ci_low", np.nan),
                         "roc_auc_ci_high": ci.get("roc_auc_ci_high", np.nan)})
    strat = pd.DataFrame(rows)
    strat.to_csv(cfg.TABLES / "size_stratified_auc.csv", index=False)
    print("\nAUC inside diameter tertiles (repeat-averaged out-of-fold predictions):")
    print(strat.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
