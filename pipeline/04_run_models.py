"""Stage 4: baselines and models.

Default: 5-fold patient-grouped cross-validation on trainval ONLY, with patient-level
bootstrap 95% intervals. Safe to run repeatedly. (Seed robustness: pipeline/06_repeated_cv.py.)
This stage never touches the sealed test set. The one test run is pipeline/10_final_test.py.

Writes results/tables/cv_metrics.csv.
"""
import argparse
import sys

import pandas as pd

from lunarad import config as cfg
from lunarad.evaluation import metric_values, patient_bootstrap
from lunarad.features import load_feature_table
from lunarad.models import INPUT_SETS, MODEL_FACTORIES, build_table, columns_for, oof_predictions
from lunarad.splits import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES))
    parser.add_argument("--inputs", nargs="+", default=list(INPUT_SETS))
    args = parser.parse_args()

    if args.final:
        sys.exit("--final was removed: it scored every input set on the test set. "
                 "Use pipeline/10_final_test.py, which scores only the pre-declared models.")

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    validate_split(nodules, splits)
    features = load_feature_table(cfg.FEATURES_CSV)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    cv_rows = []
    for protocol in cfg.PROTOCOLS:
        table = build_table(protocol, features, nodules, splits)
        trainval = table[table["split"] == "trainval"].reset_index(drop=True)
        for input_name in args.inputs:
            columns = columns_for(input_name, table)
            for model_name in args.models:
                factory = MODEL_FACTORIES[model_name]
                oof = oof_predictions(trainval, columns, factory)
                auc, ap = metric_values(trainval["malignant"], oof)
                interval = patient_bootstrap(trainval["malignant"], oof, trainval["patient_id"])
                cv_rows.append({"protocol": protocol, "input": input_name, "model": model_name,
                                "n_features": len(columns), "n_trainval": len(trainval),
                                "cv_roc_auc": auc, "cv_average_precision": ap,
                                **{f"cv_{k}": v for k, v in interval.items()}})

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(cfg.TABLES / "cv_metrics.csv", index=False)
    print(cv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
