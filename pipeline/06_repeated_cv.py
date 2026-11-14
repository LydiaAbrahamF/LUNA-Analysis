"""Stage 6: repeated cross-validation, the seed-robustness check.

The locked split fixes ONE assignment of patients to folds. This stage repeats the 5-fold
cross-validation on trainval under N_REPEATS different fold assignments (seeds 42, 43, ...),
so you can show your conclusions do not depend on one lucky or unlucky seed.
The sealed test chunk is never involved.

Writes results/tables/cv_repeats.csv          one row per (protocol, input, model, seed)
       results/tables/cv_repeats_summary.csv  mean/std/min/max over seeds + 95% patient-bootstrap CI
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lunarad import config as cfg
from lunarad.evaluation import patient_bootstrap_repeats
from lunarad.features import load_feature_table
from lunarad.models import INPUT_SETS, MODEL_FACTORIES, build_table, columns_for, repeated_oof
from lunarad.splits import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=cfg.N_REPEATS)
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES,
                        help="bootstrap replicates for the interval (lower = faster)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES))
    parser.add_argument("--inputs", nargs="+", default=list(INPUT_SETS))
    args = parser.parse_args()
    seeds = [cfg.SEED + i for i in range(args.repeats)]

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    validate_split(nodules, splits)
    features = load_feature_table(cfg.FEATURES_CSV)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)

    per_seed, summary = [], []
    for protocol in cfg.PROTOCOLS:
        table = build_table(protocol, features, nodules, splits)
        trainval = table[table["split"] == "trainval"].reset_index(drop=True)
        y = trainval["malignant"].to_numpy()
        for input_name in args.inputs:
            columns = columns_for(input_name, table)
            for model_name in args.models:
                probs = repeated_oof(trainval, columns, MODEL_FACTORIES[model_name], seeds)
                aucs = [roc_auc_score(y, probs[:, j]) for j in range(len(seeds))]
                aps = [average_precision_score(y, probs[:, j]) for j in range(len(seeds))]
                for seed, auc, ap in zip(seeds, aucs, aps):
                    per_seed.append({"protocol": protocol, "input": input_name,
                                     "model": model_name, "seed": seed,
                                     "roc_auc": auc, "average_precision": ap})
                interval = patient_bootstrap_repeats(y, probs, trainval["patient_id"],
                                                     replicates=args.bootstrap)
                summary.append({"protocol": protocol, "input": input_name, "model": model_name,
                                "n_features": len(columns), "n_repeats": len(seeds),
                                "roc_auc_mean": np.mean(aucs), "roc_auc_std_over_seeds": np.std(aucs),
                                "roc_auc_min": np.min(aucs), "roc_auc_max": np.max(aucs),
                                "average_precision_mean": np.mean(aps),
                                "roc_auc_ci_low": interval["roc_auc_ci_low"],
                                "roc_auc_ci_high": interval["roc_auc_ci_high"]})
                print(f"{protocol:17s} {input_name:20s} {model_name:8s} "
                      f"AUC {np.mean(aucs):.3f}  (seeds {np.min(aucs):.3f}-{np.max(aucs):.3f}, "
                      f"95% CI {interval['roc_auc_ci_low']:.3f}-{interval['roc_auc_ci_high']:.3f})")

    pd.DataFrame(per_seed).to_csv(cfg.TABLES / "cv_repeats.csv", index=False)
    pd.DataFrame(summary).to_csv(cfg.TABLES / "cv_repeats_summary.csv", index=False)
    print(f"\nwrote {cfg.TABLES / 'cv_repeats_summary.csv'}")


if __name__ == "__main__":
    main()
