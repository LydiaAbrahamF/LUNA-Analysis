"""Stage 10: the one test-set run.

Prerequisites (all from cross-validation, before this stage):
  * model, input set and protocol chosen  -> config.PRIMARY_MODEL / SECONDARY_MODEL / FINAL_PROTOCOL
  * decision threshold fixed (stage 9)    -> config.FINAL_THRESHOLDS
  * all of the above written, dated, in docs/decisions.md

This stage fits the primary and the secondary model on ALL trainval nodules and scores the sealed
test nodules once. A diameter-only logistic regression is scored as a reference (not a candidate).
It refuses a second run: after one run the test set is no longer sealed.

Writes results/tables/final_test_metrics.csv, final_test_threshold_metrics.csv,
       final_test_paired.csv, final_test_predictions.csv
"""
import argparse
import sys

import pandas as pd

from lunarad import config as cfg
from lunarad.evaluation import (calibration_summary, interval, metric_values, patient_bootstrap,
                                patient_bootstrap_difference, patient_bootstrap_stats,
                                threshold_metrics)
from lunarad.features import load_feature_table
from lunarad.models import build_table, columns_for, fit_predict_named
from lunarad.splits import file_hash, validate_split

SEAL = cfg.TABLES / "final_test_metrics.csv"


def check_threshold_against_stage9(label, threshold):
    """The frozen threshold must match what stage 9 produced for the same model."""
    path = cfg.TABLES / "threshold_analysis.csv"
    if not path.exists():
        print(f"warning: {path.name} not found; cannot verify the {label} threshold")
        return
    table = pd.read_csv(path)
    row = table[(table["model"] == label) & (table["rule"] == "recall >= 0.95")]
    if row.empty:
        print(f"warning: no stage 9 row for '{label}'; threshold {threshold} not verified")
    elif abs(float(row["threshold"].iloc[0]) - threshold) > 0.0015:
        sys.exit(f"threshold mismatch for {label}: config {threshold}, stage 9 "
                 f"{float(row['threshold'].iloc[0]):.4f}. Rerun stage 9 or fix config.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="I have recorded the model, input and threshold in docs/decisions.md")
    parser.add_argument("--force-final", action="store_true",
                        help="allow a second pass (the test set is then no longer sealed)")
    parser.add_argument("--expect-sha256", help="refuse to run if splits.csv has another hash")
    parser.add_argument("--primary", nargs=2, metavar=("INPUT", "MODEL"), default=cfg.PRIMARY_MODEL)
    parser.add_argument("--secondary", nargs=2, metavar=("INPUT", "MODEL"),
                        default=cfg.SECONDARY_MODEL)
    parser.add_argument("--primary-threshold", type=float, default=cfg.FINAL_THRESHOLDS["primary"])
    parser.add_argument("--secondary-threshold", type=float,
                        default=cfg.FINAL_THRESHOLDS["secondary"])
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    if not args.confirm:
        sys.exit("Refusing to touch the test set without --confirm.\n"
                 f"This will score primary {tuple(args.primary)} at threshold "
                 f"{args.primary_threshold} and secondary {tuple(args.secondary)} at "
                 f"{args.secondary_threshold}, once. Record them in docs/decisions.md first.")
    if SEAL.exists() and not args.force_final:
        sys.exit(f"{SEAL} already exists: the test set has been used. "
                 "Pass --force-final only if you accept it is no longer sealed.")

    digest = file_hash(cfg.SPLITS_CSV)
    print(f"splits.csv sha256 {digest}")
    if args.expect_sha256 and digest != args.expect_sha256:
        sys.exit("splits.csv does not match --expect-sha256: the split changed after it was locked")

    nodules = pd.read_csv(cfg.NODULES_CSV)
    splits = pd.read_csv(cfg.SPLITS_CSV)
    validate_split(nodules, splits)
    features = load_feature_table(cfg.FEATURES_CSV)
    table = build_table(cfg.FINAL_PROTOCOL, features, nodules, splits)
    trainval = table[table["split"] == "trainval"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    y, patients = test["malignant"].to_numpy(), test["patient_id"]
    print(f"fit on {len(trainval)} trainval nodules, score {len(test)} test nodules "
          f"({patients.nunique()} patients, {int(y.sum())} malignant)")

    roles = [("primary", *args.primary, args.primary_threshold),
             ("secondary", *args.secondary, args.secondary_threshold),
             ("reference", *cfg.REFERENCE_MODEL, None)]
    probabilities, metric_rows, threshold_rows, prediction_rows = {}, [], [], []
    for role, input_name, model, threshold in roles:
        label = f"{role}: {model} on {input_name}"
        if threshold is not None:
            check_threshold_against_stage9(label, threshold)
        probability = fit_predict_named(model, trainval, test, columns_for(input_name, table))
        probabilities[role] = probability
        auc, ap = metric_values(y, probability)
        ci = patient_bootstrap(y, probability, patients, replicates=args.bootstrap)
        _, cal = calibration_summary(y, probability)
        metric_rows.append({"role": role, "input": input_name, "model": model,
                            "protocol": cfg.FINAL_PROTOCOL, "n_test": len(test),
                            "n_patients": int(patients.nunique()), "n_malignant": int(y.sum()),
                            "roc_auc": auc, "average_precision": ap, **ci, **cal,
                            "forced_second_pass": bool(args.force_final and SEAL.exists())})
        prediction_rows.append(pd.DataFrame({
            "voi_name": test["voi_name"], "patient_id": test["patient_id"], "role": role,
            "malignant": y, "probability_malignant": probability}))
        if threshold is not None:
            point = threshold_metrics(y, probability, threshold)
            boot = patient_bootstrap_stats(
                y, probability, patients, lambda yy, pp, t=threshold: threshold_metrics(yy, pp, t),
                replicates=args.bootstrap)
            row = {"role": role, "input": input_name, "model": model, "threshold": threshold,
                   "target_recall": 0.95, **point}
            for name in ("sensitivity", "specificity", "ppv", "npv"):
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = interval(boot[name])
            row["target_inside_sensitivity_ci"] = (
                row["sensitivity_ci_low"] <= 0.95 <= row["sensitivity_ci_high"])
            threshold_rows.append(row)

    paired = []
    for role in ("primary", "secondary"):
        diff = patient_bootstrap_difference(y, probabilities[role], probabilities["reference"],
                                            patients, replicates=args.bootstrap)
        paired.append({"comparison": f"{role} - reference (diameter only)", **diff})
    diff = patient_bootstrap_difference(y, probabilities["secondary"], probabilities["primary"],
                                        patients, replicates=args.bootstrap)
    paired.append({"comparison": "secondary - primary", **diff})

    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics["splits_sha256"] = digest
    thresholds = pd.DataFrame(threshold_rows)
    pd.DataFrame(paired).to_csv(cfg.TABLES / "final_test_paired.csv", index=False)
    thresholds.to_csv(cfg.TABLES / "final_test_threshold_metrics.csv", index=False)
    pd.concat(prediction_rows).to_csv(cfg.TABLES / "final_test_predictions.csv", index=False)
    metrics.to_csv(SEAL, index=False)      # written last: its existence is the seal

    show = ["role", "model", "input", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
            "average_precision", "brier", "calibration_slope"]
    print("\nTest-set discrimination (95% patient-bootstrap CI):")
    print(metrics[show].round(3).to_string(index=False))
    show = ["role", "threshold", "sensitivity", "sensitivity_ci_low", "sensitivity_ci_high",
            "specificity", "specificity_ci_low", "specificity_ci_high", "ppv", "npv",
            "flagged_share", "target_inside_sensitivity_ci"]
    print("\nAt the pre-declared recall-95% threshold:")
    print(thresholds[show].round(3).to_string(index=False))
    print("\nPaired AUC differences on the test set:")
    print(pd.DataFrame(paired).drop(columns="bootstrap_valid_replicates").round(3).to_string(index=False))
    print(f"\nsealed test set used; wrote {SEAL}")


if __name__ == "__main__":
    main()
