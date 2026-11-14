"""Stage 9: is the probability trustworthy, and where do we draw the decision line?

Uses ONLY the out-of-fold predictions written by stage 8 (trainval nodules). The sealed test set
is not used. Thresholds chosen here are fixed before the final test run.

Two ways to choose the decision threshold t (flag a nodule as suspicious when p >= t):
  recall target  t = largest value whose cross-validated sensitivity is >= the target
  cost ratio k   t = 1 / (1 + k): a missed malignancy is k times worse than a false alarm
                 (only meaningful if the probabilities are calibrated)
The label rule (mean rating >= 3) is untouched; this is a separate, later decision.

Writes results/tables/calibration_table.csv, calibration_summary.csv, threshold_analysis.csv
       results/figures/calibration.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lunarad import config as cfg
from lunarad.evaluation import (calibration_summary, interval, patient_bootstrap_stats,
                                threshold_for_recall, threshold_metrics)


def analyse(label, data, bootstrap):
    y, p, ids = data["malignant"].to_numpy(), data["probability_malignant"].to_numpy(), data["patient_id"]
    cal_table, cal_stats = calibration_summary(y, p, cfg.CALIBRATION_BINS)
    rows = []
    rules = ([(f"recall >= {t:.2f}", threshold_for_recall(y, p, t), t) for t in cfg.RECALL_TARGETS]
             + [(f"cost ratio k = {k}", 1.0 / (1.0 + k), None) for k in cfg.COST_RATIOS])
    for rule, threshold, target in rules:
        point = threshold_metrics(y, p, threshold)
        boot = patient_bootstrap_stats(y, p, ids, lambda yy, pp, t=threshold: threshold_metrics(yy, pp, t),
                                       replicates=bootstrap)
        row = {"model": label, "rule": rule, "threshold": threshold, **point}
        for metric in ("sensitivity", "specificity", "ppv"):
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = interval(boot[metric])
        if target is not None:   # how much would the threshold itself move with other patients?
            moved = patient_bootstrap_stats(y, p, ids, lambda yy, pp, g=target: {
                "t": threshold_for_recall(yy, pp, g)}, replicates=bootstrap)
            row["threshold_ci_low"], row["threshold_ci_high"] = interval(moved["t"])
        rows.append(row)
    return cal_table, cal_stats, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", nargs=2, metavar=("INPUT", "MODEL"), default=cfg.PRIMARY_MODEL)
    parser.add_argument("--secondary", nargs=2, metavar=("INPUT", "MODEL"),
                        default=cfg.SECONDARY_MODEL)
    parser.add_argument("--no-secondary", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    oof = pd.read_csv(cfg.TABLES / "oof_predictions.csv")
    chosen = [("primary", *args.primary)] + ([] if args.no_secondary else [("secondary", *args.secondary)])
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    tables, summaries, thresholds = [], [], []
    for role, input_name, model in chosen:
        data = oof[(oof["input"] == input_name) & (oof["model"] == model)
                   & (oof["protocol"] == cfg.FINAL_PROTOCOL)]
        if data.empty:
            raise SystemExit(f"no out-of-fold predictions for {input_name} / {model}; run stage 8")
        label = f"{role}: {model} on {input_name}"
        cal_table, cal_stats, rules = analyse(label, data, args.bootstrap)
        tables.append(cal_table.assign(model=label))
        summaries.append({"model": label, "n": len(data), **cal_stats})
        thresholds.append(rules)
        ax.plot(cal_table["mean_predicted"], cal_table["observed_rate"], "o-", label=label)
    ax.set_xlabel("predicted probability of malignancy")
    ax.set_ylabel("observed malignant share")
    ax.set_title("Calibration (out-of-fold, trainval)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "calibration.png", dpi=200)

    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(cfg.TABLES / "calibration_table.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(cfg.TABLES / "calibration_summary.csv", index=False)
    result = pd.concat(thresholds)
    result.to_csv(cfg.TABLES / "threshold_analysis.csv", index=False)

    print("Calibration (slope 1 / intercept 0 = perfect; slope < 1 = over-confident):")
    print(summary.round(3).to_string(index=False))
    show = ["rule", "threshold", "sensitivity", "sensitivity_ci_low", "sensitivity_ci_high",
            "specificity", "specificity_ci_low", "specificity_ci_high", "ppv", "flagged_share"]
    for label, group in result.groupby("model", sort=False):
        print(f"\n{label}")
        print(group[show].round(3).to_string(index=False))
    print("\nThreshold uncertainty for the recall rules (95% interval of the cutoff itself):")
    print(result.dropna(subset=["threshold_ci_low"])[["model", "rule", "threshold",
          "threshold_ci_low", "threshold_ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
