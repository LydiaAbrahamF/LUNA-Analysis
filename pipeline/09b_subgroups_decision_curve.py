"""Stage 9b: do the results hold in every scanner group, and is the model worth using?

Uses ONLY the out-of-fold predictions written by stage 8 (trainval nodules). The test set is not used.

1. Subgroups by slice thickness. The split is stratified by thickness because thick-slice scans
   contain larger nodules. Here the AUC is reported inside each thickness group, with patient-level
   bootstrap intervals. Small groups give wide intervals; read them as a check, not as a ranking.
2. Decision curve (Vickers and Elkin, 2006). For a range of decision thresholds t, the net benefit of
   using the model is compared with flagging every nodule and flagging none. A model is useful at
   threshold t only if its net benefit is above both.

Writes results/tables/subgroup_auc.csv, decision_curve.csv and results/figures/decision_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lunarad import config as cfg
from lunarad.evaluation import interval, net_benefit, patient_bootstrap_stats


def subgroup_rows(label, data, bootstrap):
    rows = []
    groups = [("all", data)] + [(str(g), part) for g, part in data.groupby("thickness", observed=True)]
    for name, part in groups:
        y, p = part["malignant"].to_numpy(), part["probability_malignant"].to_numpy()
        row = {"model": label, "thickness": name, "n_nodules": len(part),
               "n_patients": part["patient_id"].nunique(), "n_malignant": int(y.sum())}
        if np.unique(y).size < 2:                      # AUC is undefined with one class
            row.update(roc_auc=np.nan, roc_auc_ci_low=np.nan, roc_auc_ci_high=np.nan)
        else:
            boot = patient_bootstrap_stats(y, p, part["patient_id"],
                                           lambda yy, pp: {"auc": roc_auc_score(yy, pp)}, replicates=bootstrap)
            low, high = interval(boot["auc"])
            row.update(roc_auc=float(roc_auc_score(y, p)), roc_auc_ci_low=low, roc_auc_ci_high=high)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", nargs=2, metavar=("INPUT", "MODEL"), default=cfg.PRIMARY_MODEL)
    parser.add_argument("--secondary", nargs=2, metavar=("INPUT", "MODEL"), default=cfg.SECONDARY_MODEL)
    parser.add_argument("--no-secondary", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    oof = pd.read_csv(cfg.TABLES / "oof_predictions.csv")
    nodules = pd.read_csv(cfg.NODULES_CSV)[["voi_name", "thickness"]]
    oof = oof[oof["protocol"] == cfg.FINAL_PROTOCOL].merge(nodules, on="voi_name", how="left", validate="m:1")
    assert oof["thickness"].notna().all(), "a prediction has no thickness group"

    chosen = [("primary", *args.primary)] + ([] if args.no_secondary else [("secondary", *args.secondary)])
    thresholds = np.round(np.arange(0.05, 0.61, 0.01), 2)
    subgroup, curve = [], []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (role, input_name, model) in enumerate(chosen):
        data = oof[(oof["input"] == input_name) & (oof["model"] == model)]
        if data.empty:
            raise SystemExit(f"no out-of-fold predictions for {input_name} / {model}; run stage 8")
        label = f"{role}: {model} on {input_name}"
        subgroup += subgroup_rows(label, data, args.bootstrap)
        y, p = data["malignant"].to_numpy(), data["probability_malignant"].to_numpy()
        values = pd.DataFrame([{"model": label, "threshold": t, **net_benefit(y, p, t)} for t in thresholds])
        curve.append(values)
        ax.plot(values["threshold"], values["net_benefit_model"], label=label)
        if i == 0:
            ax.plot(values["threshold"], values["net_benefit_all"], "k--", lw=1, label="flag every nodule")
            ax.axhline(0, color="grey", lw=1, label="flag none")
    ax.set_xlabel("decision threshold t")
    ax.set_ylabel("net benefit (cross-validated)")
    ax.set_ylim(bottom=-0.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / "decision_curve.png", dpi=200)

    subgroup = pd.DataFrame(subgroup)
    curve = pd.concat(curve)
    subgroup.to_csv(cfg.TABLES / "subgroup_auc.csv", index=False)
    curve.to_csv(cfg.TABLES / "decision_curve.csv", index=False)
    print(subgroup.round(3).to_string(index=False))
    useful = curve[(curve["net_benefit_model"] > curve["net_benefit_all"]) & (curve["net_benefit_model"] > 0)]
    for label, part in useful.groupby("model"):
        print(f"{label}: net benefit above both default strategies for t in "
              f"[{part['threshold'].min():.2f}, {part['threshold'].max():.2f}] (of 0.05 to 0.60 examined)")
    print(f"wrote {cfg.TABLES / 'subgroup_auc.csv'}, decision_curve.csv and figures/decision_curve.png")


if __name__ == "__main__":
    main()
