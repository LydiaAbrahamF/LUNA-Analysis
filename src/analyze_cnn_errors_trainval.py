"""Descriptive subgroup/error analysis of CNN and diameter trainval OOF scores.

This script reads only the shared trainval OOF predictions and trainval metadata.
It never opens test predictions, test features, or test images.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/processed/cnn_oof_predictions_trainval.csv"
NODULES = ROOT / "data/processed/nodules.csv"
PATCH_INDEX = ROOT / "data/processed/patches_v1_trainval_index.csv"
OUT = ROOT / "results/cnn/cnn_trainval_subgroup_metrics.csv"
REPORT = ROOT / "results/cnn/cnn_trainval_error_analysis.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def auc_ap(labels: list[int], scores: list[float]) -> tuple[float, float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan, math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    auc = (sum(r for r, y in zip(ranks, labels) if y) - positives * (positives + 1) / 2) / (positives * negatives)
    ranked = sorted(zip(scores, labels), reverse=True)
    tp = 0
    seen = 0
    ap = 0.0
    for _, y in ranked:
        seen += 1
        tp += y
        if y:
            ap += tp / seen
    return auc, ap / positives


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main() -> None:
    preds = read_csv(PREDICTIONS)
    metadata = read_csv(NODULES)
    patch_index = read_csv(PATCH_INDEX)
    meta = {r["voi_name"]: r for r in metadata if r["split"] == "trainval"}
    if len(preds) != 828 or any(p["voi_name"] not in meta for p in preds):
        raise ValueError("Expected exactly 828 trainval OOF predictions matched to trainval metadata.")
    if {int(p["fold"]) for p in preds} != set(range(5)):
        raise ValueError("Expected shared trainval folds 0-4.")
    if any(int(p["y_true"]) != int(meta[p["voi_name"]]["malignant"]) for p in preds):
        raise ValueError("OOF labels do not match the shared trainval metadata.")
    pad = {r["voi_name"]: float(r["pad_fraction"]) for r in patch_index if r["split"] == "trainval"}
    if set(pad) != {p["voi_name"] for p in preds}:
        raise ValueError("Padding index does not match the 828 trainval OOF predictions.")
    positive_pad = [v for v in pad.values() if v > 0]
    q1, q2 = quantile(positive_pad, 1 / 3), quantile(positive_pad, 2 / 3)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for p in preds:
        m = meta[p["voi_name"]]
        primary = int(m["malignant"])
        alt_majority = int(m["label_majority"])
        alt_mean = int(m["label_mean"])
        fields = {
            "all_trainval": "all",
            "size_bin": m["size_bin"],
            "thickness": m["thickness"],
            "label_consensus": "ambiguous" if (primary != alt_majority or primary != alt_mean) else "all_rules_agree",
            "padding": ("zero" if pad[p["voi_name"]] == 0 else
                       "positive_low" if pad[p["voi_name"]] <= q1 else
                       "positive_mid" if pad[p["voi_name"]] <= q2 else "positive_high"),
        }
        for name, value in fields.items():
            groups[(name, value)].append(p)

    records = []
    report_lines = [
        "# CNN trainval error analysis (descriptive)", "",
        "This analysis uses only the 828 shared trainval out-of-fold predictions and trainval metadata. It does not read the held-out test set. The 0.5 threshold is a descriptive fixed threshold, not tuned.", "",
        "ROC-AUC and average precision are reported for subgroup ranking. Sensitivity, specificity and accuracy use score >= 0.5. Small subgroup counts make estimates uncertain; no subgroup result is used to select a model.", "",
        "| Group | Category | n | malignant | CNN AUC | CNN AP | CNN sens. | CNN spec. | Diameter AUC | Diameter AP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (group_name, category), rows in sorted(groups.items()):
        labels = [int(r["y_true"]) for r in rows]
        for model, col in (("CNN_v1_32mm", "cnn_score"), ("Diameter_logistic", "diameter_score")):
            scores = [float(r[col]) for r in rows]
            auc, ap = auc_ap(labels, scores)
            pred = [int(s >= 0.5) for s in scores]
            tp = sum(y == 1 and q == 1 for y, q in zip(labels, pred))
            tn = sum(y == 0 and q == 0 for y, q in zip(labels, pred))
            pos, neg = sum(labels), len(labels) - sum(labels)
            records.append({"group": group_name, "category": category, "model": model, "n": len(rows),
                            "n_malignant": pos, "n_benign": neg, "roc_auc": auc, "average_precision": ap,
                            "sensitivity_at_0.5": tp / pos if pos else math.nan,
                            "specificity_at_0.5": tn / neg if neg else math.nan,
                            "accuracy_at_0.5": sum(y == q for y, q in zip(labels, pred)) / len(rows)})
        c = next(r for r in records if r["group"] == group_name and r["category"] == category and r["model"] == "CNN_v1_32mm")
        d = next(r for r in records if r["group"] == group_name and r["category"] == category and r["model"] == "Diameter_logistic")
        fmt = lambda v: "NA" if not isinstance(v, (int, float)) or math.isnan(v) else f"{v:.3f}"
        report_lines.append(f"| {group_name} | {category} | {len(rows)} | {sum(labels)} | {fmt(c['roc_auc'])} | {fmt(c['average_precision'])} | {fmt(c['sensitivity_at_0.5'])} | {fmt(c['specificity_at_0.5'])} | {fmt(d['roc_auc'])} | {fmt(d['average_precision'])} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    report_lines += [
        "", "## Interpretation limits", "",
        f"Padding groups are zero padding, then tertiles among positive padding (cut points {q1:.3f} and {q2:.3f}; determined from trainval only). These are descriptive subgroup results from a single fixed trainval OOF run. The fold models can have different score calibration; subgroup AUC/AP rank within the pooled OOF scores and should be interpreted cautiously. Subgroups were not used for tuning. The ambiguity flag marks a nodule when either the majority-vote or mean-rating alternative label differs from the dataset label. This analysis does not estimate clinical utility and does not replace the locked test evaluation.", "",
        "The CNN and diameter scores are compared on the same nodules. The test set remains untouched.", "",
        "Reproduce with `python -m src.analyze_cnn_errors_trainval`. Aggregate CSV: `results/cnn/cnn_trainval_subgroup_metrics.csv`.", "",
    ]
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
