"""Metrics and patient-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import BOOTSTRAP_REPLICATES, SEED


def metric_values(y, probability):
    """(ROC-AUC, average precision)."""
    return float(roc_auc_score(y, probability)), float(average_precision_score(y, probability))


def patient_bootstrap(y, probability, patient_ids, replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    """95% percentile intervals from resampling PATIENTS (not nodules) with replacement.

    Nodules of one patient are correlated, so resampling nodules would give intervals
    that are too narrow.
    """
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    ids = np.asarray(patient_ids)
    unique = pd.unique(ids)
    by_patient = {patient: np.flatnonzero(ids == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_patient[patient] for patient in sampled])
        if np.unique(y[index]).size < 2:
            continue
        auc, ap = metric_values(y[index], probability[index])
        aucs.append(auc)
        aps.append(ap)
    if not aucs:
        raise ValueError("no valid bootstrap replicate (a class is missing)")
    return {
        "roc_auc_ci_low": float(np.percentile(aucs, 2.5)),
        "roc_auc_ci_high": float(np.percentile(aucs, 97.5)),
        "average_precision_ci_low": float(np.percentile(aps, 2.5)),
        "average_precision_ci_high": float(np.percentile(aps, 97.5)),
        "bootstrap_valid_replicates": len(aucs),
    }


def patient_bootstrap_repeats(y, probabilities, patient_ids, replicates=BOOTSTRAP_REPLICATES,
                              seed=SEED):
    """95% interval for the repeat-averaged ROC-AUC of repeated cross-validation.

    probabilities has shape (n_nodules, n_repeats): one out-of-fold prediction per repeat.
    Each bootstrap replicate resamples PATIENTS once, computes the AUC of every repeat on that
    same sample, and averages over repeats. The interval therefore reflects patient sampling
    noise of the repeat-averaged estimate. Seed-to-seed fold noise is reported separately
    (std/min/max over repeats).
    """
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(probabilities) != len(y):
        raise ValueError("probabilities must have shape (n_nodules, n_repeats)")
    ids = np.asarray(patient_ids)
    unique = pd.unique(ids)
    by_patient = {patient: np.flatnonzero(ids == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_patient[patient] for patient in sampled])
        if np.unique(y[index]).size < 2:
            continue
        means.append(np.mean([roc_auc_score(y[index], probabilities[index, j])
                              for j in range(probabilities.shape[1])]))
    if not means:
        raise ValueError("no valid bootstrap replicate (a class is missing)")
    return {"roc_auc_ci_low": float(np.percentile(means, 2.5)),
            "roc_auc_ci_high": float(np.percentile(means, 97.5)),
            "bootstrap_valid_replicates": len(means)}


def patient_bootstrap_difference(y, probs_a, probs_b, patient_ids,
                                 replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    """Paired 95% interval for AUC(A) - AUC(B), resampling PATIENTS.

    probs_a and probs_b have shape (n_nodules, n_repeats) (or 1-D) and score the SAME nodules.
    Both models are evaluated on the identical bootstrap sample in every replicate, so the
    interval is for the difference itself, which is what a comparison needs. Comparing two
    separate intervals by eye does not do that. 'p_not_better' is the share of replicates in
    which A was not better than B (a bootstrap one-sided p-value, not a significance test).
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(probs_a, dtype=float).reshape(len(y), -1)
    b = np.asarray(probs_b, dtype=float).reshape(len(y), -1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("probs_a and probs_b must score the same nodules")
    ids = np.asarray(patient_ids)
    unique = pd.unique(ids)
    by_patient = {patient: np.flatnonzero(ids == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_patient[patient] for patient in sampled])
        if np.unique(y[index]).size < 2:
            continue
        auc_a = np.mean([roc_auc_score(y[index], a[index, j]) for j in range(a.shape[1])])
        auc_b = np.mean([roc_auc_score(y[index], b[index, j]) for j in range(b.shape[1])])
        diffs.append(auc_a - auc_b)
    if not diffs:
        raise ValueError("no valid bootstrap replicate (a class is missing)")
    point = float(np.mean([roc_auc_score(y, a[:, j]) for j in range(a.shape[1])])
                  - np.mean([roc_auc_score(y, b[:, j]) for j in range(b.shape[1])]))
    return {"auc_difference": point,
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_not_better": float(np.mean(np.asarray(diffs) <= 0)),
            "bootstrap_valid_replicates": len(diffs)}


def _ratio(a, b):
    return a / b if b else float("nan")


def threshold_metrics(y, probability, threshold):
    """Confusion counts and rates when every nodule with probability >= threshold is flagged."""
    y = np.asarray(y, dtype=int)
    flag = np.asarray(probability, dtype=float) >= threshold
    tp, fn = int((flag & (y == 1)).sum()), int((~flag & (y == 1)).sum())
    fp, tn = int((flag & (y == 0)).sum()), int((~flag & (y == 0)).sum())
    return {"sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
            "ppv": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn),
            "flagged_share": (tp + fp) / len(y), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def threshold_for_recall(y, probability, target):
    """Largest threshold whose sensitivity on these data is at least target."""
    y = np.asarray(y, dtype=int)
    positives = np.sort(np.asarray(probability, dtype=float)[y == 1])[::-1]
    if positives.size == 0:
        raise ValueError("no malignant nodule in y")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    return float(positives[int(np.ceil(target * positives.size)) - 1])


def patient_bootstrap_stats(y, probability, patient_ids, function, replicates=BOOTSTRAP_REPLICATES,
                            seed=SEED):
    """DataFrame with one row per bootstrap replicate: function(y_sample, p_sample) -> dict.
    Replicates resample PATIENTS. Use it to get intervals for any statistic."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    ids = np.asarray(patient_ids)
    unique = pd.unique(ids)
    by_patient = {patient: np.flatnonzero(ids == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_patient[patient] for patient in sampled])
        if np.unique(y[index]).size < 2:
            continue
        rows.append(function(y[index], probability[index]))
    if not rows:
        raise ValueError("no valid bootstrap replicate (a class is missing)")
    return pd.DataFrame(rows)


def interval(values):
    """(2.5th, 97.5th) percentile, ignoring NaN."""
    values = np.asarray(values, dtype=float)
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def calibration_summary(y, probability, bins=10):
    """Reliability table (quantile bins), Brier score, calibration intercept and slope.

    Slope 1 and intercept 0 mean perfectly calibrated; slope < 1 means over-confident.
    """
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    table = (pd.DataFrame({"p": p, "y": y})
             .assign(bin=lambda d: pd.qcut(d["p"].rank(method="first"), bins, labels=False))
             .groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                 observed_rate=("y", "mean")).reset_index())
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e9, max_iter=1000).fit(logit, y)
    return table, {"brier": float(brier_score_loss(y, p)),
                   "calibration_intercept": float(fit.intercept_[0]),
                   "calibration_slope": float(fit.coef_[0][0])}
