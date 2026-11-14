"""Metrics and patient-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
