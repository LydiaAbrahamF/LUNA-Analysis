"""Label definitions: pure functions of a nodule's list of radiologist ratings (1-5).

Primary label (config.LABEL_RULE, default "dataset"): the LUNA16 authors' rule, malignant if at least
two raters rated the nodule above 3. The mean-based and majority rules are kept for sensitivity analyses.
Mean comparisons use integer sums (sum >= threshold * n), so a mean of exactly 3.0 is exact.
"Boundary" nodules (column is_tie) sit exactly on the decision boundary of the active rule, where one
rating could flip the label: n_high == 2 for the dataset rule, mean == 3.0 for the mean rules.
"""
from __future__ import annotations

import ast

import pandas as pd

from .config import LABEL_RULE, LABEL_THRESHOLD, N_HIGH_REQUIRED


def _clean(ratings):
    ratings = list(ratings)
    if not ratings:
        raise ValueError("a nodule needs at least one rating")
    return ratings


def parse_ratings(value):
    """Accept a list, or the string form a list takes after a CSV round trip."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return _clean(value)


def n_high(ratings):
    """Number of raters who rated the nodule above 3 (a 4 or a 5)."""
    return sum(r > 3 for r in _clean(ratings))


def rating_mean(ratings):
    ratings = _clean(ratings)
    return sum(ratings) / len(ratings)


def label_dataset_rule(ratings):
    """The LUNA16 authors' rule: at least two raters above 3."""
    return int(n_high(ratings) >= N_HIGH_REQUIRED)


def label_mean_ge3(ratings):
    """Sensitivity label: mean rating >= 3 (ties at exactly 3.0 are malignant)."""
    ratings = _clean(ratings)
    return int(sum(ratings) >= LABEL_THRESHOLD * len(ratings))


def label_mean_gt3(ratings):
    """Sensitivity label: strictly mean > 3 (ties are benign)."""
    ratings = _clean(ratings)
    return int(sum(ratings) > LABEL_THRESHOLD * len(ratings))


def label_majority(ratings):
    """Sensitivity label: more than half of the raters above 3."""
    ratings = _clean(ratings)
    return int(n_high(ratings) > len(ratings) / 2)


RULES = {"dataset": label_dataset_rule, "mean_ge3": label_mean_ge3,
         "mean_gt3": label_mean_gt3, "majority": label_majority}


def _rule(rule):
    rule = rule or LABEL_RULE
    if rule not in RULES:
        raise ValueError(f"unknown label rule {rule!r}; choose from {sorted(RULES)}")
    return rule


def is_malignant(ratings, rule=None):
    """Primary label under the active rule (config.LABEL_RULE unless a rule is given)."""
    return RULES[_rule(rule)](ratings)


def is_tie(ratings, rule=None):
    """True when the nodule sits exactly on the decision boundary of the rule."""
    ratings = _clean(ratings)
    rule = _rule(rule)
    if rule == "dataset":
        return int(n_high(ratings) == N_HIGH_REQUIRED)
    if rule == "majority":
        return int(2 * n_high(ratings) == len(ratings))
    return int(sum(ratings) == LABEL_THRESHOLD * len(ratings))


def add_label_columns(df, ratings_col="ratings"):
    """Return a copy of df with the primary label and the sensitivity columns added."""
    out = df.copy()
    parsed = out[ratings_col].apply(parse_ratings)
    out["n_raters"] = parsed.apply(len)
    out["n_high"] = parsed.apply(n_high)
    out["rating_mean"] = parsed.apply(rating_mean)
    out["malignant"] = parsed.apply(is_malignant)
    out["is_tie"] = parsed.apply(is_tie)
    out["label_dataset_rule"] = parsed.apply(label_dataset_rule)
    out["label_mean_ge3"] = parsed.apply(label_mean_ge3)
    out["label_mean_gt3"] = parsed.apply(label_mean_gt3)
    out["label_majority"] = parsed.apply(label_majority)
    return out
