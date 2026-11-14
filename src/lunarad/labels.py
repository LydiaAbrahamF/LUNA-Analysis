"""Label definitions: pure functions of a nodule's list of radiologist ratings (1-5).

Primary label: malignant if the MEAN rating >= LABEL_THRESHOLD (ties count as malignant).
The other rules are kept only for sensitivity analyses.
Comparisons use integer sums (sum >= threshold * n), so a mean of exactly 3.0 is exact.
"""
from __future__ import annotations

import ast

import pandas as pd

from .config import LABEL_THRESHOLD


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


def is_malignant(ratings, threshold=LABEL_THRESHOLD):
    """Primary label: mean rating >= threshold."""
    ratings = _clean(ratings)
    return int(sum(ratings) >= threshold * len(ratings))


def is_tie(ratings, threshold=LABEL_THRESHOLD):
    """True when the mean rating is exactly the threshold (the indeterminate nodules)."""
    ratings = _clean(ratings)
    return int(sum(ratings) == threshold * len(ratings))


def label_mean_gt3(ratings):
    """Sensitivity label: ties are benign (strictly mean > 3)."""
    ratings = _clean(ratings)
    return int(sum(ratings) > 3 * len(ratings))


def label_dataset_rule(ratings):
    """Sensitivity label: the LUNA16 authors' rule, at least two raters above 3."""
    return int(n_high(ratings) >= 2)


def label_majority(ratings):
    """Sensitivity label: more than half of the raters above 3."""
    ratings = _clean(ratings)
    return int(n_high(ratings) > len(ratings) / 2)


def add_label_columns(df, ratings_col="ratings"):
    """Return a copy of df with the primary label and the sensitivity columns added."""
    out = df.copy()
    parsed = out[ratings_col].apply(parse_ratings)
    out["n_raters"] = parsed.apply(len)
    out["n_high"] = parsed.apply(n_high)
    out["rating_mean"] = parsed.apply(rating_mean)
    out["malignant"] = parsed.apply(is_malignant)
    out["is_tie"] = parsed.apply(is_tie)
    out["label_mean_gt3"] = parsed.apply(label_mean_gt3)
    out["label_dataset_rule"] = parsed.apply(label_dataset_rule)
    out["label_majority"] = parsed.apply(label_majority)
    return out
