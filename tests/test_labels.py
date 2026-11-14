import os

import pandas as pd
import pytest

from lunarad import config as cfg
from lunarad.labels import (add_label_columns, is_malignant, is_tie, label_dataset_rule,
                            label_majority, label_mean_gt3, n_high, parse_ratings, rating_mean)


@pytest.mark.parametrize("ratings, expected", [
    ([5, 5, 5, 4], 1),
    ([2, 1, 1, 1], 0),
    ([4, 4, 2, 2], 1),       # exactly two raters above 3: the minimum, malignant
    ([4, 3, 3, 3], 0),       # only one rater above 3
    ([3, 3, 3], 0),          # mean 3.0 but nobody above 3
    ([5, 5, 1, 1, 1, 1, 1], 1),   # 7 raters, still only two needed
    ([4, 5, 5, 4], 1),
])
def test_dataset_rule(ratings, expected):
    assert is_malignant(ratings, "dataset") == expected
    assert label_dataset_rule(ratings) == expected


def test_default_rule_is_the_dataset_rule():
    assert cfg.LABEL_RULE == "dataset" or os.environ.get("LUNA_LABEL_RULE")
    assert is_malignant([4, 4, 2, 2], cfg.LABEL_RULE) == is_malignant([4, 4, 2, 2])


def test_boundary_nodules_per_rule():
    assert is_tie([4, 4, 2, 2], "dataset") == 1       # exactly two high
    assert is_tie([4, 4, 4, 2], "dataset") == 0
    assert is_tie([3, 3, 3], "mean_ge3") == 1 and is_tie([2, 4], "mean_ge3") == 1
    assert is_tie([3, 3, 4], "mean_ge3") == 0
    assert is_tie([4, 4, 2, 2], "majority") == 1


def test_sensitivity_labels():
    assert is_malignant([3, 3, 3], "mean_ge3") == 1
    assert label_mean_gt3([3, 3, 3]) == 0            # ties are benign here
    assert label_mean_gt3([4, 3, 3]) == 1
    assert label_dataset_rule([4, 4, 1, 1, 1, 1, 1]) == 1   # 7 raters, only 2 high
    assert is_malignant([4, 4, 1, 1, 1, 1, 1], "mean_ge3") == 0   # but the mean is 1.86
    assert label_majority([4, 4, 2, 2]) == 0                 # 2 of 4 is not a majority


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        is_malignant([3, 3, 3], "nonsense")


def test_helpers():
    assert n_high([5, 5, 3, 4]) == 3
    assert rating_mean([5, 5, 3, 4]) == 4.25
    assert parse_ratings("[5, 5, 5, 4]") == [5, 5, 5, 4]
    with pytest.raises(ValueError):
        is_malignant([])


def test_add_label_columns_accepts_strings_from_csv():
    df = pd.DataFrame({"ratings": ["[3, 3, 3]", "[1, 1, 2]", "[5, 4, 4, 5]"]})
    out = add_label_columns(df)
    assert out["label_dataset_rule"].tolist() == [0, 0, 1]
    assert out["label_mean_ge3"].tolist() == [1, 0, 1]
    assert out["is_tie"].tolist() == [int(is_tie(r)) for r in ([3, 3, 3], [1, 1, 2], [5, 4, 4, 5])]
    assert out["n_raters"].tolist() == [3, 3, 4]


@pytest.mark.skipif(os.environ.get("LUNA_DATA") is not None or not cfg.LABEL_AUDIT_CSV.exists(),
                    reason="needs the real label_audit.csv")
def test_known_answer_on_the_real_dataset():
    audit = add_label_columns(pd.read_csv(cfg.LABEL_AUDIT_CSV).drop(columns=["malignant"]))
    nodules = pd.read_csv(cfg.NODULES_CSV)
    assert list(nodules["voi_name"]) == list(audit["voi_name"])
    assert (nodules["malignant"].to_numpy() == audit["malignant"].to_numpy()).all()
    if len(audit) == 994 and cfg.LABEL_RULE == "dataset":
        assert int(audit["malignant"].sum()) == 344
