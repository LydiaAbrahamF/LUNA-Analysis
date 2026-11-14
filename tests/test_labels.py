import os

import pandas as pd
import pytest

from lunarad import config as cfg
from lunarad.labels import (add_label_columns, is_malignant, is_tie, label_dataset_rule,
                            label_majority, label_mean_gt3, n_high, parse_ratings, rating_mean)


@pytest.mark.parametrize("ratings, expected", [
    ([5, 5, 5, 4], 1),
    ([2, 1, 1, 1], 0),
    ([3, 3, 3], 1),          # tie, counts as malignant
    ([2, 4], 1),             # tie with different raters
    ([4, 4, 2, 2], 1),       # tie
    ([3, 3, 2], 0),          # mean 2.67
    ([3, 3, 2, 2], 0),       # mean 2.5
])
def test_primary_label(ratings, expected):
    assert is_malignant(ratings) == expected


def test_ties_are_exactly_the_mean_equal_to_threshold():
    assert is_tie([3, 3, 3]) == 1 and is_tie([2, 4]) == 1
    assert is_tie([3, 3, 4]) == 0 and is_tie([2, 3]) == 0


def test_sensitivity_labels():
    assert label_mean_gt3([3, 3, 3]) == 0            # ties are benign here
    assert label_mean_gt3([4, 3, 3]) == 1
    assert label_dataset_rule([4, 4, 1, 1, 1, 1, 1]) == 1   # 7 raters, only 2 high
    assert is_malignant([4, 4, 1, 1, 1, 1, 1]) == 0         # but the mean is 1.86
    assert label_majority([4, 4, 2, 2]) == 0                 # 2 of 4 is not a majority


def test_helpers():
    assert n_high([5, 5, 3, 4]) == 3
    assert rating_mean([5, 5, 3, 4]) == 4.25
    assert parse_ratings("[5, 5, 5, 4]") == [5, 5, 5, 4]
    with pytest.raises(ValueError):
        is_malignant([])


def test_add_label_columns_accepts_strings_from_csv():
    df = pd.DataFrame({"ratings": ["[3, 3, 3]", "[1, 1, 2]", "[5, 4, 4, 5]"]})
    out = add_label_columns(df)
    assert out["malignant"].tolist() == [1, 0, 1]
    assert out["is_tie"].tolist() == [1, 0, 0]
    assert out["n_raters"].tolist() == [3, 3, 4]


@pytest.mark.skipif(os.environ.get("LUNA_DATA") is not None or not cfg.NODULES_CSV.exists(),
                    reason="needs the real nodules.csv")
def test_known_answer_on_the_real_dataset():
    nodules = add_label_columns(pd.read_csv(cfg.NODULES_CSV))
    if len(nodules) == 994:
        assert int(nodules["malignant"].sum()) == 550
        assert int(nodules["is_tie"].sum()) == 146
