import numpy as np
import pandas as pd
import pytest

from lunarad.splits import build_split, file_hash, summarize_split, validate_split


@pytest.fixture
def nodules():
    rng = np.random.default_rng(1)
    patients = rng.integers(0, 150, 300)
    return pd.DataFrame({
        "voi_name": [f"n{i}.nii.gz" for i in range(300)],
        "patient_id": [f"P{p:03d}" for p in patients],
        "malignant": (rng.random(300) < 0.5).astype(int),
        "thickness": rng.choice(["thin", "mid", "thick"], 300),
    })


def test_split_is_valid_and_patient_disjoint(nodules):
    splits = build_split(nodules)
    validate_split(nodules, splits)
    test_patients = set(splits.loc[splits["split"] == "test", "patient_id"])
    train_patients = set(splits.loc[splits["split"] == "trainval", "patient_id"])
    assert test_patients.isdisjoint(train_patients)
    assert set(splits.loc[splits["split"] == "trainval", "fold"]) == {0, 1, 2, 3, 4}
    assert (splits.loc[splits["split"] == "test", "fold"] == -1).all()
    assert (splits["voi_name"] == nodules["voi_name"]).all()


def test_split_is_deterministic(nodules):
    assert build_split(nodules).equals(build_split(nodules))
    assert not build_split(nodules).equals(build_split(nodules, seed=7))


def test_validate_catches_a_leak(nodules):
    splits = build_split(nodules)
    test_patient = splits.loc[splits["split"] == "test", "patient_id"].iloc[0]
    row = splits.index[(splits["patient_id"] == test_patient)][0]
    leaked = splits.copy()
    leaked.loc[row, ["split", "fold"]] = ["trainval", 0]
    with pytest.raises(ValueError):
        validate_split(nodules, leaked)


def test_summary_and_hash(nodules, tmp_path):
    splits = build_split(nodules)
    assert summarize_split(nodules, splits)["nodules"].sum() == len(nodules)
    path = tmp_path / "splits.csv"
    splits.to_csv(path, index=False)
    assert len(file_hash(path)) == 64


def test_refold_keeps_patients_together_and_is_seeded():
    import numpy as np
    import pandas as pd
    from lunarad.splits import refold
    rng = np.random.default_rng(0)
    n = 300
    tv = pd.DataFrame({"patient_id": [f"P{i // 2}" for i in range(n)],
                       "malignant": rng.integers(0, 2, n),
                       "thickness": rng.choice(["thin", "thick"], n)})
    a, b = refold(tv, 1), refold(tv, 1)
    assert (a == b).all()                                  # same seed, same folds
    assert not (a == refold(tv, 2)).all()                  # different seed, different folds
    assert set(a) == set(range(5))
    assert (tv.assign(f=a).groupby("patient_id")["f"].nunique() == 1).all()
