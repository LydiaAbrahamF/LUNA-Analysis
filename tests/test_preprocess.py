import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    PreprocessConfig,
    build_dataset,
    hu_to_unit,
    preprocess_images,
    sample_patch,
)


def test_hu_to_unit_scales_and_clips_values():
    cfg = PreprocessConfig(hu_min=-1000, hu_max=400)

    result = hu_to_unit(np.array([-1200, -300, 500]), cfg)

    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_sample_patch_returns_one_channel_per_offset():
    vol = np.ones((5, 5, 4), dtype=np.float32)
    cfg = PreprocessConfig(window_mm=2, out_px=2, offsets_mm=(0.0, 1.0))

    patch, inside = sample_patch(vol, np.eye(4), np.array([2.0, 2.0, 1.0]), cfg, -1)

    assert patch.shape == (2, 2, 2)
    assert inside.shape == (2, 2)
    assert inside.all()
    np.testing.assert_allclose(patch, 1.0)


def test_preprocess_images_returns_normalized_fixed_size_patch():
    image = np.full((8, 8, 5), -500, dtype=np.float32)
    mask = np.zeros_like(image)
    mask[3:6, 3:6, 2] = 1
    cfg = PreprocessConfig(window_mm=4, out_px=8, offsets_mm=(0.0, 1.0))

    patch = preprocess_images(
        nib.Nifti1Image(image, np.eye(4)),
        nib.Nifti1Image(mask, np.eye(4)),
        cfg,
    )

    assert patch.x.shape == (2, 8, 8)
    assert patch.mask.shape == (8, 8)
    assert patch.mask.any()
    np.testing.assert_allclose(patch.x, (-500 - cfg.hu_min) / (cfg.hu_max - cfg.hu_min))


def test_preprocess_images_centers_on_mask_centroid():
    image = np.zeros((8, 8, 5), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[2:5, 3:6, 1:3] = 1

    patch = preprocess_images(
        nib.Nifti1Image(image, np.eye(4)),
        nib.Nifti1Image(mask, np.eye(4)),
        PreprocessConfig(window_mm=4, out_px=8),
    )

    np.testing.assert_allclose(patch.center_world, [3.0, 4.0, 1.5])


def test_preprocess_images_rejects_mismatched_shapes():
    image = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    mask = nib.Nifti1Image(np.zeros((4, 4, 3)), np.eye(4))

    with pytest.raises(ValueError, match="differ in shape"):
        preprocess_images(image, mask)


def test_preprocess_images_rejects_empty_mask():
    image = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    mask = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))

    with pytest.raises(ValueError, match="empty mask"):
        preprocess_images(image, mask)


def test_build_dataset_saves_arrays_index_and_config(tmp_path):
    proc = tmp_path / "data" / "processed"
    voi_dir = tmp_path / "data" / "VOIs" / "VOIs"
    (voi_dir / "image").mkdir(parents=True)
    (voi_dir / "nodule_mask").mkdir(parents=True)
    proc.mkdir(parents=True)
    voi_name = "nodule.nii.gz"
    table = pd.DataFrame(
        [{
            "voi_name": voi_name,
            "patient_id": "patient-1",
            "malignant": 0,
            "split": "trainval",
            "fold": 0,
        }]
    )
    table.to_csv(proc / "nodules.csv", index=False)
    image = np.full((5, 5, 3), -500, dtype=np.float32)
    mask = np.zeros_like(image)
    mask[2, 2, 1] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), voi_dir / "image" / voi_name)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), voi_dir / "nodule_mask" / voi_name)

    cfg = PreprocessConfig(name="test", window_mm=2, out_px=4)
    X, M, index = build_dataset(cfg, root=tmp_path)

    assert X.shape == (1, 1, 4, 4)
    assert M.shape == (1, 4, 4)
    assert M[0].any()
    assert index.loc[0, "voi_name"] == voi_name
    assert "pad_fraction" in index
    np.testing.assert_array_equal(np.load(proc / "patches_test_X.npy"), X)
    np.testing.assert_array_equal(np.load(proc / "patches_test_mask.npy"), M)
    assert pd.read_csv(proc / "patches_test_index.csv").equals(index)
    assert json.loads((proc / "patches_test_config.json").read_text())["name"] == "test"


def test_build_dataset_requires_split_and_fold_columns(tmp_path):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    pd.DataFrame({"voi_name": ["nodule.nii.gz"]}).to_csv(proc / "nodules.csv", index=False)

    with pytest.raises(SystemExit, match="missing.*patient_id.*malignant.*split.*fold"):
        build_dataset(root=tmp_path)
