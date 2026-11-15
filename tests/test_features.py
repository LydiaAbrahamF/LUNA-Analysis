import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("radiomics")

from lunarad.config import N_FEATURES, PROTOCOLS  # noqa: E402
from lunarad.features import extract_one, make_extractor  # noqa: E402


def _write_nodule(folder, spacing=(0.7, 0.7, 2.5), shift=0.0):
    rng = np.random.default_rng(0)
    volume = rng.normal(-600, 50, (12, 40, 40)).astype("float32")
    mask = np.zeros(volume.shape, dtype="uint8")
    mask[3:9, 12:28, 12:28] = 1
    image, mask_image = sitk.GetImageFromArray(volume), sitk.GetImageFromArray(mask)
    image.SetSpacing(spacing)
    mask_image.SetSpacing((spacing[0] + shift, spacing[1], spacing[2]))
    sitk.WriteImage(image, str(folder / "image.nii.gz"))
    sitk.WriteImage(mask_image, str(folder / "mask.nii.gz"))
    return folder / "image.nii.gz", folder / "mask.nii.gz"


@pytest.mark.parametrize("protocol", list(PROTOCOLS))
def test_extraction_returns_all_features(tmp_path, protocol):
    image, mask = _write_nodule(tmp_path)
    values = extract_one(image, mask, make_extractor(protocol))
    assert len([k for k in values if k.startswith("original_")]) == N_FEATURES
    assert values["spacing_z"] == pytest.approx(2.5)


def test_geometry_mismatch_is_rejected(tmp_path):
    image, mask = _write_nodule(tmp_path, shift=0.1)
    with pytest.raises(ValueError):
        extract_one(image, mask, make_extractor("original_spacing"))


def test_two_slice_crop_is_extracted_repeatably():
    """Regression test: with B-spline resampling a crop of only 2 slices gave different values on every run."""
    import numpy as np
    import SimpleITK as sitk
    from lunarad.features import make_extractor

    rng = np.random.default_rng(0)
    array = rng.normal(-600, 150, (2, 44, 44)).astype(np.int16)
    yy, xx = np.mgrid[:44, :44]
    blob = ((yy - 22) ** 2 + (xx - 22) ** 2) < 49
    array[:, blob] += 500
    mask_array = np.zeros_like(array, dtype=np.uint8)
    mask_array[:, blob] = 1
    image, mask = sitk.GetImageFromArray(array), sitk.GetImageFromArray(mask_array)
    for volume in (image, mask):
        volume.SetSpacing((0.7, 0.7, 2.5))
    extractor = make_extractor("isotropic_1mm")
    runs = [{k: float(v) for k, v in extractor.execute(image, mask, label=1).items() if k.startswith("original_")}
            for _ in range(5)]
    assert all(run == runs[0] for run in runs)
    assert -1100 < runs[0]["original_firstorder_Mean"] < 500      # a plausible HU value, not garbage
