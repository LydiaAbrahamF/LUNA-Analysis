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
