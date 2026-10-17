# LUNA16 Pulmonary Nodule Analysis

Benign vs. malignant classification of lung nodules (primary task) and nodule detection (secondary task) on the LUNA16 dataset. The repository documents each step, the reasoning behind it, and the results, so the work can be reproduced.

**Status (2026-10-14):** metadata EDA, image geometry validation, the patient-level split, and the preprocessing code are done. The real-data preprocessing run is pending, and no classifier has been trained yet.

## Research questions

1. **RQ1:** Does an image model add predictive value beyond nodule diameter?
2. **RQ2:** How much do preprocessing choices (resampling, HU window, patch size) change performance?
3. **RQ3 (secondary):** Can candidate generation plus classification localise nodules at low false-positive rates?

## Dataset

LUNA16 is a curated subset of LIDC-IDRI: low-dose chest CT scans with nodules annotated by several radiologists. Each annotation includes the nodule centre, diameter, and subjective features, including a 1 to 5 malignancy rating. A nodule is labelled malignant if at least two radiologists rated it above 3.

| Item | Count |
|---|---|
| Nodules with an image and mask (classification set) | 994 (650 benign, 344 malignant) |
| Volumes of interest (VOIs) and masks (NIfTI) | 994 / 996 |
| Full CT scans available locally | 3 |

Only 3 full CT scans are available, so the detection task is blocked until the full CT download is obtained.

The data is not included in this repository (`data/` is in `.gitignore`).

## Key findings from the EDA

- **Labels:** One metadata row is one radiologist annotation, not one nodule. Aggregating by patient and coordinates gives 997 nodules, and 994 have an image. 116 of the 344 malignant labels (34%) are exact 2-2 rater ties, so labels depend on the consensus rule for about 9 to 12% of nodules.
- **Coordinates:** Metadata coordinates follow the DICOM LPS convention, so x and y must be negated before applying the file's inverse affine. This was verified on 30 random VOIs, on all 22 RAS-stored files, and by exact VOI-to-CT crop equality.
- **Heterogeneous images:** Orientation is mixed (972 LPS, 22 RAS), in-plane pixel size is 0.46 to 0.98 mm, and slice thickness takes 10 values from 0.45 to 2.5 mm. Resampling to a common physical scale is required.
- **Shortcut risk:** VOI depth alone predicts malignancy with AUC 0.881, because it tracks the annotation bounding box and therefore nodule size. Zero-padded, variable-depth volumes would leak size, so the pipeline avoids them and never feeds the mask to the model.
- **Scanner confounding:** Thicker slices are associated with more malignant nodules (29%, 33%, 40%; p = 0.0053), but the association disappears within nodule-size groups. Scanner settings alone give AUC 0.566.
- **Diameter is a strong baseline:** Annotated diameter alone gives AUC 0.918 (grouped 5-fold CV), and 0.73 to 0.78 within size groups. An image model must beat this baseline, not chance.

| Baseline | AUC |
|---|---|
| Scanner settings only | 0.566 |
| Diameter only | 0.918 |
| Diameter + scanner settings | 0.922 |
| VOI depth only (shortcut) | 0.881 |

## Evaluation split

The split is locked before any modelling. `StratifiedGroupKFold` (seed 42) groups by patient and stratifies by label and slice-thickness group. One fold is held out as the test set (166 nodules, 91 patients), and the rest form 5 cross-validation folds. Assertions check that no patient appears in two splits. The test set is touched once, at the end. The split is saved in `data/processed/splits.csv`.

## Preprocessing (v1)

| Step | Choice |
|---|---|
| Orientation | Reorient image and mask to RAS on load |
| Centre | Centroid of the nodule mask, in physical coordinates (the mask is never a model input) |
| In-plane window | 32 x 32 mm sampled to 64 x 64 pixels (0.5 mm per pixel) |
| Slice | One central axial slice, interpolated at an exact position in mm |
| Intensity | Clip to [-1000, 400] HU and scale to [0, 1] |
| Outside the VOI | Fill with air (-1000 HU) and record the fraction as `pad_fraction` |

The padding fraction is audited for leakage: the QA notebook tests whether it predicts the label or correlates with diameter. Planned ablations for RQ2: window 32 vs. 40 mm, one vs. three slices, and upper HU limit 400 vs. 600.

## Repository structure

```
src/preprocess.py            preprocessing pipeline and dataset builder
tests/test_preprocess.py     unit tests (8 passing, on synthetic data)
notebooks/                   EDA and preprocessing QA notebooks
results/figures/             EDA and preprocessing figures
data/                        raw and processed data (not tracked)
```

## How to run

```bash
pip install -r requirements.txt
python -m pytest -q          # unit tests
python -m src.preprocess     # build patches from the real VOIs
```

Outputs are written to `data/processed/`: the patch array, an index CSV, and the settings used (JSON).

## Limitations

- Labels are radiologist opinion, not pathology, and larger nodules are rated more suspicious.
- The small-nodule group has only about 13 malignant cases, so results in that group are uncertain. Bootstrap confidence intervals will be reported.
- Single dataset, no external validation.
- Single-slice input discards 3D context. 3D and multi-slice variants are planned as ablations.
- Preprocessing has been tested on synthetic data only. The real-data run and QA are pending.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Metadata EDA | Mostly done |
| 2 | Image loading and geometry validation | Done |
| 3 | Preprocessing | Implemented, real-data QA pending |
| 4 | Evaluation protocol and split | Split locked, metric code to write |
| 5 | Baselines (diameter, scanner) | Done; radiomics baselines to do |
| 6 | CNN classifier | Not started |
| 7 | Evaluation and error analysis | Not started |
| 8 | Detection (needs full CT scans) | Not started |

## Data and acknowledgements

LUNA16 is derived from the LIDC-IDRI collection. The dataset description used here is by Torres and Gil (UAB). Please follow the original dataset licences. The data is not redistributed here.

