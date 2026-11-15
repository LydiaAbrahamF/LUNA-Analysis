# LUNA16 Pulmonary Nodule Analysis

Benign vs. malignant classification of lung nodules (primary task) and nodule detection (secondary task) on the LUNA16 dataset. The repository documents each step, the reasoning behind it, and the results, so the work can be reproduced.

**Status (2026-11-15):** Metadata EDA, image geometry validation, the shared split, initial Radiomics baselines, and nested trainval-only selected-feature comparison are complete. Real-data CNN preprocessing QA, a fixed trainval CNN baseline, subgroup error analysis, a four-input trainval-only CNN ablation, and the three-sample mask-grid comparison are complete. The test set has already been evaluated for the pre-specified Radiomics baseline; it was not used in the CNN ablation. Related-work review and detection remain pending; detection needs the full CT dataset.

## Research questions

1. **RQ1:** How much do preprocessing choices (resampling, HU window, patch size) change performance?
2. **RQ2 (secondary):** Can candidate generation plus classification localise nodules at low false-positive rates?

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
- **Diameter is a strong baseline:** Annotated diameter alone achieved an AUC of 0.918 in an earlier grouped five-fold CV. AUCs within individual size groups ranged from 0.73 to 0.78. The current locked-split Radiomics baseline is reported below. An image model should be compared against these baselines, not against chance alone.
- **Masked-region HU summary:** In a descriptive trainval-only summary, malignant-labelled nodules had higher per-nodule median HU values than benign-labelled nodules, with overlapping distributions. This does not establish independent predictive value and was not used to tune a model.
- **Radiologist feature agreement:** Trainval-only category summaries show high agreement for internal structure and lower agreement for lobulation, margin, sphericity, and subtlety. These are annotation-pattern descriptions, not independent predictors or model results.
- **CNN preprocessing QA:** The fixed-size pipeline generated finite 64 × 64 trainval patches with no empty masks. Padding appears in 76.6% of patches and is strongly associated with native in-plane pixel spacing (Spearman ρ = −0.942), so it may encode acquisition resolution. This was a trainval-only QA finding, not a reason to tune on the held-out test set.

Trainval-only aggregate figures (generated from summary CSVs; no test-set data):

- [Masked-region HU distribution](results/figures/eda/hu_distribution_trainval.svg)
- [Radiologist-feature unanimity](results/figures/eda/annotator_agreement_trainval.svg)

| Baseline | AUC |
|---|---|
| Scanner settings only | 0.566 |
| Diameter only | 0.918 |
| Diameter + scanner settings | 0.922 |
| VOI depth only (shortcut) | 0.881 |

## Evaluation split

The shared split has 828 trainval nodules from 460 patients and 166 test nodules from 91 patients. Patients do not cross splits. The trainval set has five patient-level folds (0–4); test rows use fold -1. The test set was evaluated once in the pre-specified Radiomics baseline comparison and was not used for tuning.

## Preprocessing (v1)

| Step | Choice |
|---|---|
| Orientation | Reorient image and mask to RAS on load |
| Centre | Centroid of the nodule mask, in physical coordinates (the mask is never a model input) |
| In-plane window | 32 x 32 mm sampled to 64 x 64 pixels (0.5 mm per pixel) |
| Slice | One central axial slice, interpolated at an exact position in mm |
| Intensity | Clip to [-1000, 400] HU and scale to [0, 1] |
| Outside the VOI | Fill with air (-1000 HU) and record the fraction as `pad_fraction` |

The padding fraction is audited for leakage: the QA checks whether it predicts the label or correlates with nodule geometry and acquisition spacing. The trainval-only 32 mm vs. 40 mm geometry check and CNN performance comparisons for window size, one vs. three slices, and HU upper limit 400 vs. 600 are complete. See the [CNN input ablation results](results/cnn/cnn_input_ablation_report_trainval.md); these exploratory results do not establish a preferred input.

A three-sample check compared the archived manual-grid mask volumes with a fresh PyRadiomics ROI-aligned resampling run. All image-mask pairs passed geometry checks, but the voxel counts and volume changes differed between grids. This small check does not establish which grid is preferable for all nodules; see the [comparison report](results/resampling_comparison/mask_resampling_comparison_report.md) and [updated CSV](results/resampling_comparison/mask_resampling_comparison_updated.csv).

## Radiomics baseline (initial)

The confirmed target is `malignant` (at least two ratings above 3). Original-spacing and 1 mm isotropic PyRadiomics features were extracted for 828 trainval and 166 test nodules: 107 features per nodule and protocol, with no extraction failures or non-finite values. The test set was evaluated only after the baseline plan was fixed.

| Protocol | Input | Trainval five-fold OOF ROC-AUC | Test ROC-AUC (95% patient-bootstrap CI) |
|---|---|---:|---:|
| Original spacing | Diameter | 0.913 | 0.961 (0.927–0.988) |
| Original spacing | Radiomics | 0.927 | 0.929 (0.892–0.964) |
| Original spacing | Diameter + Radiomics | 0.927 | 0.930 (0.893–0.965) |
| 1 mm isotropic | Diameter | 0.913 | 0.961 (0.927–0.988) |
| 1 mm isotropic | Radiomics | 0.924 | 0.937 (0.892–0.975) |
| 1 mm isotropic | Diameter + Radiomics | 0.924 | 0.937 (0.892–0.975) |

This is an initial baseline on a 166-nodule test set. Test intervals overlap, so the point estimates do not establish a clear Radiomics gain or a preferred resampling protocol. No test results were used to select a model or protocol. Detailed results and patient-level predictions remain local.

## Trainval-only selected-feature comparison

Feature count was selected separately within each outer trainval fold using four-fold patient-grouped inner CV (candidate k = 5, 10, 20, or 50). Feature ranking and all preprocessing were fitted within training folds. The analysis did not read test features or test metrics. Inner CV selected k = 20 in every outer fold for both protocols.

| Protocol | Model input | Trainval outer OOF ROC-AUC | Trainval outer OOF average precision |
|---|---|---:|---:|
| Original spacing | All Radiomics | 0.9269 | 0.8733 |
| Original spacing | Selected Radiomics | 0.9305 | 0.8723 |
| Original spacing | Diameter + all Radiomics | 0.9269 | 0.8731 |
| Original spacing | Diameter + selected Radiomics | 0.9304 | 0.8716 |
| 1 mm isotropic | All Radiomics | 0.9235 | 0.8695 |
| 1 mm isotropic | Selected Radiomics | 0.9258 | 0.8592 |
| 1 mm isotropic | Diameter + all Radiomics | 0.9236 | 0.8692 |
| 1 mm isotropic | Diameter + selected Radiomics | 0.9261 | 0.8596 |

Selected features produced slightly higher pooled ROC-AUC but lower average precision than the corresponding all-feature models; the decrease in average precision was larger for 1 mm isotropic features. This exploratory trainval comparison does not establish a reliable improvement or a preferred protocol. Aggregate outputs are saved under `results/radiomics/selected_feature_*_trainval.csv`.

## Repository structure

```
src/preprocess.py            preprocessing pipeline and dataset builder
src/qa_preprocess_trainval.py trainval-only real-data patch QA
src/compare_preprocess_windows_trainval.py trainval-only 32 vs. 40 mm geometry comparison
src/train.py                 exploratory CNN baseline on shared trainval folds only
src/run_cnn_input_ablations_trainval.py trainval-only CNN input ablation
src/analyze_cnn_errors_trainval.py trainval-only subgroup error summary
src/compare_mask_resampling_grids.py three-sample manual/PyRadiomics grid comparison
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
python -m src.evaluate_selected_features_trainval  # trainval-only nested comparison
python -m src.analyze_hu_distribution_trainval     # trainval-only HU summary
python -m src.analyze_annotator_features_trainval  # trainval-only reader-feature summary
python -m src.plot_trainval_eda                    # plots from aggregate trainval-only CSVs
python -m src.compare_preprocess_windows_trainval --voi-root /path/to/LUNA_Dataset/VOIs
python -m src.train --device cpu                   # exploratory CNN baseline; trainval only
```

Outputs are written to `data/processed/`: the patch array, an index CSV, and the settings used (JSON).
The selected-feature analysis requires the local trainval feature table and shared metadata tables; it does not read test features or metrics. Its aggregate outputs are written to `results/radiomics/`.
The HU summary requires the local VOI images and masks plus the shared metadata tables. It reads trainval VOIs only; per-nodule outputs should remain local.
The annotator-feature summary requires the local `MetadatabyAnnotation.xlsx` and shared metadata tables. Pass `--metadata /path/to/MetadatabyAnnotation.xlsx` if it is not at the default Desktop path.

## CNN preprocessing QA (trainval only)

The real-data preprocessing was run on the 828 shared trainval VOIs only; no test images were processed for this QA. All patches had shape `(1, 64, 64)`, finite values in `[0, 1]`, non-empty masks, and matching source image/mask geometry. A visual spot-check found no obvious centring or orientation failure. However, 76.6% of patches had some air padding, and `pad_fraction` correlated strongly with native in-plane spacing (Spearman ρ = −0.942). See [`results/preprocessing_qa_trainval.md`](results/preprocessing_qa_trainval.md). This potential acquisition-resolution shortcut was investigated in a pre-specified trainval-only CNN input comparison. The test set had already been evaluated for the Radiomics baseline and was not used in the CNN comparison.

A geometry-only 32 mm versus 40 mm field-of-view check at 0.5 mm/pixel found that the wider VOI crop reduced mask-edge contact (3.6% to 1.6%) but increased patches with any padding (76.6% to 97.6%). This did not compare classifier performance and does not select a final window; a full-CT patch source would be needed to provide real surrounding tissue without VOI-edge air fill.

## Exploratory CNN baseline (trainval only)

A small CNN was evaluated with the five shared patient-level folds. Each outer fold used an inner patient-grouped validation split to select the training epoch, then refit on the full outer training portion. No test rows, images, predictions, or metrics were read. The comparison uses the same outer folds for a diameter-only logistic-regression baseline.

| Model | Mean fold ROC-AUC ± SD | Mean fold average precision ± SD | Mean fold sensitivity | Mean fold specificity |
|---|---:|---:|---:|---:|
| CNN, 32 mm central slice | 0.885 ± 0.018 | 0.820 ± 0.021 | 0.510 | 0.928 |
| Diameter logistic regression | 0.912 ± 0.019 | 0.847 ± 0.039 | 0.668 | 0.939 |

This single fixed CNN baseline underperformed the diameter baseline on the trainval fold averages. It is exploratory, not a final model comparison. Aggregate fold and summary tables are under `results/cnn/`; per-nodule predictions remain local under `data/processed/`.

### Initial subgroup error analysis (trainval only)

The fixed CNN and same-fold diameter OOF scores were summarized by nodule size, slice-thickness group, label-rule agreement, and padding fraction. No test rows were accessed. With the fixed descriptive 0.5 threshold, CNN AUC was 0.556 in small nodules (13 malignant cases), 0.700 in medium nodules, and 0.656 in large nodules. In 165 ambiguous-label nodules CNN AUC was 0.673, compared with 0.882 among 663 nodules where the dataset, majority, and mean-rating rules agreed. CNN AUC was 0.901 in zero-padding patches and 0.721 in the highest-padding tertile. This is a warning signal, not proof that the CNN uses padding, because padding is associated with source spacing and nodule geometry. Diameter remained stronger in these summaries. Subgroup results are descriptive and uncertain; fold-specific models may have different score calibration. See [`results/cnn/cnn_trainval_error_analysis.md`](results/cnn/cnn_trainval_error_analysis.md) and [`results/cnn/cnn_trainval_subgroup_metrics.csv`](results/cnn/cnn_trainval_subgroup_metrics.csv). Re-run with `python -m src.analyze_cnn_errors_trainval`.

### CNN input ablation (trainval only)

Four fixed inputs were compared under the same five patient-level trainval folds. Each used the same CNN architecture and fold-local inner validation for epoch selection; only input field of view, number of slices, or HU upper limit changed. No test images, labels, predictions, or metrics were used.

| Input | Mean fold ROC-AUC ± SD | Mean fold average precision ± SD |
|---|---:|---:|
| 32 mm, one slice, HU max 400 | 0.885 ± 0.018 | 0.820 ± 0.021 |
| 40 mm, one slice, HU max 400 | 0.885 ± 0.026 | 0.802 ± 0.035 |
| 32 mm, three slices, HU max 400 | 0.877 ± 0.018 | 0.801 ± 0.025 |
| 32 mm, one slice, HU max 600 | 0.889 ± 0.017 | 0.811 ± 0.027 |
| Diameter logistic reference | 0.912 ± 0.019 | 0.847 ± 0.039 |

No CNN input variant clearly improved over the 32 mm baseline in both metrics, and all remained below the diameter reference. This is a single-seed exploratory comparison, not evidence of a preferred input. See [`results/cnn/cnn_input_ablation_report_trainval.md`](results/cnn/cnn_input_ablation_report_trainval.md), [`results/cnn/cnn_input_ablation_summary_trainval.csv`](results/cnn/cnn_input_ablation_summary_trainval.csv), and [`results/cnn/cnn_input_ablation_by_fold_trainval.csv`](results/cnn/cnn_input_ablation_by_fold_trainval.csv). Re-run using `python -m src.run_cnn_input_ablations_trainval --voi-root /path/to/LUNA_Dataset/VOIs --device cpu`.

## Limitations

- Labels are radiologist opinion, not pathology, and larger nodules are rated more suspicious.
- The small-nodule group has only about 13 malignant cases, so results in that group are uncertain. Patient-cluster bootstrap confidence intervals are reported for test metrics.
- Single dataset, no external validation.
- Single-slice input discards 3D context. A fixed three-slice comparison was completed, but the exploratory result does not establish which input is preferable.
- The real-data preprocessing QA used trainval only and flagged a strong association between padding and native in-plane spacing. The 32 mm vs. 40 mm geometry comparison and CNN input performance ablation are complete. Their findings are exploratory and do not establish a preferred input.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Metadata EDA | Core EDA and trainval-only radiologist-feature summaries complete; remaining items listed in the progress report |
| 2 | Image loading and geometry validation | Done |
| 3 | Preprocessing | Real-data trainval QA and 32 mm vs. 40 mm geometry and CNN input performance comparisons complete |
| 4 | Evaluation protocol and split | Shared patient split verified; initial baseline evaluation done |
| 5 | Baselines (diameter, scanner, radiomics) | Initial baselines and nested trainval-only selected-feature comparison complete |
| 6 | CNN classifier | Exploratory baseline and four-input trainval-only ablation complete; further modeling is optional and should be pre-specified |
| 7 | Evaluation and error analysis | Trainval subgroup/error analyses complete; the test set was already used for the pre-specified Radiomics baseline and must not be used to tune CNN inputs |
| 8 | Detection (needs full CT scans) | Not started |

## Data and acknowledgements

LUNA16 is derived from the LIDC-IDRI collection. The dataset description used here is by Torres and Gil (UAB). Please follow the original dataset licences. The data is not redistributed here.

