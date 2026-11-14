# Pulmonary nodule malignancy on LUNA16 / LIDC-IDRI: data analysis, preprocessing and radiomics

A practice project in medical image data analysis. It takes 994 lung nodules from LUNA16 / LIDC-IDRI through the full
workflow: exploring and validating the data, preprocessing the images, extracting PyRadiomics features, and
classifying radiologist-assessed malignancy with machine learning. The pipeline is scripted, tested and
containerised so that anyone with the data can reproduce the numbers. It does not claim a new scientific finding.

> **What is predicted.** A nodule is labelled *malignant* if at least two radiologists rated it above 3 on the
> 1 to 5 malignancy scale (the rule given by the dataset authors), otherwise *benign*. This is radiologist opinion,
> **not** a biopsy result. This is practice code, not a medical device.

## Contents

1. [What is in the repository](#what-is-in-the-repository)
2. [Data](#data)
3. [Quick start with Docker](#quick-start-with-docker-recommended)
4. [Local install](#local-install-without-docker)
5. [Run the analysis](#run-the-analysis)
6. [Method](#method)
7. [Results](#results)
8. [Limitations](#limitations)
9. [Troubleshooting](#troubleshooting)

## What is in the repository

```
eda/                  exploratory data analysis and preprocessing QA (scripts in notebook format)
notebooks/            the same two analyses as Jupyter notebooks
src/lunarad/          the library
    config.py           every path and constant, in one place
    labels.py           label rules (the dataset rule is the default)
    splits.py           patient-level split
    preprocess.py       image preprocessing: one fixed-size patch per nodule
    features.py         PyRadiomics feature extraction
    models.py           models, tuning, cross-validation
    evaluation.py       metrics, bootstrap intervals, calibration, thresholds, net benefit
pipeline/             numbered scripts, run in order (see "Run the analysis")
tests/                automated tests on synthetic data
scripts/compare_results.py   compares two results folders number by number
report/paper/         LaTeX source of the report (IEEE journal format, two columns)
results/tables/       result tables written by the pipeline (CSV)
results/figures/      figures written by the pipeline
Dockerfile, requirements-docker.txt   the pinned environment
data/                 the dataset (NOT in git, see "Data")
```

## Data

The dataset is **not included**. It is derived from LUNA16 / LIDC-IDRI; follow the original licences.
<!-- TODO: add the exact download link or course source of the VOI + metadata package. -->

Place it like this (the folder names are the ones in the download):

```
data/
  MetadatabyAnnotation.xlsx            one row per radiologist annotation
  VOIs/VOIs/image/*.nii.gz             994 nodule image crops (volumes of interest)
  VOIs/VOIs/nodule_mask/*.nii.gz       nodule masks (996 files)
  CT/CT/image/*.nii.gz                 full scans (optional; used only by a quality check)
```

To keep the data elsewhere, set the environment variable `LUNA_DATA` (and `LUNA_RESULTS` for the outputs).
Files derived from patients are written to `data/processed/` and are never committed.

## Quick start with Docker (recommended)

Docker fixes the operating system, Python 3.11 and every library version. PyRadiomics 3.0.1 does not build on
Python 3.12 or newer; the image takes care of that.

Install Docker Desktop, then from the repository root:

**1. Build the image** (once, about 5 to 10 minutes):

```bash
docker build -t lunarad .
```

**2. Check the code** (runs the tests on synthetic data; no dataset needed):

```bash
docker run --rm lunarad
```

**3. Run the pipeline on the real data.**

PowerShell (Windows):

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" lunarad bash pipeline/run_all.sh
```

bash (Linux / macOS):

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" lunarad bash pipeline/run_all.sh
```

This runs every stage except the final test evaluation and takes roughly one hour on a laptop.

## Local install (without Docker)

You need **Python 3.11**.

```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel versioneer numpy==2.4.6
pip install --no-build-isolation pyradiomics==3.0.1          # PyRadiomics must be built against the same numpy
pip install -r requirements-docker.txt
pip install -e . --no-deps                                   # makes `lunarad` importable
python -m pytest -q                                          # should end with: passed
```

If you skip `pip install -e .`, set `PYTHONPATH=src` in every new terminal (PowerShell: `$env:PYTHONPATH = "src"`).

## Run the analysis

Run everything from the repository root.

### Part A: explore and validate the data (optional, independent of Part C)

`eda/01_eda.py` (also `notebooks/01_eda.ipynb`) audits the metadata and images: what one row means, how nodules are
identified, coordinate conventions, image orientation and spacing, intensity ranges, and which simple variables
already predict the label. It writes figures to `results/figures/eda/`. It writes its own tables as
`data/processed/eda_*.csv` and does not touch the files used by the pipeline.

### Part B: image preprocessing (optional, independent of Part C)

```bash
python -m lunarad.preprocess            # needs pipeline stages 1 and 2 first
```

Builds one fixed-size image patch per nodule (see [Method](#method)) and saves the patches to
`data/processed/patches_v1_*`. `eda/02_preprocessing_qa.py` then checks what a model would see. The patches are
prepared for an image-based model; the radiomics models below do not use them.

### Part C: radiomics pipeline

| # | Command | What it does | Check that it worked |
|---|---|---|---|
| 1 | `python pipeline/01_build_dataset.py --qa` | Builds the nodule table (one label column) | prints `label rule: dataset` and `994 \| malignant 344 \| benign 650` |
| 2 | `python pipeline/02_make_split.py` | Creates the patient-level split once | prints the split's SHA-256; a second run does nothing unless `--force` |
| 3 | `python pipeline/03_extract_features.py` | PyRadiomics, 107 features, 2 protocols (a few minutes) | `rows: 1988 (expected 1988) \| failures: 0` |
| 4 | `python pipeline/04_run_models.py` | Cross-validated reference models and feature sets | `cv_metrics.csv` |
| 5 | `python pipeline/05_robustness.py` | Feature stability when the mask is resampled | `feature_ranking_stability.csv` |
| 6 | `python pipeline/06_repeated_cv.py` | Repeats the cross-validation under 10 fold assignments | `cv_repeats_summary.csv` |
| 7 | `python pipeline/07_compare.py` | Paired AUC differences; AUC inside diameter groups | `paired_differences.csv`, `size_stratified_auc.csv` |
| 8 | `python -W ignore::FutureWarning pipeline/08_model_comparison.py` | Tuned model families and an ensemble (15 to 20 minutes) | `model_comparison.csv`, `oof_predictions.csv` |
| 9 | `python pipeline/09_threshold_calibration.py` | Calibration; decision thresholds for recall targets | `threshold_analysis.csv`, `figures/calibration.png` |
| 9b | `python pipeline/09b_subgroups_decision_curve.py` | AUC by slice-thickness group; decision curve | `subgroup_auc.csv`, `figures/decision_curve.png` |
| 9c | `python pipeline/09c_figures.py` | Forest plots of model AUCs and paired differences | `figures/model_auc_forest.png`, `figures/paired_difference_forest.png` |
| 10 | `python pipeline/10_final_test.py --confirm` | **The single test-set evaluation. Run once.** | `final_test_metrics.csv` |

`bash pipeline/run_all.sh` runs the tests and stages 1 to 9c. Stage 10 is deliberately left out.

**About stage 10.** The test set gives an honest estimate only if nothing is chosen after looking at it. So the model,
the feature set and the decision thresholds are fixed in `src/lunarad/config.py` from cross-validation results and
committed to git first. The script refuses to run without `--confirm`, refuses if the thresholds differ from what
stage 9 produced, refuses if the split file has changed (`--expect-sha256 <hash>`), and refuses a second run.

**Other label rules.** For a sensitivity analysis, set `LUNA_LABEL_RULE` to `mean_ge3`, `mean_gt3` or `majority`
before stage 1, and point `LUNA_DATA` and `LUNA_RESULTS` at separate folders so that analyses never mix.

**Checking a replication.** Stage 1 and stage 3 must match exactly (994 nodules, 344 malignant, 1988 feature rows,
0 failures, 107 features). Model results should match the tables below within about 0.005 AUC. To compare two full runs:

```bash
python scripts/compare_results.py results results_docker --tolerance 0.005
```

## Method

**Data understanding (EDA).** One row of the metadata is one radiologist's annotation, not one nodule, so annotations
are aggregated per nodule. Two different nodules share one identifier in patient LIDC-IDRI-0332 and are separated by
their coordinates. Metadata coordinates follow the DICOM convention, so x and y are negated before conversion to
voxels; this was verified by showing that a nodule crop equals the corresponding block of the full scan. Images come in
two orientations (972 LPS, 22 RAS), with in-plane pixel sizes from 0.46 to 0.98 mm and slice thicknesses from 0.45 to
2.5 mm.

**Shortcuts found and removed.** The depth of a nodule crop almost copies the nodule size (it alone predicts the label
with AUC 0.88), so crop extent and bounding-box columns are never model inputs. Scanner settings are weakly associated
with the label, but the association disappears inside nodule-size groups. Radiologist ratings are kept in a separate
audit file so that the modelling table has exactly one label column.

**Label.** Malignant if at least two radiologists rated the nodule above 3: 344 malignant, 650 benign (34.6%).

**Image preprocessing.** Each nodule is reoriented to a common orientation, centred on the centroid of its mask, and
sampled in physical space as a 32 x 32 mm window at 0.5 mm per pixel (64 x 64 pixels) on the central axial slice.
Intensities are clipped to [-1000, 400] HU and scaled to [0, 1]. Area outside the crop is filled with air and its
fraction is recorded. The mask is used only for centring.

**Radiomic features.** PyRadiomics 3.0.1, 107 features per nodule (shape 14, first order 18, GLCM 24, GLRLM 16,
GLSZM 16, GLDM 14, NGTDM 5), bin width 25, under two protocols: original voxel spacing and resampling to 1 mm
isotropic voxels.

**Split.** The unit is the patient: 234 of 551 patients have more than one nodule, so nodules are never split across
sets. One sixth of the patients form a held-out test set (166 nodules, 92 patients, 52 malignant); the remaining 828
nodules form 5 cross-validation folds. Split SHA-256: `f9dbb69fdf2292712aa0501d6ce2dee7aac3ee93be437c205a2b72cef0630745`.

**Models.** Logistic regression (untuned, L2, elastic net), random forest, gradient boosting, an RBF support vector
machine, and an equal-weight average of three of them. Hyper-parameters are tuned inside each training fold on
patient-grouped inner folds. Nodule diameter alone and scanner settings alone serve as simple reference models.

**Uncertainty.** 95% intervals come from resampling patients (2000 bootstrap replicates). Two models are compared
through the interval of their *paired* AUC difference: if it includes zero, the data cannot distinguish them.

**Choices fixed before the test evaluation** (from cross-validation only):

| Choice | Value | Reason |
|---|---|---|
| Feature protocol | original spacing | no measurable difference from 1 mm isotropic (+0.007, interval -0.002 to +0.017); fewer processing steps |
| Primary model | L2 logistic regression on the 107 radiomic features | the simplest tuned model; no other family is measurably better |
| Secondary model | equal-weight ensemble on the same features | reported alongside as a check |
| Decision rule | flag a nodule when the cross-validated sensitivity target of 95% is met | missing a malignant nodule is treated as worse than a false alarm |
| Thresholds | 0.154 (primary), 0.129 (secondary) | the cutoffs that reach 95% sensitivity in cross-validation |

## Results

Cross-validation on 828 nodules (459 patients, 292 malignant), original spacing.

| Topic | Result |
|---|---|
| Primary model, AUC | 0.933 (95% interval 0.914 to 0.949) |
| Secondary model, AUC | 0.937 (0.918 to 0.954) |
| All seven model families | AUC between 0.923 and 0.937; intervals overlap almost completely |
| Radiomics compared with diameter alone | +0.011 AUC (-0.004 to +0.025): not distinguishable |
| First-order features alone compared with diameter | -0.031 (-0.050 to -0.014): worse |
| Calibration of the primary model | slope 1.12, intercept 0.09, Brier score 0.094 |
| At the 95% sensitivity cutoff (primary) | sensitivity 0.952 (0.926 to 0.974), specificity 0.705 (0.664 to 0.746) |
| By slice thickness (primary) | AUC 0.909 thin, 0.941 at 1.25 mm, 0.948 thick; intervals overlap |
| Decision curve | net benefit above "flag all" and "flag none" for every threshold from 0.05 to 0.60 |

In short: on this dataset the radiomic features classify radiologist-assessed malignancy well, about as well as
nodule diameter alone, and the choice of model family does not matter.

**Held-out test set (166 nodules, scored once).** <!-- TODO: fill in from results/tables/final_test_metrics.csv and final_test_threshold_metrics.csv after stage 10 -->

| Model | AUC (95% interval) | Sensitivity at the fixed cutoff | Specificity at the fixed cutoff |
|---|---|---|---|
| Primary | *pending* | *pending* | *pending* |
| Secondary | *pending* | *pending* | *pending* |
| Diameter alone (reference) | *pending* | n/a | n/a |

## Limitations

* Labels are radiologist ratings, not histopathology. Nodules with exactly two ratings above 3 sit on the boundary of
  the rule; one different rating would change their label.
* One dataset and one split; no external validation. The test set has 52 malignant nodules, so its intervals are wide.
* About 35% of the nodules are malignant, far more than in screening, so predictive values do not transfer.
* The decision threshold is chosen on cross-validated predictions, so sensitivity on new data can fall below the target.
* The split was created under an earlier label definition (mean rating of at least 3) and kept when the label changed.
  It is still patient-level, but the malignant share differs slightly between the test set (31%) and the rest (35%).
* The test set was scored once before, under that earlier label and with a different model. No choice in the present
  analysis used that result. About 13% of the test set also overlaps with the test set of an earlier exploratory
  analysis of this dataset.
* The comparisons are descriptive and not corrected for multiple testing.
* No image-based (CNN) model was trained; the preprocessing for one is included, the model is not.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'lunarad'` | Run `pip install -e . --no-deps`, or set `PYTHONPATH=src` in this terminal. |
| PyRadiomics fails to build | Use Python 3.11; install numpy first, then `pip install --no-build-isolation pyradiomics==3.0.1`. |
| `features.csv` has 10 or 40 rows | A quick run with `--limit` overwrote it. Rerun stage 3 without `--limit`. |
| Stage 4 fails with "only one class" | The feature table is too small (see the row above). |
| Stage 2 says the split exists and does nothing | Intended: the split is created once. Use `--force` only if you mean to replace it. |
| Stage 10 says no threshold is frozen, or `threshold mismatch` | Run stage 9 and copy its two 95% cutoffs into `FINAL_THRESHOLDS` in `src/lunarad/config.py`. |
| Stage 10 says the test set has been used | `final_test_metrics.csv` already exists. The test set is meant to be scored once. |
| Many lines of `FutureWarning` from the SVM in stage 8 | Harmless. Run with `python -W ignore::FutureWarning`. |
| Docker cannot see the data on Windows | Run from the repository root and keep the quotes around `"${PWD}/data:/app/data"`. |

## Acknowledgements

LUNA16 is derived from the LIDC-IDRI collection (Armato et al., Medical Physics, 2011; Setio et al., Medical Image
Analysis, 2017). Features are computed with PyRadiomics (van Griethuysen et al., Cancer Research, 2017). The dataset
package used here was prepared by Torres and Gil (UAB).
<!-- TODO: contributors and their parts; licence for this code. -->
