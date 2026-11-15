# LUNA-Analysis

LUNA-Analysis is a practice project in medical image data analysis on lung nodules from LUNA16 / LIDC-IDRI. It has
three parts:

1. an exploratory analysis and an image-preprocessing module that validate the data and prepare nodule patches,
2. the `lunarad` package and a numbered pipeline that extract PyRadiomics features and classify
   radiologist-assessed malignancy with machine learning,
3. a written report in IEEE journal format.

The label is radiologist opinion, **not** a biopsy result: a nodule is *malignant* if at least two radiologists
rated it above 3 on the 1 to 5 scale. This is practice code, not a medical device, and it does not claim a new
scientific finding.

## Findings so far

- **Radiomic features classify radiologist-assessed malignancy well, and the estimate held on unseen patients.**
  - Primary model (L2 logistic regression on 107 features): cross-validated AUC 0.933 (95% interval 0.914 to
    0.949) on 828 nodules; test AUC 0.931 (0.884 to 0.967) on 166 nodules scored once.
- **Most of the signal is nodule size.**
  - Diameter alone reaches a cross-validated AUC of 0.924 (0.903 to 0.943). One shape feature alone reaches 0.93.
  - Radiomics minus diameter is +0.011 (-0.004 to +0.025) in cross-validation: not distinguishable from zero.
    On the test set it is +0.033 (0.004 to 0.068). The two disagree, so the advantage is not established.
- **The model family does not matter.**
  - Seven families (three logistic regressions, random forest, gradient boosting, RBF SVM, an ensemble) score
    between 0.923 and 0.937. Of 29 paired differences, 23 have an interval that includes zero.
- **The label definition changes the conclusions.**
  - Under a mean-rating label that was used first, diameter scored 0.83 and radiomics looked clearly better.
    Under the dataset's documented rule both effects disappear.
- **B-spline resampling is not repeatable on very thin nodule crops.**
  - 37 of the 994 crops have only 2 slices. With B-spline interpolation, two extractions of the same data
    disagreed on 1392 of 212,716 values and the extractor sometimes crashed. With linear interpolation two
    extractions agree on every value.
- **At a sensitivity-first operating point the cost is specificity.**
  - A threshold chosen for 95% sensitivity in cross-validation detected 48 of 52 malignant test nodules
    (0.923, interval 0.847 to 0.982) at a specificity of 0.658.

The full account, with tables and figures, is in the report (`report/paper/main.pdf`).

---

## The lunarad package

### Scope

Supported:

- Data: the LUNA16 / LIDC-IDRI nodule package with `MetadatabyAnnotation.xlsx`, VOI crops and nodule masks (NIfTI)
- Labels: the dataset rule (default), and three alternatives for sensitivity analyses
- Features: PyRadiomics 3.0.1, 107 `original_*` features, two resampling protocols
- Models: logistic regression (untuned, L2, elastic net), random forest, gradient boosting, RBF SVM, an
  equal-weight ensemble
- Evaluation: patient-grouped cross-validation, patient-level bootstrap intervals, paired differences,
  calibration, decision thresholds, decision curves, one held-out test evaluation

Not supported:

- Image-based (CNN) models: the preprocessing for one is included, the model is not
- Filtered or wavelet radiomic features, feature harmonisation across scanners
- Other datasets, external validation, pathology-confirmed labels

### Data

The dataset is **not included**. The source collection is
[LIDC-IDRI on The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/);
follow its licence and citation requirements.

This project does not read the raw collection directly. It uses a derived package of nodule crops (VOIs),
nodule masks and an annotation table (`MetadatabyAnnotation.xlsx`) prepared by Torres and Gil.
```
data/
  MetadatabyAnnotation.xlsx            one row per radiologist annotation
  VOIs/VOIs/image/*.nii.gz             994 nodule image crops (volumes of interest)
  VOIs/VOIs/nodule_mask/*.nii.gz       nodule masks (996 files)
  CT/CT/image/*.nii.gz                 full scans (optional; used only by a quality check)
```

Environment variables relocate folders: `LUNA_DATA` (raw data), `LUNA_PROCESSED` (derived tables, default
`data/processed/`), `LUNA_RESULTS` (outputs). Files derived from patients are never committed.

### Installation with Docker (recommended)

Docker fixes the operating system, Python 3.11 and every library version.

```bash
docker build -t lunarad .          # once, a few minutes
docker run --rm lunarad            # runs the tests on synthetic data; no dataset needed
```

Run the pipeline on the real data. The raw data is mounted read-only and everything the run writes goes to two
separate folders, so a container run cannot overwrite the files of a local run.

```powershell
# PowerShell (Windows)
mkdir processed_docker, results_docker -Force
Copy-Item data\processed\splits.csv processed_docker\     # optional: reuse an existing split
docker run --rm -e LUNA_PROCESSED=/app/processed -v "${PWD}/data:/app/data:ro" -v "${PWD}/processed_docker:/app/processed" -v "${PWD}/results_docker:/app/results" lunarad bash pipeline/run_all.sh
```

```bash
# bash (Linux / macOS)
mkdir -p processed_docker results_docker
cp data/processed/splits.csv processed_docker/              # optional: reuse an existing split
docker run --rm -e LUNA_PROCESSED=/app/processed -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/processed_docker:/app/processed" -v "$(pwd)/results_docker:/app/results" lunarad bash pipeline/run_all.sh
```

This runs every stage except the test evaluation and takes about one hour on a laptop.

### Local installation

Python 3.11 is required: PyRadiomics 3.0.1 does not build on newer versions.

```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip "setuptools<81" wheel versioneer numpy==1.26.4
pip install --no-build-isolation pyradiomics==3.0.1          # must be built against the same numpy
pip install -r requirements-docker.txt
pip install -e . --no-deps                                   # makes `lunarad` importable
```

Without `pip install -e .`, set `PYTHONPATH=src` in every new terminal (PowerShell: `$env:PYTHONPATH = "src"`).

### Tests

```bash
python -m pytest -q
```

About 50 tests run on synthetic data: the label rules, the split rules, feature-class counts, a regression test
for 2-slice crops, image preprocessing, the models, the bootstrap and threshold functions, and the whole pipeline
end to end. One extra test checks the known label counts on the real dataset when it is present.

---

## Data understanding and preprocessing

This part validates the data before any modelling. It is independent of the radiomics pipeline.

### Exploratory analysis (`eda/01_eda.py`)

- **One row is one annotation, not one nodule.** Annotations are aggregated per nodule.
- **Nodule identity.** Two different nodules of patient LIDC-IDRI-0332 share one identifier and are separated
  by their coordinates.
- **Coordinates.** Metadata coordinates follow the DICOM convention, so x and y are negated before conversion
  to voxels. Verified by showing that a nodule crop equals the corresponding block of the full scan.
- **Geometry.** Two orientations (972 LPS, 22 RAS); in-plane pixel size 0.46 to 0.98 mm; slice thickness 0.45 to
  2.5 mm. 171 of the 994 crops have fewer than 4 slices and 37 have only 2.
- **Shortcuts.** The depth of a nodule crop almost copies nodule size (it alone predicts the label with AUC
  0.88), so crop extent and bounding-box columns are never model inputs.

The same analysis is available as `notebooks/01_eda.ipynb`. It writes figures to `results/figures/eda/` and its
own tables as `data/processed/eda_*.csv`; it does not touch the files used by the pipeline.

### Image preprocessing (`src/lunarad/preprocess.py`)

```bash
python -m lunarad.preprocess            # needs pipeline stages 1 and 2 first
```

- **Patch.** One 64 x 64 pixel patch per nodule: a 32 x 32 mm window at 0.5 mm per pixel on the central axial
  slice, sampled in physical space after reorientation to a common orientation.
- **Centring.** On the centroid of the nodule mask. The mask is used for nothing else.
- **Intensities.** Clipped to [-1000, 400] HU and scaled to [0, 1]. Area outside the crop is filled with air and
  its fraction is recorded.
- **Quality check.** `eda/02_preprocessing_qa.py` shows what a model would see.

The patches are prepared for an image-based model. The radiomics models below do not use them.

---

## Radiomics study

The study extracts handcrafted features from each nodule and asks how well they classify the label, compared with
two simple references: nodule diameter alone and scanner settings alone.

| Input set | Contents | Features |
|---|---|---|
| `diameter` | nodule diameter from the metadata (reference) | 1 |
| `scanner` | in-plane pixel size and slice thickness (reference) | 2 |
| `shape_only`, `firstorder_only`, `texture_only` | one feature family each | 14, 18, 75 |
| `radiomics` | all PyRadiomics features | 107 |

### Pipeline

- **Label** (`labels.py`): malignant if at least two radiologists rated the nodule above 3. 344 malignant, 650
  benign. Ratings are kept in a separate audit file so the modelling table has exactly one label column.
- **Split** (`splits.py`): by patient, because 234 of 551 patients have more than one nodule.
  - One sixth of the patients form the held-out test set: 166 nodules, 92 patients, 52 malignant.
  - The other 828 nodules (459 patients, 292 malignant) form 5 cross-validation folds.
  - SHA-256 of the split file: `f9dbb69fdf2292712aa0501d6ce2dee7aac3ee93be437c205a2b72cef0630745`.
- **Features** (`features.py`): PyRadiomics 3.0.1, bin width 25, 107 features (shape 14, first order 18, GLCM 24,
  GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5), under two protocols.
  - `original_spacing`: no interpolation. Used for all main results.
  - `isotropic_1mm`: resampled to 1 mm voxels with **linear** interpolation. A robustness check only.
  - Extraction runs in isolated worker processes with retries and can be resumed.
- **Uncertainty** (`evaluation.py`): 95% intervals from 2000 bootstrap replicates that resample patients. Two
  models are compared through the interval of their *paired* AUC difference; if it includes zero, the data
  cannot distinguish them.

### Models and tuning (`models.py`)

- **Families**: logistic regression (untuned `C = 1`, L2, elastic net), random forest (300 trees), histogram
  gradient boosting, RBF support vector machine.
- **Ensemble**: the equal-weight average of the probabilities of L2 logistic regression, random forest and
  gradient boosting. No fitted weights. Stacking was not used.
- **Tuning**: small grids, chosen inside each training fold by 3-fold patient-grouped cross-validation, so the
  validation fold never influences tuning.
- **Choices fixed before the test evaluation**, from cross-validation only:
  - protocol: original spacing
  - primary model: L2 logistic regression on `radiomics`; secondary model: the ensemble on `radiomics`
  - decision rule: the largest threshold that reaches 95% sensitivity in cross-validation, giving 0.154
    (primary) and 0.129 (secondary)

### Running

```bash
python pipeline/01_build_dataset.py --qa        # nodule table; prints 994 | malignant 344 | benign 650
python pipeline/02_make_split.py                # patient-level split, created once; prints its SHA-256
python pipeline/03_extract_features.py          # 107 features x 2 protocols; rows: 1988 | failures: 0
python pipeline/04_run_models.py                # cross-validated feature sets
python pipeline/05_robustness.py                # feature ranking stability, sensitivity to resampling
python pipeline/06_repeated_cv.py               # 10 fold assignments
python pipeline/07_compare.py                   # paired differences, AUC inside diameter groups
python -W ignore::FutureWarning pipeline/08_model_comparison.py    # model families and ensemble (15-20 min)
python pipeline/09_threshold_calibration.py     # calibration, decision thresholds
python pipeline/09b_subgroups_decision_curve.py # slice-thickness subgroups, decision curve
python pipeline/09c_figures.py                  # forest plots
python pipeline/10_final_test.py --confirm      # the single test-set evaluation
python pipeline/10b_test_figures.py             # ROC curve and confusion matrix (drawing only)
```

- `bash pipeline/run_all.sh` runs the tests and stages 1 to 9c. Stage 10 is deliberately left out.
- **Stage 10 is run once.** It refuses to run without `--confirm`, if the thresholds in `src/lunarad/config.py`
  differ from what stage 9 produced, if the split file has changed (`--expect-sha256 <hash>`), or if a test
  result already exists. Fix the choices in `config.py` and commit them first.
- **Other label rules**: set `LUNA_LABEL_RULE` to `mean_ge3`, `mean_gt3` or `majority` before stage 1, and point
  `LUNA_PROCESSED` and `LUNA_RESULTS` at separate folders.
- **Checking a replication**: `python scripts/compare_results.py results results_docker --tolerance 0.005`
  compares two results folders table by table.

### Current results

Cross-validation on 828 nodules, original spacing, 95% patient-bootstrap intervals.

| Input or model | AUC | 95% interval | Source |
|---|---|---|---|
| Diameter alone (reference) | 0.924 | 0.903 to 0.943 | `cv_repeats_summary.csv` |
| Scanner settings alone (reference) | 0.578 | 0.531 to 0.625 | `cv_repeats_summary.csv` |
| Shape features | 0.931 | 0.913 to 0.949 | `cv_repeats_summary.csv` |
| First-order features | 0.893 | 0.866 to 0.917 | `cv_repeats_summary.csv` |
| Texture features | 0.930 | 0.909 to 0.947 | `cv_repeats_summary.csv` |
| All radiomics, logistic regression | 0.935 | 0.917 to 0.952 | `cv_repeats_summary.csv` |
| **Primary: L2 logistic regression, radiomics** | **0.933** | 0.914 to 0.949 | `model_comparison.csv` |
| Secondary: ensemble, radiomics | 0.937 | 0.918 to 0.954 | `model_comparison.csv` |

Held-out test set: 166 nodules, 92 patients, 52 malignant, scored once (`final_test_metrics.csv`).

| Model | AUC | Sensitivity at the fixed threshold | Specificity |
|---|---|---|---|
| Primary | 0.931 (0.884 to 0.967) | 0.923 (0.847 to 0.982), 48 of 52 | 0.658 (0.561 to 0.742) |
| Secondary | 0.931 (0.885 to 0.969) | 0.904 (0.803 to 0.981), 47 of 52 | 0.693 (0.602 to 0.775) |
| Diameter alone (reference) | 0.898 (0.831 to 0.952) | n/a | n/a |

How to read these numbers:

- **The intervals are wide and they matter.** The test set has 52 malignant nodules. A sensitivity of 0.923 and
  the 0.95 target cannot be told apart at this sample size.
- **Radiomics and diameter are about level.** The paired difference is +0.011 (-0.004 to +0.025) in
  cross-validation and +0.033 (0.004 to 0.068) on the test set. Only the second excludes zero.
- **Scanner settings are weakly predictive, not at chance.** Thick-slice scans contain larger nodules.
- **Calibration is good.** Primary model: slope 1.12, intercept 0.09, Brier score 0.094 in cross-validation;
  slope 1.16 on the test set.
- **Performance is similar across slice-thickness groups**: 0.909 (thin), 0.941 (1.25 mm), 0.948 (thick), with
  overlapping intervals.
- **The comparisons are descriptive.** 29 paired differences were examined without a multiplicity correction.
- **Predictive values do not transfer.** About 35% of these nodules are malignant, far more than in screening.

### Replication check

The cross-validation pipeline (stages 1 to 9c) was run in the Docker container on Linux and compared with a
local run on Windows. All 17 result tables agreed: 16 to four decimals and one within 0.0025, against a tolerance
of 0.005 set beforehand. The test evaluation was not repeated in the container.

### Earlier label definition (superseded)

The first version of this study labelled a nodule malignant when the mean rating was at least 3 (550 malignant,
444 benign). It is kept only as an option for sensitivity analyses.

- Under that label diameter scored 0.83 and radiomics about 0.89, which looked like a clear gain for radiomics.
- It was replaced by the dataset's documented rule so that results are comparable with other work on the data.
- Consequences that remain visible:
  - the split was stratified under the earlier label and kept, so the malignant share differs slightly between
    the test set (31%) and the rest (35%);
  - the test set was scored once under the earlier label with a different model; no choice in the present
    analysis used that result;
  - about 13% of the test set overlaps with the test set of an earlier exploratory analysis of this dataset.

---

## Project structure

```
LUNA-Analysis/
├── src/lunarad/              # the package
│   ├── config.py             # every path and constant
│   ├── labels.py             # label rules
│   ├── splits.py             # patient-level split
│   ├── preprocess.py         # image patches
│   ├── features.py           # PyRadiomics extraction
│   ├── models.py             # models, tuning, cross-validation
│   └── evaluation.py         # metrics, bootstrap, calibration, thresholds, net benefit
├── pipeline/                 # numbered stages 01 ... 10b and run_all.sh
├── eda/                      # exploratory analysis and preprocessing QA (scripts)
├── notebooks/                # the same two analyses as notebooks
├── tests/                    # automated tests on synthetic data
├── scripts/                  # compare_results.py, compare_features.py
├── report/paper/             # LaTeX source and figures of the report
├── results/tables/           # result tables written by the pipeline (CSV)
├── results/figures/          # figures written by the pipeline
├── Dockerfile                # the pinned environment
├── requirements-docker.txt   # pinned library versions
└── data/                     # the dataset (not in git)
```

## Known issues

- Labels are radiologist ratings, not histopathology. 155 nodules have exactly two ratings above 3 and sit on the
  boundary of the rule.
- One dataset and one split; no external validation.
- The decision threshold is chosen on the cross-validated predictions it is evaluated on, so sensitivity on new
  data falls a little below the target.
- Three-dimensional texture features on crops with 2 or 3 slices carry limited meaning; the `isotropic_1mm`
  protocol cannot be relied on for those nodules.
- The compiled PyRadiomics code can end the Python process without an error message. Stage 3 isolates and
  retries the affected chunk; rerun the command to resume. Keep `INTERPOLATOR = "sitkLinear"` in `config.py`.
- `python pipeline/03_extract_features.py --limit N` is a quick test and writes nothing.
- Stage 8 prints many `FutureWarning` lines from the SVM. They are harmless; `-W ignore::FutureWarning` hides
  them.
- `ModuleNotFoundError: No module named 'lunarad'` means the package is not installed: run
  `pip install -e . --no-deps` or set `PYTHONPATH=src`.
- On Windows, keep `*.sh text eol=lf` in `.gitattributes`, or `run_all.sh` fails inside the container.

## Acknowledgments

Built on PyRadiomics, SimpleITK and scikit-learn. LUNA16 is derived from the LIDC-IDRI collection (Armato et al.,
2011; Setio et al., 2017). The dataset package used here was prepared by Torres and Gil (UAB).
<!-- TODO: contributors and their parts; licence for this code. -->