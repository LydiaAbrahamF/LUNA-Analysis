# LUNA16 Pulmonary Nodule Analysis: EDA Progress Report

**Status date:** 2026-10-13
**Project:** Pulmonary nodule malignancy classification (primary task) and nodule detection (secondary task) on the LUNA16 dataset
**Progress:** Phase 0 (setup) partly done, Phase 1 (metadata EDA) mostly done, Phase 2 (image loading and geometry validation) done. Phase 3 (preprocessing) is next.

This document records what was done in the exploratory data analysis (EDA) so far, why each step was done, what it showed, and what decision followed. It is written so that each section can later be adapted into the "Dataset" and "Methods" sections of the final report.

---

## 1. Project context

### 1.1 Dataset

LUNA16 is a curated subset of the LIDC-IDRI collection of low-dose chest CT scans with pulmonary nodules annotated by expert radiologists. The dataset description (Torres and Gil, UAB) lists these key facts:

- Each nodule was annotated by several radiologists in a two-phase reading (independent reading, then review against the other radiologists' annotations).
- Each annotation includes the nodule centre (x, y, z in mm), diameter (mm), and subjective features: subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, and **malignancy** (rated 1 to 5).
- **Label rule used by the dataset authors:** a nodule is *malignant* if at least two radiologists rated its malignancy above 3, otherwise *benign*.
- CT scans, nodule volumes of interest (VOIs) and binary nodule masks are stored in NIfTI format (`.nii.gz`).

### 1.2 Tasks and research questions

| | Task | Needs |
|---|---|---|
| Primary | Classify nodules as benign or malignant | Labels plus nodule images (VOIs) |
| Secondary | Detect and localise nodules | Full CT scans plus candidate generation |

- **RQ1:** Does an image-based model add predictive value beyond simple measurements such as nodule diameter?
- **RQ2:** How much do preprocessing choices (resampling, HU windowing, patch size) change performance?
- **RQ3 (secondary):** Can a candidate-generation plus classification pipeline localise nodules at low false-positive rates?

RQ1 was sharpened by the EDA results in Section 3.10.

### 1.3 Definitions used in this document

| Term | Meaning |
|---|---|
| Voxel | One 3D "pixel" of a CT volume, with a brightness value |
| Spacing | Voxel size in mm (sx, sy in-plane, sz = slice thickness) |
| HU | Hounsfield unit, the CT brightness scale (air about -1000, water 0, bone several hundred) |
| VOI | Volume of interest: a small box cut out of the full CT around one nodule |
| Mask | Binary volume, same shape as the VOI, equal to 1 on the nodule |
| Annotation | One radiologist's record for one nodule (one row in the metadata table) |
| AUC | Area under the ROC curve: 0.5 is chance, 1.0 is perfect ranking |

---

## 2. Data inventory

| Item | Count / description | Location |
|---|---|---|
| Metadata table | `MetadatabyAnnotation.xlsx`, one row per annotation | `data/` |
| Full CT scans | **3** | `data/CT/CT/image` |
| VOI images | **994** | `data/VOIs/VOIs/image` |
| VOI masks | **996** | `data/VOIs/VOIs/nodule_mask` |
| Acquisition parameters | `AcqParams.xlsx` (not yet analysed) | `data/` |

**Important consequence:** only 3 full CT scans are available locally. The classification task can use the 994 VOIs. The detection task needs many full scans, so the full CT download must be obtained before the detection phase.

Metadata columns: `patient_id, nodule_id, seriesuid, coordX, coordY, coordZ, diameter_mm, bboxLow/High X/Y/Z, Malignancy, Malignancy_value, anotation_id` (spelled with one "n" in the file), and the pairs `X` / `X_value` for Calcification, InternalStructure, Lobulation, Margin, Sphericity, Spiculation, Subtlety, Texture, plus `len_mal_details`.

---

## 3. EDA log: what we did and why

Each step follows the same format: **Goal**, **Method**, **Result**, **Decision**.

### 3.1 Understanding the metadata layout

**Goal.** Find out what one row of the table represents, because the label has to be computed per nodule, not per annotation.

**Method.** Printed the columns and the first rows. Grouped by `(patient_id, nodule_id)` and counted rows per group and distinct `coordX` values per group.

**Result.**
- One row is **one radiologist's annotation**, not one nodule.
- 995 groups had a single consistent location. One group had two distinct `coordX` values.
- Rows per group: 4 rows (667 groups), 3 rows (323), and unusual counts of 5 (3 groups), 6, 7, and 9 (one each). Total 996 groups.

**Decision.** Aggregate annotations into one row per nodule before labelling or counting.

### 3.2 Building the malignancy label

**Goal.** Turn the several radiologist ratings of a nodule into one binary label.

**Method.** Implemented the dataset's rule as a function and tested it on the two worked examples in the dataset description:

```python
def is_malignant(ratings):                  # e.g. [5, 5, 5, 4]
    return int(sum(r > 3 for r in ratings) >= 2)

assert is_malignant([5, 5, 5, 4]) == 1      # LIDC-IDRI-0001, nodule 1 (from the description)
assert is_malignant([2, 1, 1, 1]) == 0      # LIDC-IDRI-0004, nodule 1 (from the description)
```

**Result.** Both unit tests pass. The first grouping gave 996 nodules (650 benign, 346 malignant), which was later corrected (Section 3.3).

**Decision.** Keep the dataset authors' rule as the primary label. Plan a sensitivity analysis with alternative rules (majority vote, mean rating above 3), because "at least two" has a different meaning for 3, 4, or 7 raters. The sensitivity cell is written but its output has not been reviewed yet (Section 7).

### 3.3 Anomalous nodules and the nodule table fix

**Goal.** Explain the six groups with more than 4 annotations and the one group with two locations.

**Method.** Printed all rows of the six groups.

**Result.**

| Patient / nodule | Annotations | Ratings | Finding |
|---|---|---|---|
| LIDC-IDRI-0204 / 1 | 6 | 4, 3, 5, 5, 4, 4 | One nodule, extra reads |
| LIDC-IDRI-0252 / 1 | 5 | 5, 3, 5, 4, 4 | One nodule, extra reads |
| **LIDC-IDRI-0332 / 1** | **9** | **(5, 5, 5, 5) and (5, 5, 3, 3, 3)** | **Two different nodules sharing one ID** |
| LIDC-IDRI-0608 / 6 | 5 | 5, 5, 2, 4, 4 | One nodule, extra reads |
| LIDC-IDRI-0865 / 1 | 5 | 5, 5, 5, 5, 5 | One nodule, extra reads |
| LIDC-IDRI-0942 / 1 | 7 | 5, 4, 5, 5, 5, 3, 3 | One nodule, extra reads |

- Five groups have distinct annotation IDs and different ratings, so they are not duplicate rows. The reason for having more than four reads cannot be determined from the table. All five are malignant under the rule, and they remain malignant if only the first four annotations are used.
- LIDC-IDRI-0332 contains two clusters about 33 mm apart (diameters 24.7 mm and 11.5 mm), so they are two distinct nodules. The second has exactly 2 of 5 ratings above 3, the minimum for "malignant".
- The first aggregation averaged the two diameters (about 18 mm), which was wrong for both.

**Decision.** Include the coordinates in the grouping key so that 0332 splits into two nodules, and use the first diameter of each group. Result: **997 nodules (650 benign, 347 malignant)**.

### 3.4 First look at images (3 full CT scans)

**Goal.** Confirm that the files open and look like chest CTs, and learn their basic properties.

**Method.** Loaded scans with nibabel, displayed axial, coronal and sagittal views with a fixed lung window (-1400 to 200 HU), and read the headers.

**Result.** Three scans of about 133 to 140 slices, in-plane spacing 0.66 to 0.82 mm, slice thickness 2.5 mm, orientation LPS. Voxels are strongly anisotropic (about 0.7 x 0.7 x 2.5 mm).

**Decision.** Use a fixed display window (not per-image percentiles) so that images are comparable. Plan resampling to common spacing. Replace hard-coded absolute Windows paths with repo-relative paths and add `data/` to `.gitignore`.

### 3.5 Coordinate system validation

**Goal.** The metadata gives nodule positions in millimetres, whereas images are voxel grids. A wrong conversion would silently cut the wrong region. Verify the conversion before extracting any patch.

**Method.**
1. Converted the nodule centre of LIDC-IDRI-0001 to voxel indices in the full CT, trying (x, y) as given and with both signs flipped. Both fell inside the 512 x 512 slice, so this test could not decide.
2. Used the nodule's VOI as ground truth: a correct conversion must land near the VOI centre. Tested all four sign combinations on VOI `LIDC-IDRI-0001_R_1` (shape 53 x 64 x 11, axes L, P, S).
3. Repeated the best option on 30 random VOIs, measuring the offset from the VOI centre in voxels.
4. Computed where the VOI corner lies in the full CT, cut the corresponding block from the CT, and compared it with the VOI array.

**Result.**

| Option | Voxel position in VOI (centre about 26, 31.5, 5) |
|---|---|
| x, y as given | (-135.9, -212.6, 4.7) outside |
| flip y only | (-135.9, 33.0, 4.7) outside |
| flip x only | (24.0, -212.6, 4.7) outside |
| **flip x and y** | **(24.0, 33.0, 4.7) inside** |

- 30-VOI check: offsets in voxels (dx, dy, dz) have mean (0.12, -0.26, 0.12), standard deviation (1.14, 1.49, 0.41), worst cases (-4.8, -7.3, -0.9).
- Corner of VOI `LIDC-IDRI-0001_R_1` in the CT: (292, 334, 85), whole numbers. Block comparison: **VOI == CT crop: True**. So a VOI is an exact, non-resampled crop of the CT.

**Interpretation.** The metadata coordinates follow the DICOM LPS convention, while nibabel's world frame is RAS, so x and y must be negated before applying the inverse affine.

**Decision.** Use `(-x, -y, z)` with the file's own inverse affine everywhere. Turn the crop-equality check into a unit test in the repo.

### 3.6 Matching nodules to VOI files and masks

**Goal.** Make sure every nodule used for classification has an image and a mask.

**Method.** Built the expected VOI file name `<patient_id>_R_<nodule_id>.nii.gz` for each nodule and checked it against the folder contents. Compared image and mask file lists.

**Result.**
- 994 VOI image files, 996 mask files.
- Two masks have no image: `LIDC-IDRI-1002_R_1` and `LIDC-IDRI-1002_R_2`.
- Both LIDC-IDRI-0332 nodules map to the same file name `LIDC-IDRI-0332_R_1`, so only one can own it. The owner is chosen by which nodule lies near the VOI centre.
- After removing nodules without images, the final table has **994 nodules: 650 benign and 344 malignant (34.6% malignant)**. The three excluded nodules (two from LIDC-IDRI-1002, one from LIDC-IDRI-0332) were all malignant (347 to 344).

**Decision.** Classification dataset = 994 nodules. Document the three exclusions in the report.

### 3.7 Image inventory of all files

**Goal.** Check consistency of orientation, spacing, intensity range and mask alignment across all files, not only a few.

**Method.** Read the header of every full CT, VOI image and mask. Loaded all VOI images to compute brightness statistics. Checked that each VOI has a mask with the same shape and affine, and listed the values in each mask.

**Result.**

*Orientation*

| Kind | LPS | RAS |
|---|---|---|
| Full CT (3) | 3 | 0 |
| VOI image (994) | 972 | 22 |
| VOI mask (996) | 974 | 22 |

*Spacing (VOI images)*

- In-plane pixel size: min 0.461 mm, 25% 0.645, median 0.703, 75% 0.742, max 0.977. Pixels are square in all 994 files.
- Slice thickness `sz` has 10 distinct values:

| sz (mm) | 0.45 | 0.5 | 0.625 | 0.7 | 0.75 | 1.0 | 1.25 | 1.8 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Files | 4 | 8 | 118 | 34 | 11 | 161 | 275 | 76 | 34 | 273 |

*Intensity (HU)*

- All 994 VOI images are `int16`.
- Minimum per VOI: median -1005, 25% -1024, lowest -1445, highest -789.
- Maximum per VOI: median 828.5, mean 834.5, lowest -506, highest 3071.
- Mean per VOI: median -655, range -939 to -82.
- The values are therefore real Hounsfield units. Many VOIs contain very bright structures (bone, calcification or saturated voxels at the top of the 12-bit range).

*Masks*

- All 994 images have a mask, with identical shape and affine.
- Mask values are exactly {0, 1} in every file.
- Nodule size in voxels: min 36, 25% 188, median 400, 75% 1257, max 33157 (heavily right-skewed; voxel size differs between scans, so voxel counts are not comparable across scans).

*Coordinate check on the 22 RAS files*

- Offsets from VOI centre (voxels): mean (-0.05, -0.83, -0.11), standard deviation (1.25, 1.61, 0.58). The conversion is valid for RAS files as well.
- After `nibabel.as_closest_canonical`, all 994 VOIs report orientation RAS.

**Decisions.**
1. Load every image and mask through one function that reorients to a common orientation.
2. Resample to a common spacing (mandatory: pixel size varies about 2-fold, slice thickness about 5-fold).
3. Clip HU to a fixed window before scaling. The window is a hyperparameter to test: the plan starts at [-1000, 400], but calcified nodules may exceed 400 HU.

### 3.8 VOI geometry

**Goal.** Choose a sensible patch size and understand how much padding is needed.

**Method.** Read the shape of every VOI and multiplied it by the spacing.

**Result.**

| | mean | median | min | max |
|---|---|---|---|---|
| nx (voxels) | 44.9 | 44 | 43 | 91 |
| ny (voxels) | 44.8 | 44 | 43 | 109 |
| nz (voxels) | 9.1 | 6 | 2 | 62 |
| mm_x | 31.4 | 30.2 | 20.3 | 72.8 |
| mm_y | 31.3 | 30.2 | 19.8 | 74.6 |
| mm_z | 11.5 | 8.1 | 2.5 | 61.3 |

VOIs are about 30 mm wide in-plane but thin in z (median 6 slices, about 8 mm). The first VOI examined (53 x 64 x 11) was not typical.

**Decision / open issue.** A fixed 3D cube (for example 32 mm) would require heavy z-padding for a typical VOI. Padding amount might correlate with nodule size and leak the answer. Planned check: correlate `mm_z` with diameter and compute AUC of `mm_z` alone (not yet run). Candidate designs: a 2.5D / thin-slab model, 3D with controlled padding and a size-cue ablation, or patches cut from full CTs once they are available.

### 3.9 Scanner-related confounding

**Goal.** Check whether acquisition settings predict the label, because a model could learn scanner characteristics instead of nodule appearance.

**Method.** Grouped nodules by slice thickness (thin up to 1.0 mm, 1.25 mm, thick 1.8 mm or more) and by tertiles of in-plane pixel size, computed the malignant share per group, applied a chi-square test, compared nodule diameters per group, and trained logistic regression on scanner settings alone (5-fold cross-validation, folds grouped by patient).

**Result.**

| Slice thickness | Nodules | Benign | Malignant | Malignant share |
|---|---|---|---|---|
| thin (<= 1.0 mm) | 336 | 238 | 98 | 29% |
| 1.25 mm | 275 | 184 | 91 | 33% |
| thick (>= 1.8 mm) | 383 | 228 | 155 | 40% |

- Chi-square p-value: **0.0053**.
- Pixel size tertiles: small 31% (382 nodules), medium 34% (286), large 39% (326).
- Diameter by thickness (mean, median): thin 7.7 mm, 6.2 mm; 1.25 mm: 8.0 mm, 6.0 mm; thick 10.0 mm, 8.1 mm. Thick-slice scans contain larger nodules and no very small ones.
- AUC from scanner settings alone: **0.566**.

Stratifying by nodule size (tertiles) removes the thickness effect:

| Size group | thin: n, malignant share | 1.25 mm | thick |
|---|---|---|---|
| small | 139, 0.05 | 117, 0.05 | 76, 0.01 |
| medium | 107, 0.21 | 84, 0.24 | 140, 0.15 |
| large | 90, 0.76 | 74, 0.88 | 167, 0.80 |

Within each size group the malignant share does not increase with thickness. The overall association arises because thick-slice scans contain proportionally more large nodules (44% of thick-slice nodules are in the large group, against 27% for thin slices).

**Decision.** Scanner confounding is mostly explained by nodule size. Still: stratify splits by label and thickness group, report performance per thickness group, report the scanner-only AUC (0.566) as a baseline, and consider blur augmentation. Hospital or protocol differences cannot be excluded as an additional factor.

### 3.10 Diameter as a baseline predictor

**Goal.** Establish how strong a trivial predictor is, so that the image model can be judged against it.

**Method.** Logistic regression with 5-fold cross-validation grouped by patient, using different feature sets. Also computed the AUC of raw diameter inside each size tertile.

**Result.**

| Features | AUC |
|---|---|
| Scanner settings only | 0.566 |
| **Diameter only** | **0.918** |
| Diameter + scanner settings | 0.922 |

Within size tertiles (AUC of diameter as a ranking score):

| Size group | Nodules | Diameter range | Malignant share | AUC of diameter |
|---|---|---|---|---|
| small | 332 | 3.3 to 5.7 mm | 4% (about 13 nodules) | 0.729 |
| medium | 331 | 5.7 to 8.7 mm | 19% | 0.779 |
| large | 331 | 8.7 to 32.3 mm | 80% | 0.767 |

**Interpretation.**
- Annotated diameter alone gives AUC 0.918, and scanner settings add 0.004. A CNN's overall AUC must be compared with this, not with 0.5.
- Diameter remains informative inside size groups (0.73 to 0.78), so an image model has room to improve in each group (roughly 0.22 to 0.27 AUC).
- The small group has about 13 malignant nodules, so its AUC is highly uncertain. Bootstrap confidence intervals will be reported.
- The diameter used here is a radiologist-derived measurement. A CNN can estimate size from the image, so it may learn size first.

**Decision.** Reframe RQ1: *does an image model add predictive value beyond nodule diameter?* Evaluate overall and within size groups.

---

## 4. Key numbers at a glance

| Quantity | Value |
|---|---|
| Nodules after aggregation (with 0332 split) | 997 (650 benign, 347 malignant) |
| Nodules with an image and mask (classification set) | 994 (650 benign, 344 malignant) |
| Excluded nodules | 3, all malignant (LIDC-IDRI-1002 nodules 1 and 2, and one LIDC-IDRI-0332 nodule) |
| Full CT scans available | 3 |
| Orientation in VOIs | 972 LPS, 22 RAS (all converted to RAS on load) |
| In-plane spacing | 0.461 to 0.977 mm |
| Slice thickness | 0.45 to 2.5 mm, 10 distinct values |
| HU range | -1445 to 3071, `int16` |
| Median VOI extent | 30 x 30 x 8 mm |
| Diameter-only AUC (grouped CV) | 0.918 |
| Scanner-only AUC (grouped CV) | 0.566 |

---

## 5. Decisions carried into preprocessing (Phase 3)

| # | Finding | Decision |
|---|---|---|
| 1 | Mixed LPS / RAS orientation | Reorient images and masks identically on load |
| 2 | Pixel size 0.46 to 0.98 mm, slice thickness 0.45 to 2.5 mm | Resample to a common spacing (test 1 mm isotropic and a coarser variant as an ablation) |
| 3 | HU extremes (-1445 to 3071) | Clip to a fixed window, then scale to 0-1 (test the upper limit) |
| 4 | VOIs thin in z and variable in size | Decide patch strategy after the padding-leakage check |
| 5 | Diameter is a strong predictor | Evaluate against the diameter baseline overall and per size group |
| 6 | Thickness correlates with size | Stratify splits by label and thickness group; report per-group results |
| 7 | Multiple nodules per patient | Split by patient (never by nodule), with an assertion that no patient appears in two splits |
| 8 | 34.6% malignant | Class weights or balanced sampling; report AUC, AUPRC, sensitivity, specificity |
| 9 | Coordinate convention verified | Reuse `(-x, -y, z)` with each file's affine for any patch extraction from full CTs |

---

## 6. Limitations and threats to validity

- **Labels are radiologist opinion, not pathology.** They come from subjective 1-5 ratings thresholded by a rule. Models learn to imitate this opinion, and larger nodules are rated more suspicious, which partly explains the strength of diameter.
- **Rule sensitivity.** "At least two raters above 3" is a different bar for 3, 4 and 7 raters. Alternative labels are still to be compared.
- **Few malignant nodules in the small group** (about 13), so conclusions inside that group are weak.
- **Only 3 full CT scans** are available, which blocks the detection task and the extraction of patches with real surrounding tissue.
- **Possible hospital or protocol effects** beyond slice thickness cannot be examined from the available metadata alone.
- **VOI extents may carry size information** (padding concern), not yet tested.
- **Single dataset**, no external validation.

---

## 7. Remaining EDA work

| Item | Purpose | Status |
|---|---|---|
| Label sensitivity (PDF rule vs. majority vs. mean rating), rating histogram | Quantify how many labels depend on the rule | Code written, output not yet reviewed |
| Class balance and diameter-by-class figures | Report figures | Code written, figure not yet reviewed |
| Resolve LIDC-IDRI-0332 owner of the VOI | Confirm which nodule the file belongs to | Final table has 994 unique rows, but the printed distances should be recorded in the log |
| Padding-leakage check (`mm_z` vs. diameter, AUC of `mm_z`) | Choose patch strategy | Not run |
| Distribution of other radiologist features (spiculation, margin, texture, and so on) and their relation to malignancy | Understand feature signal and annotator subjectivity | Not done |
| Annotator disagreement per nodule (standard deviation of ratings) | Characterise label noise | Not done |
| `AcqParams.xlsx` | Compare with header-derived spacing, check kVp and other settings | Not opened |
| HU histograms and example images (benign vs. malignant, very bright, very faint) | Visual understanding for the report | Not done |
| Save `data/processed/nodules.csv` and the image inventory | Reproducible inputs for later phases | Image inventory saved, nodules table saved by the finalising cell |
| Obtain the full CT dataset | Needed for the detection task | Not done |

---

## 8. Reproducibility notes

Functions and checks to move from the notebook into `src/` with unit tests:

| Function / check | Purpose |
|---|---|
| `is_malignant(ratings)` | Label rule, tested on the two examples in the dataset description |
| `offset_vox(voi, x, y, z)` | Convert annotation coordinates to voxels using `(-x, -y, z)` and the inverse affine, expressed relative to the VOI centre |
| `load_canonical(path)` | Load an image or mask and reorient to a common orientation |
| VOI-equals-CT-crop test | Regression test for the coordinate conversion (LIDC-IDRI-0001_R_1 at corner (292, 334, 85)) |
| `describe_file(path, kind)` | Header inventory of every file |

Outputs written so far: `data/processed/image_inventory.csv`, `data/processed/nodules.csv`. Figures belong in `results/figures/eda/`.

Software used: Python, pandas, NumPy, nibabel, matplotlib, scikit-learn (logistic regression, grouped cross-validation, ROC AUC), SciPy (chi-square test). Pin exact versions in `environment.yml`.

Statistical notes: AUC values from cross-validation are pooled out-of-fold predictions with folds grouped by patient. The within-size-group AUCs use raw diameter as a ranking score (no model fitting). No random seed has been fixed yet for any split; this will be done in Phase 4.

---

## 9. Appendix: log entries (copy to `LOG.md`)

- 996 groups became 997 nodules after splitting LIDC-IDRI-0332 (two nodules under one ID, about 33 mm apart).
- Classification set = 994 nodules with image and mask (650 benign, 344 malignant). Three malignant nodules excluded for lack of an image.
- Coordinate convention: negate x and y, then apply the inverse affine of the file. Verified on 30 random VOIs, on all 22 RAS files, and by exact VOI-to-CT crop equality on one case.
- 22 of 994 VOIs are stored as RAS, the rest as LPS. All are reoriented on load.
- Slice thickness has 10 distinct values (0.45 to 2.5 mm), in-plane pixel size 0.46 to 0.98 mm. Resampling is required.
- Thickness and malignancy are associated (29%, 33%, 40%; p = 0.0053), but the association disappears within nodule-size groups. Scanner-only AUC 0.566.
- Diameter-only AUC 0.918, within-size-group AUC 0.73 to 0.78. The image model must be compared against this baseline.
