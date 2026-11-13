# LUNA16 Pulmonary Nodule Analysis: Progress Report 

Status date: 2026-11-14
Project: Pulmonary nodule malignancy classification (primary task) and nodule detection (secondary task) on the LUNA16 dataset
Progress: Metadata EDA, image geometry validation, and the shared patient-level split are complete. Radiomics features were extracted for the 994-nodule classification set, and the initial locked-split baseline evaluation is complete. A trainval-only feature-ranking stability screen is also complete. CNN real-data preprocessing QA, CNN modelling, detailed error analysis, and detection remain pending.

This report records the EDA, evaluation split, preprocessing design, Radiomics baseline, and feature-ranking stability screen. It documents the methods, findings, limitations, and remaining work.

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

**Decision.** Keep the dataset authors' rule as the primary label. Plan a sensitivity analysis with alternative rules (majority vote, mean rating above 3), because "at least two" has a different meaning for 3, 4, or 7 raters. The label-rule comparison is summarised in Section 3.11; model-level sensitivity analysis remains future work.

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

**Padding-leakage check (run).** The Spearman correlation between VOI depth `mm_z` and annotated diameter is **0.91**. In-plane width `mm_x` correlates 0.38 with diameter. Using a box dimension alone as a predictor of malignancy gives AUC **0.881** for `mm_z` (for comparison: diameter 0.918, scanner settings 0.566) and 0.62 for `mm_x`. The depth of a VOI is therefore almost a copy of the nodule size. VOI extent is a free shortcut: any pipeline that pads VOIs to a fixed size lets the amount of padding reveal the nodule size without the model looking at lung tissue. Size shown inside the image is legitimate information. Size encoded in the box extent or padding is an artefact that would not exist for new scans.

**Decision.** Do not feed zero-padded, variable-depth volumes to the classifier, and do not feed the nodule mask as an input channel (the mask may be used only for centring and cropping). Candidate designs: (a) 2.5D input, meaning the central axial slice (plus neighbours where available) at a fixed physical in-plane window; (b) a 3D crop to a small fixed depth, after counting how many VOIs are shallower than that depth; (c) fixed-size patches cut from the full CT scans once they are available (best option, since it gives real surrounding tissue with no box artefact). Report the `mm_z`-only AUC (0.881) as an additional shortcut baseline. Confirmed cause: VOI depth follows the annotation bounding box in z (Spearman 0.94 between VOI depth and bounding-box depth), and the bounding-box depth tracks diameter (0.89). VOI width does not follow the bounding box (0.33), which is why width leaks much less (AUC 0.62).

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
- In an earlier patient-grouped five-fold CV, annotated diameter alone achieved an AUC of 0.918; adding scanner settings gave 0.922. The current locked-split baseline reports a trainval OOF AUC of 0.913 for diameter, using a different evaluation setup.
- Raw diameter achieved AUCs from 0.73 to 0.78 within the three size groups. These are within-group ranking results, not separate fitted models.
- The small size group has only about 13 malignant nodules, so its within-group AUC is uncertain. An image model should be compared with the diameter and Radiomics baselines, not with chance alone.
- The diameter used here is a radiologist-derived measurement. A CNN can estimate size from the image, so it may learn size first.

**Decision.** Reframe RQ1: *does an image model add predictive value beyond nodule diameter?* Evaluate overall and within size groups.

### 3.11 Label noise: how much does the consensus rule matter?

**Goal.** Quantify how many labels depend on the choice of consensus rule, because labels are thresholded subjective ratings.

**Method.** Compared the dataset rule (at least two raters above 3) with a majority vote (more than half of the raters above 3) and with a mean rating above 3, on the 994 nodules.

**Result.**

| Dataset rule (PDF) | Majority benign | Majority malignant | Mean benign | Mean malignant |
|---|---|---|---|---|
| benign (650) | 650 | 0 | 575 | 75 |
| malignant (344) | **116** | 228 | 15 | 329 |

Ratings per nodule and number of ratings above 3:

| Raters | 0 high | 1 high | 2 high | 3 high | 4 high | 5 high | Nodules |
|---|---|---|---|---|---|---|---|
| 3 | 181 | 82 | 39 | 21 | | | 323 |
| 4 | 229 | 158 | **116** | 118 | 45 | | 666 |
| 5 | | | | | 2 | 1 | 3 |
| 6 | | | | | | 1 | 1 |
| 7 | | | | | | 1 | 1 |

- All 116 nodules that change under majority vote are exact 2-2 ties among 4 raters (two raters above 3, two at or below 3). They are 34% of all malignant labels and 11.7% of all nodules.
- The mean rule changes 90 nodules (9.1%): 75 benign to malignant and 15 malignant to benign.
- The dataset rule is the most permissive of the three.

**Decision.** Keep the dataset rule as the primary label, since it matches the dataset definition. The label-rule comparison here quantifies how many labels change under alternative rules. Retraining and evaluating models with alternative labels, and comparing performance on clear versus ambiguous nodules, remain future analyses. Whether model errors concentrate in ambiguous nodules has not yet been evaluated.

### 3.12 Patient-level split (locked before modelling)

**Goal.** Fix the evaluation protocol before any model is trained, so that the test set cannot influence design choices.

**Method.** `StratifiedGroupKFold` with groups = patient and strata = label combined with slice-thickness group. One of six folds (about 16.7%) is the held-out test set, and the remainder is split into 5 cross-validation folds. Seed 42. Assertions check that no patient appears in both test and train/validation, and in both a validation fold and its training folds. Saved to `data/processed/splits.csv`.

**Result.**

| Split | Fold | Nodules | Malignant share | Patients |
|---|---|---|---|---|
| test | none | 166 | 0.35 | 91 |
| trainval | 0 | 165 | 0.35 | 90 |
| trainval | 1 | 166 | 0.34 | 94 |
| trainval | 2 | 166 | 0.35 | 93 |
| trainval | 3 | 165 | 0.35 | 92 |
| trainval | 4 | 166 | 0.34 | 91 |

Slice-thickness mix (thin / 1.25 mm / thick) is 0.34 / 0.28 / 0.39 in both the test and the train/validation portions.

**Decision.** The shared split was fixed before modelling. The pre-specified baseline evaluation used the test set once; test results must not be used for further model, feature, or protocol selection. Any future tuning must use the five trainval folds only.

### 3.13 Preprocessing design (Phase 3)

**Goal.** Turn each nodule VOI into a fixed-size input for the classifier, without letting the way the dataset was built (box size, scanner resolution, file orientation) leak the answer.

**Implementation.** `src/preprocess.py` defines the preprocessing settings and dataset builder; unit tests are in `tests/test_preprocess.py`, and visual QA is in `02_preprocessing_qa`. The pipeline has been tested on synthetic data. When run on the real VOIs, it will write `patches_v1_X.npy`, `patches_v1_mask.npy` (for QA only), `patches_v1_index.csv`, and `patches_v1_config.json` to `data/processed/`. The real-data run has not yet been completed.

**Pipeline and rationale.**

| Step | Choice (v1) | Rationale from the EDA |
|---|---|---|
| 1. Orientation | Reorient image and mask to RAS on load | 22 of 994 VOIs are stored as RAS, the rest as LPS (Section 3.7) |
| 2. Centre | Centroid of the nodule mask, in physical coordinates | The mask outlines the nodule exactly. It is used only for centring and is never a model input, because it would reveal size |
| 3. In-plane window | 32 x 32 mm sampled to 64 x 64 pixels (0.5 mm per pixel) | Largest annotated diameter is 32.3 mm. Native pixel size varies from 0.46 to 0.98 mm, so a common physical scale is needed |
| 4. Slice | One axial slice at the nodule centre, linearly interpolated at an exact position in mm (optional extra slices at fixed mm offsets) | Slice thickness varies from 0.45 to 2.5 mm and VOI depth leaks nodule size (AUC 0.881), so no variable-depth 3D volume is used |
| 5. Intensity | Clip to [-1000, 400] HU, scale to [0, 1] | HU values range from -1445 to 3071 (bone, saturation). A fixed window keeps lung and nodule detail and needs no dataset statistics |
| 6. Outside the VOI | Fill with air (-1000 HU) and record the fraction as `pad_fraction` | VOIs are about 44 pixels wide, so many are narrower than 32 mm |

The image is sampled directly from the original voxel grid in physical space (the file's affine maps voxels to millimetres), so pixel size, slice thickness and orientation are handled in one interpolation step. Resampling cannot add detail: thick-slice scans remain blurrier after interpolation.

**Padding and shortcut audit.** The padded fraction is saved for every patch. The QA notebook tests whether it predicts the label (AUC), correlates with diameter, or differs by slice thickness. If it does, the padding strategy must change (edge replication, a smaller window, or excluding heavily padded VOIs).

**Tests (8, all passing).** Same patch for LPS and RAS storage of the same data; nodule centred; the same object sampled at two pixel sizes gives the same patch; the patch does not change when the VOI is cropped shallower in z; padding is air and is counted; HU clipping and scaling; output shape, type and range; the multi-slice option contains the central slice.

**Status.** Implemented and tested on a synthetic dataset that mimics the real data (mixed orientations, pixel sizes and slice thicknesses). The run on the real 994 VOIs and the QA notebook are pending.

**Planned ablations (RQ2).** Window 32 vs 40 mm, one slice vs three slices (offsets -1, 0, +1 mm), upper HU limit 400 vs 600.

### 3.14 Radiomics baseline and feature-ranking stability screen

**Radiomics extraction.** The confirmed target was `malignant`. Original-spacing and 1 mm isotropic features were extracted for all 994 nodules: 828 trainval nodules and 166 test nodules. Each nodule had 107 Radiomics features per protocol. The extracted tables had no extraction failures or non-finite feature values.

**Baseline evaluation.** The pre-specified comparison used five-fold out-of-fold predictions within trainval and evaluated the held-out test set once. Each model used L2 logistic regression (`C=1.0`), with median imputation and standardisation fitted within each training fold. No feature selection or hyperparameter tuning was performed.

| Protocol | Input | Trainval OOF ROC-AUC | Test ROC-AUC (95% patient-bootstrap CI) |
|---|---|---:|---:|
| Original spacing | Diameter | 0.913 | 0.961 (0.927–0.988) |
| Original spacing | Radiomics | 0.927 | 0.929 (0.892–0.964) |
| Original spacing | Diameter + Radiomics | 0.927 | 0.930 (0.893–0.965) |
| 1 mm isotropic | Diameter | 0.913 | 0.961 (0.927–0.988) |
| 1 mm isotropic | Radiomics | 0.924 | 0.937 (0.892–0.975) |
| 1 mm isotropic | Diameter + Radiomics | 0.924 | 0.937 (0.892–0.975) |

The test confidence intervals overlap. These initial results do not establish a clear Radiomics gain or a preferred resampling protocol. Test results were not used to tune or select a model or protocol.

**Feature-ranking stability screen.** A univariate ANOVA F-score ranking was run separately for each protocol and each trainval fold. Each ranking used only the other four trainval folds. Nine features appeared in the top 10 in all five folds for both protocols; most were shape or size-related. This is a descriptive stability screen only: no feature subset was selected or evaluated as a model, and the test set was not used.
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
| VOI-depth-only AUC | 0.881 (VOI width only: 0.62) |
| Held-out test set | 166 nodules, 91 patients |
| Shared trainval set | 828 nodules, 460 patients |
| Trainval folds | 5 patient-level folds, 165–166 nodules per fold |
| Confirmed classification target | `malignant` |
| Radiomics extraction | 994 nodules under each of 2 protocols; 107 features per nodule; no failures or non-finite values |
| Initial Radiomics baseline | 6 pre-specified model/protocol comparisons; five-fold trainval OOF plus one test evaluation |
| Feature-ranking stability screen | 9 features appeared in the top 10 in all five folds under both protocols; descriptive only |
| Preprocessing v1 status | Implemented and tested on synthetic data; real-data run and QA pending |

---

## 5. Decisions from the EDA and how Phase 3 implements them

| # | Finding | Decision |
|---|---|---|
| 1 | Mixed LPS / RAS orientation | Reorient images and masks identically on load |
| 2 | Pixel size 0.46 to 0.98 mm, slice thickness 0.45 to 2.5 mm | Resample to a common spacing (test 1 mm isotropic and a coarser variant as an ablation) |
| 3 | HU extremes (-1445 to 3071) | Clip to a fixed window, then scale to 0-1 (test the upper limit) |
| 4 | VOIs thin in z, variable in size, and their depth leaks nodule size (AUC 0.881) | Avoid padding to fixed volumes. Start with 2.5D input at a fixed in-plane window, keep 3D as an ablation, and move to full-CT patches when available |
| 5 | Diameter is a strong predictor; the initial locked-split baseline is complete | Compare image and Radiomics models against the diameter baseline. The earlier grouped-CV diameter AUC (0.918) and current locked-split trainval OOF AUC (0.913) come from different evaluation setups. Do not use test results to choose a model |
| 6 | Thickness correlates with size | Stratify splits by label and thickness group; report per-group results |
| 7 | Multiple nodules per patient | Split by patient (never by nodule), with an assertion that no patient appears in two splits |
| 8 | 34.6% malignant | Report ROC-AUC and average precision; any class weighting or resampling must be decided using training data only |
| 9 | Coordinate convention verified | Reuse `(-x, -y, z)` with each file's affine for any patch extraction from full CTs |
| 10 | 116 malignant labels (34%) are exact 2-2 rater ties; labels depend on the rule for about 9-12% of nodules | Keep the dataset rule as primary, add sensitivity analyses and a clear-vs-ambiguous breakdown |
| 11 | Patient-level split locked (test: 166 nodules from 91 patients; five trainval folds) | Use the shared split. The pre-specified baseline test comparison has been run once; keep the test set out of future tuning |
| 12 | Most consistently top-ranked Radiomics features are shape- or size-related | Treat the trainval-only ranking as descriptive. Do not call it a selected feature set or evidence of added value beyond diameter |

---

## 6. Limitations and threats to validity

- **Labels are radiologist opinion, not pathology.** They come from subjective 1-5 ratings thresholded by a rule. Models learn to imitate this opinion, and larger nodules are rated more suspicious, which partly explains the strength of diameter.
- **Rule sensitivity.** "At least two raters above 3" is a different bar for 3, 4 and 7 raters. 116 of the 344 malignant labels (34%) are exact 2-2 ties among 4 raters, and 90 nodules change under a mean-rating rule (Section 3.11).
- **Few malignant nodules in the small group** (about 13), so conclusions inside that group are weak.
- **Small held-out test set.** The initial Radiomics test comparison contains 166 nodules from 91 patients. Patient-bootstrap confidence intervals overlap, so these results do not establish a reliable performance difference between diameter and Radiomics or between resampling protocols.
- **Feature screen is exploratory.** The trainval-only univariate ranking is not a validated feature-selection procedure. Many consistently high-ranked features describe shape or size, and no selected-feature model has been evaluated.
- **Only 3 full CT scans** are available, which blocks the detection task and the extraction of patches with real surrounding tissue.
- **Possible hospital or protocol effects** beyond slice thickness cannot be examined from the available metadata alone.
- **VOI extents carry size information.** VOI depth alone gives AUC 0.881 and correlates 0.91 with diameter, so zero-padding to a fixed volume must be avoided (Section 3.8).
- **Single dataset**, no external validation.
- **Single-slice 2D input** discards 3D context. 3D and multi-slice variants are planned as ablations.
- **Padding frame.** Many VOIs are narrower than the 32 mm window, so patches contain an air-filled border. The padding audit on the real data decides whether this is acceptable.
- **Preprocessing validated on synthetic data only** so far. The real-data run and QA are pending.

---

## 7. Remaining EDA and project work

| Item | Purpose | Status |
|---|---|---|
| Label sensitivity and rating ambiguity | Quantify how labels depend on the consensus rule | Completed; see Section 3.11 |
| Class-balance and diameter-by-class figures | Prepare figures for the report | Code written; figures still need review |
| LIDC-IDRI-0332 VOI assignment | Ensure the correct nodule owns the shared VOI filename | Resolved for the final 994-nodule table; record the distance check in the audit log if needed |
| Padding-leakage check | Check whether VOI depth tracks nodule size | Completed; see Section 3.8 |
| Other radiologist features and their relation to malignancy | Describe feature signal and annotator subjectivity | Not done |
| Annotator disagreement per nodule | Characterise label noise | Not done |
| `AcqParams.xlsx` | Compare acquisition settings with header-derived spacing | Not opened |
| HU histograms and example images | Visually compare benign, malignant, bright, and faint nodules | Not done |
| Save processed metadata and image inventory | Keep reproducible input tables | Completed locally; raw data and patient-level files are not for GitHub upload |
| Obtain the full CT dataset | Enable the detection task | Not done; only 3 full CT scans are available locally |
| Run preprocessing on real VOIs and review QA | Confirm the CNN preprocessing and padding behaviour on real data | Pending; current validation is on synthetic data |
| Compare selected-feature models | Check whether fold-local feature selection helps within trainval | Not done; define the procedure using trainval only and keep test results out of selection |

---

## 8. Reproducibility notes

The shared patient-level split is stored in `data/processed/splits.csv`. It has five trainval folds (0–4) and a held-out test split (fold -1); no patient appears in both trainval and test. Use this shared file for all model development and evaluation. Do not regenerate the split for comparisons.

| Script | Purpose |
|---|---|
| `src/preprocess.py` | Build fixed-size CNN inputs from VOIs |
| `src/extract_radiomics_split.py` | Validate the shared metadata and extract Radiomics features for a requested split |
| `src/evaluate_baselines.py` | Run the pre-specified diameter and Radiomics baseline comparisons |
| `src/analyze_resampling_trainval.py` | Summarise feature changes between extraction protocols using trainval data |
| `src/analyze_feature_stability_trainval.py` | Rank feature stability using trainval folds only; this is descriptive, not model selection |
| `src/check_mask_resampling.py` | Compare mask geometry using PyRadiomics-aligned resampling |

The baseline used L2 logistic regression (`C=1.0`) with median imputation and standardisation fitted within each training fold. The shared five trainval folds produced out-of-fold predictions. The held-out test set was evaluated once under the pre-specified plan and was not used to tune or select models or protocols.

The feature-ranking stability screen uses only the other four trainval folds to rank features for each fold. It does not read the test set or evaluate a selected-feature model.

Key environment versions for the Radiomics run: Python 3.11.18, PyRadiomics 3.0.1, SimpleITK 2.5.7, and scikit-learn 1.9.1. Install the repository dependencies with `pip install -r requirements.txt`.

Per-nodule feature tables, test predictions, raw VOIs, and masks remain local and should not be uploaded to GitHub. Only scripts, documentation, and aggregate results belong in the repository.
---

## 9. Roadmap and next steps

| Phase | Content | Status |
|---|---|---|
| 0 | Repository, environment, related-work notes | Repository and environment are set up; related-work review remains pending |
| 1 | Metadata EDA | Core EDA complete; remaining items are listed in Section 7 |
| 2 | Image loading and geometry validation | Done |
| 3 | Preprocessing | Implemented and tested on synthetic data; real-data QA pending |
| 4 | Evaluation protocol and split | Shared patient-level split verified; initial baseline evaluation complete, with the test set used once |
| 5 | Baselines | Initial diameter, scanner, and Radiomics comparisons complete; selected-feature model comparison pending |
| 6 | CNN classifier | Not started; real-data preprocessing QA remains pending |
| 7 | Evaluation and error analysis | Initial baseline metrics complete; detailed error analysis pending |
| 8 | Detection (secondary task) | Not started; needs the full CT scans |
| 9 | Report and repository polish | In progress |

---

## 10. Appendix: log entries (copy to `LOG.md`)

- 996 groups became 997 nodules after splitting LIDC-IDRI-0332 (two nodules under one ID, about 33 mm apart).
- Classification set = 994 nodules with image and mask (650 benign, 344 malignant). Three malignant nodules excluded for lack of an image.
- Coordinate convention: negate x and y, then apply the inverse affine of the file. Verified on 30 random VOIs, on all 22 RAS files, and by exact VOI-to-CT crop equality on one case.
- 22 of 994 VOIs are stored as RAS, the rest as LPS. All are reoriented on load.
- Slice thickness has 10 distinct values (0.45 to 2.5 mm), in-plane pixel size 0.46 to 0.98 mm. Resampling is required.
- Thickness and malignancy are associated (29%, 33%, 40%; p = 0.0053), but the association disappears within nodule-size groups. Scanner-only AUC 0.566.
- Earlier patient-grouped CV: diameter-only AUC 0.918; raw-diameter AUCs within size groups were 0.73–0.78. The current locked-split trainval OOF diameter AUC is 0.913; these results use different evaluation setups. Compare image models against the diameter and Radiomics baselines, not against chance alone.
- VOI depth (`mm_z`) correlates 0.91 with diameter and alone gives AUC 0.881; in-plane width gives 0.62. Zero-padded fixed-size volumes would leak nodule size, so the patch design avoids them.
- VOI depth follows the annotation bounding box in z (Spearman 0.94); VOI width does not (0.33).
- 116 of 344 malignant labels are exact 2-2 rater ties (34%). The majority-vote rule would make 228 malignant, the mean-rating rule would change 90 nodules.
- Split locked: test 166 nodules (91 patients), 5 folds of about 165 nodules, balanced by label (34-35% malignant) and thickness mix. Seed 42, saved in `data/processed/splits.csv`.
- Preprocessing v1 implemented: orientation to RAS, centring on the mask centroid, 32 mm window at 0.5 mm per pixel (64 x 64), one central slice interpolated in mm, HU clipped to [-1000, 400] and scaled to [0, 1], air padding recorded as `pad_fraction`. 8 unit tests pass on synthetic data; real-data QA pending.
- Radiomics features: 994 nodules were extracted under original-spacing and 1 mm isotropic protocols, with 107 features per nodule and protocol and no extraction failures or non-finite values.
- Initial baseline: six pre-specified comparisons used five-fold trainval OOF predictions and one held-out test evaluation. Test results were not used for tuning or selection; confidence intervals overlap.
- Feature-ranking screen: nine features appeared in the top 10 in all five trainval folds under both protocols. This descriptive screen did not select or evaluate a feature subset.
