# Decisions log

Every choice below was made from cross-validation on the trainval nodules only.
The sealed test set had not been used when these were written.

## 1. Label (2026-11-10)
- Label = 1 (malignant) if the mean radiologist malignancy rating >= 3, else 0.
- Result on 994 nodules with image and mask: 550 malignant, 444 benign (55.3% malignant).
- 146 nodules (14.7%) have a mean of exactly 3.0 ("ties") and are counted as malignant.
- Why ties are malignant: <WRITE THIS IN YOUR OWN WORDS. Anchor it in the high-recall priority:
  an uncertain nodule should not be counted as benign.>
- Sensitivity labels (mean > 3: 404 malignant / 590 benign; dataset rule n_high >= 2) are kept in
  label_audit.csv for analysis only. The modelling table has one label column.
- The label is a radiologist rating, not a biopsy result. The model predicts radiologist-assessed
  suspicion, not confirmed cancer.

## 2. Split (locked 2026-11-14)
- Unit: patient. No patient appears in more than one split or fold.
- StratifiedGroupKFold, strata = label x slice-thickness group, seed 42.
- 6 outer chunks: 1 sealed test set, 5 trainval chunks used as the CV folds.
- Test: 166 nodules, 92 patients. Trainval: 828 nodules.
- splits.csv sha256: <PASTE: Get-FileHash data/processed/splits.csv -Algorithm SHA256>
- The split is not regenerated. A changed hash means a changed split and invalidates all results.

## 3. Features and protocol (2026-11-14)
- 107 PyRadiomics features (shape 14, first-order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14,
  NGTDM 5), bin width 25, no filters.
- Primary protocol: original spacing. Sensitivity analysis: 1 mm isotropic.
- Evidence (10-seed repeated CV, radiomics): 0.894 vs 0.882; paired difference +0.012,
  95% CI [-0.000, +0.025]. Supported for first-order (+0.024, CI [+0.008, +0.041]) and texture
  (+0.020, CI [+0.003, +0.037]); borderline for the full set. 11 comparisons were run, so
  borderline results are not claimed as findings.
- <ONE SENTENCE IN YOUR OWN WORDS on why original spacing is the primary protocol.>

## 4. What the features add beyond size (2026-11-14)
- Scanner settings alone: AUC 0.530 (95% CI 0.483-0.578), i.e. chance. No scanner shortcut.
- Diameter alone: AUC 0.828 [0.799, 0.857].
- Radiomics beat diameter by +0.065 AUC (95% CI [+0.040, +0.091]).
- Texture features carry the gain: diameter + texture vs diameter +0.062 [+0.039, +0.086];
  all radiomics vs diameter + texture +0.003 [-0.004, +0.011]. Shape and first-order add nothing
  once diameter and texture are included.
- Within diameter tertiles, radiomics AUC was 0.75 / 0.81 / 0.90 (small / medium / large) against
  0.66 / 0.63 / 0.74 for diameter, so the gain is not only nodule size.

## 5. Model choice (2026-11-14)
- Candidates: logistic regression (untuned, L2-tuned, elastic net), random forest, gradient
  boosting, RBF SVM, and an equal-weight average of L2 logistic regression, random forest and
  gradient boosting. Hyperparameters were tuned inside each training fold on patient-grouped
  inner splits.
- All models landed within 0.88-0.90 AUC. No model beat the L2 logistic regression by a margin
  that excludes 0 on diameter + texture.
- PRIMARY: L2 logistic regression on diameter + texture (76 features), original spacing.
  CV AUC 0.892 [0.869, 0.914]. Reason: the simplest model tied for best.
- SECONDARY (reported next to the primary, not chosen after seeing test results): equal-weight
  ensemble on all 107 radiomics features. CV AUC 0.902 [0.879, 0.923].
- Stacking was not used: it learns weights from correlated base models on 828 nodules and needs
  a nested patient-grouped CV to avoid leakage.

## 6. Decision threshold (2026-11-14)
- The label threshold (mean rating >= 3) is separate from the decision threshold below.
- Rule: flag a nodule as suspicious if predicted probability >= t, with t chosen so that
  cross-validated sensitivity is >= 95%.
- Primary t = 0.264 (95% interval for the cutoff 0.203-0.323). CV sensitivity 0.952
  [0.931, 0.973], specificity 0.528 [0.474, 0.582], 74% of nodules flagged.
- Secondary t = 0.266. CV sensitivity 0.952 [0.931, 0.972], specificity 0.534 [0.479, 0.590].
- This treats a missed malignancy as about 2.8 times as bad as a false alarm ((1 - t) / t).
- Calibration (out-of-fold): primary slope 1.008, intercept -0.006; secondary slope 1.114.
- Why 95% and not 90% or 98%: from 90% to 95% recall specificity falls about 10 points, from 95%
  to 98% about 18 points, and at 98% recall 83% of nodules are flagged.

## 7. Test protocol
- The sealed test set is scored once, by pipeline/10_final_test.py, for the primary model, the
  secondary model, and a diameter-only reference. Nothing is retuned or re-chosen afterwards.
- Result is reported with patient-level bootstrap 95% CIs, whatever it is.
- Test run date: <FILL IN AFTER RUNNING>
- Test result summary: <FILL IN AFTER RUNNING>

## Known limitations
- Labels are radiologist ratings, not pathology.
- 55% malignant share in this dataset: PPV does not transfer to a screening population.
- 92 test patients (about 91 malignant nodules): test sensitivity will have a wide interval.
- The recall threshold was chosen on the same cross-validated predictions it is evaluated on, so
  realised sensitivity on new patients is expected to be slightly below 95%.