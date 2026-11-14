# Trainval-only selected-feature comparison plan

## Question

Does selecting a smaller Radiomics feature subset improve or preserve trainval performance compared with the diameter-only and all-feature baselines?

## Data boundary

- Read `results/radiomics_trainval/features_trainval.csv`, `data/processed/nodules.csv`, and `data/processed/splits.csv` only.
- Use the shared five outer trainval folds (0–4), grouped by patient.
- Do not read test feature tables, test predictions, or test metrics in this analysis.

## Prespecified method

- Compare original-spacing and 1 mm isotropic Radiomics separately.
- Use outer folds 0–4 for out-of-fold evaluation.
- Within each outer training portion, choose the number of selected Radiomics features from k = 5, 10, 20, or 50 using four-fold `StratifiedGroupKFold` inner CV grouped by patient.
- Rank features with univariate ANOVA F-scores (`SelectKBest`). Choose k by mean inner-CV ROC-AUC; if scores tie, the smaller k is preferred because candidates are evaluated in ascending order.
- Keep logistic regression fixed at L2, `C=1.0`, `max_iter=5000`; fit median imputation and standardisation inside each CV training fold. Do not tune any choices using the outer validation fold.
- Compare diameter only, all Radiomics, diameter plus all Radiomics, selected Radiomics, and diameter plus selected Radiomics. The diameter-only and all-feature rows are recomputed within the same outer folds for a like-for-like comparison.
- Report pooled outer-fold ROC-AUC and average precision, plus selected k and feature membership by outer fold. This is a trainval model-development analysis, not a final test evaluation.

## Interpretation limits

The inner CV selects k and the outer folds estimate performance for this selection procedure. Do not use these results to claim external validation or to tune against the held-out test set. The test set remains untouched.
