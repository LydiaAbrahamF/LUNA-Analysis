# LUNA-Analysis — Task 2: 5-fold feature comparison

## Purpose
Compare three feature sets for binary lung nodule classification: **Diameter**, **Radiomics**, and **Diameter + Radiomics**, using the *existing* `trainval` fold IDs 0–4. No held-out `test` rows are used.

## Input files
Place these CSVs in the same folder as `evaluate_aligned.py`:
- `splits.csv`: `voi_name`, `patient_id`, `split`, `fold`
- `nodules.csv`: `voi_name`, `diameter_mm`, `malignant`
- `features_trainval.csv`: radiomics features and `protocol`

The script joins data by `voi_name`, checks patient-level fold consistency and label consistency, and selects 107 `original_*` radiomics features, excluding `original_spacing_x/y/z` (acquisition metadata, not radiomics descriptors).

## Requirements and Windows / VS Code instructions
Use Python 3.10+ in VS Code's terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate_aligned.py --data-dir . --output-dir results_aligned
```


For the alternative protocol (original spacing):

```powershell
.\.venv\Scripts\python.exe evaluate_aligned.py --data-dir . --output-dir results_original_aligned --protocol original_spacing
```

## Method
- 828 `trainval` nodules in 5 preassigned folds; each fold is held out once. The 166 `test` nodules are never used.
- Features: diameter only (1); radiomics only (107); both (108).
- Protocol: `isotropic_1mm` by default. `original_spacing` can be assessed separately, not mixed with the primary comparison.
- Fixed classifier: `SimpleImputer(median) -> StandardScaler -> LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)`.
- All preprocessing is fit on training rows within each fold only. No hyperparameter tuning is performed on validation results.
- Metrics: ROC AUC (ranking), accuracy, balanced accuracy, sensitivity, specificity (the latter four at probability threshold 0.5).
- Summary reports unweighted mean and sample standard deviation across the five folds, plus AUC from concatenated out-of-fold predictions (`oof_auc`).

## Outputs

The evaluation script generates three CSV files for each Radiomics protocol:

- `results_summary.csv`: Mean and standard deviation of five-fold evaluation metrics, including ROC AUC, accuracy, balanced accuracy, sensitivity, specificity, and OOF AUC.
- `results_by_fold.csv`: Detailed results for each of the five folds and three feature sets (15 rows per protocol).
- `oof_predictions.csv`: Out-of-fold predictions for each nodule and feature set.

The results are saved in two separate folders:

- `results_aligned/`: Isotropic 1 mm protocol.
- `results_original_aligned/`: Original spacing protocol.

The combined comparison is available in `task2_combined_results.csv`.

### Summary of Results

| Protocol | Feature Set | Mean AUC | Accuracy | OOF AUC |
|---|---|---:|---:|---:|
| Isotropic 1 mm | Diameter | 0.9125 | 0.8454 | 0.9126 |
| Isotropic 1 mm | Radiomics | 0.9245 | 0.8647 | 0.9235 |
| Isotropic 1 mm | Diameter + Radiomics | 0.9242 | 0.8647 | 0.9236 |
| Original spacing | Diameter | 0.9125 | 0.8454 | 0.9126 |
| Original spacing | Radiomics | 0.9274 | 0.8828 | 0.9269 |
| Original spacing | Diameter + Radiomics | 0.9274 | 0.8828 | 0.9269 |

Original-spacing Radiomics achieved the highest mean AUC (approximately 0.9274). Adding diameter did not substantially improve performance. These differences are descriptive; no statistical significance testing was performed.

## Interpretation and limitations
This is an initial **baseline**, not a final clinical diagnostic model. Labels are derived from radiologist malignancy ratings rather than pathology-confirmed diagnosis. Radiomics features are correlated and high-dimensional relative to sample size; further feature selection, calibration, and hyperparameter tuning would require an inner cross-validation loop. Any choice of protocol/model based on these trainval results must not use the held-out test set. Report the same split and fixed settings for all methods. Performance differences here are descriptive; no statistical significance test is performed.

## Reproducibility note

The script requires `splits.csv`, `nodules.csv`, and `features_trainval.csv` in the directory passed to `--data-dir`. These input datasets are not included in this delivery folder and must be obtained through the project data workflow. The reported numbers were generated in a Windows Python 3.12 environment with scikit-learn 1.5.2. A SciPy `OptimizeWarning` concerning `iprint` was observed; the run completed, but reruns with other dependency versions should be checked for consistency.
