# Task 2 — Five-fold feature comparison

## Objective
Compare Diameter, Radiomics, and Diameter + Radiomics using preassigned trainval folds 0–4, separately for isotropic_1mm and original_spacing. The held-out test set is not used in this evaluation.

## Results

| Protocol | Features | Mean AUC ± SD | Mean accuracy | OOF AUC | Mean sensitivity | Mean specificity |
|---|---|---:|---:|---:|---:|---:|
| isotropic_1mm | Diameter | 0.9125 ± 0.0188 | 84.54% | 0.9126 | 66.77% | 93.91% |
| isotropic_1mm | Radiomics | 0.9245 ± 0.0153 | 86.47% | 0.9235 | 78.32% | 90.78% |
| isotropic_1mm | Diameter + Radiomics | 0.9242 ± 0.0149 | 86.47% | 0.9236 | 78.32% | 90.78% |
| original_spacing | Diameter | 0.9125 ± 0.0188 | 84.54% | 0.9126 | 66.77% | 93.91% |
| original_spacing | Radiomics | 0.9274 ± 0.0180 | 88.28% | 0.9269 | 81.48% | 91.88% |
| original_spacing | Diameter + Radiomics | 0.9274 ± 0.0181 | 88.28% | 0.9269 | 81.48% | 91.88% |

## Method
- 828 trainval nodules, 5 preassigned folds; 107 Radiomics features or 108 features when diameter is included.
- Median imputation, standardization, Logistic Regression (C=1.0, solver=lbfgs, max_iter=5000); preprocessing fit within each training fold only.
- Accuracy, sensitivity and specificity use a probability threshold of 0.5. Reported means and sample standard deviations are calculated across five folds. OOF AUC is calculated from pooled out-of-fold predictions.

## Interpretation and limitations
Radiomics features outperform diameter alone on mean AUC for both protocols. Original spacing gives slightly higher Radiomics AUC than isotropic 1 mm. Adding diameter does not materially change AUC. These are descriptive comparisons; no statistical significance test has been performed. Results are trainval-only and should not be presented as held-out test performance.

## Reproduction (Windows PowerShell)
Place splits.csv, nodules.csv and features_trainval.csv in the directory passed as `--data-dir` (the current directory in the commands below), then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe evaluate_aligned.py --data-dir . --output-dir results_aligned --protocol isotropic_1mm
.\.venv\Scripts\python.exe evaluate_aligned.py --data-dir . --output-dir results_original_aligned --protocol original_spacing
```

Use the aligned evaluation script. The three input CSVs must be supplied separately; they are not part of the five-file delivery. The reported results were generated with scikit-learn 1.5.2, and reruns under different dependency versions should be checked for consistency. On the user’s Windows environment, scikit-learn 1.5.2 produced an `OptimizeWarning: Unknown solver options: iprint` but the evaluations completed.

## Deliverables
- Evaluation script: evaluate_aligned.py (existing user file)
- Execution notes: this report
- Consolidated results: task2_combined_results.csv
- Detailed per-fold results and OOF predictions: generated in the respective output folders
