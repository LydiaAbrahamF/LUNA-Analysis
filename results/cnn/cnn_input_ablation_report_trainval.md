# CNN input ablation (trainval only)

Four fixed CNN input configurations were compared under the five shared patient-level trainval folds (828 nodules). Epoch selection used one patient-grouped inner validation split within each outer training partition; the same architecture and optimization settings were used for each variant. The diameter logistic reference was refit on the same outer folds. The held-out test set was not read.

| Input | Fold-mean ROC-AUC ± SD | Fold-mean AP ± SD | Fold-mean sensitivity | Fold-mean specificity | Pooled OOF ROC-AUC |
|---|---:|---:|---:|---:|---:|
| CNN: 32 mm, 1 slice, HU max 400 | 0.885 ± 0.018 | 0.820 ± 0.021 | 0.510 | 0.928 | 0.836 |
| CNN: 40 mm, 1 slice, HU max 400 | 0.885 ± 0.026 | 0.802 ± 0.035 | 0.629 | 0.887 | 0.864 |
| CNN: 32 mm, 3 slices, HU max 400 | 0.877 ± 0.018 | 0.801 ± 0.025 | 0.734 | 0.821 | 0.849 |
| CNN: 32 mm, 1 slice, HU max 600 | 0.889 ± 0.017 | 0.811 ± 0.027 | 0.667 | 0.860 | 0.854 |
| Diameter logistic reference | 0.912 ± 0.019 | 0.847 ± 0.039 | 0.668 | 0.939 | 0.913 |

## Interpretation

This is a single-seed, exploratory comparison. Use the outer-fold means as the primary comparison; pooled OOF scores are secondary because separately fitted fold models may have different score scales. No subgroup or test-set result was used to select an input. The test set was not evaluated.

The 40 mm view contains more air padding for some VOIs and may not add real surrounding lung tissue beyond the stored crop. Three-slice sampling uses fixed physical offsets (-1, 0, +1 mm), but native slice thickness varies substantially. A single seed and five folds do not establish a universally best input; repeated seeds and independent validation would be needed for a stronger claim.

Reproduce with `python -m src.run_cnn_input_ablations_trainval --voi-root /path/to/LUNA_Dataset/VOIs --device cpu`. Aggregate outputs: `cnn_input_ablation_summary_trainval.csv` and `cnn_input_ablation_by_fold_trainval.csv`. Per-nodule OOF scores and generated patch arrays remain local under `data/processed/` and must not be uploaded.
