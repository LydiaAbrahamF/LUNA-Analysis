# CNN trainval error analysis (descriptive)

This analysis uses only the 828 shared trainval out-of-fold predictions and trainval metadata. It does not read the held-out test set. The 0.5 threshold is a descriptive fixed threshold, not tuned.

ROC-AUC and average precision are reported for subgroup ranking. Sensitivity, specificity and accuracy use score >= 0.5. Small subgroup counts make estimates uncertain; no subgroup result is used to select a model.

| Group | Category | n | malignant | CNN AUC | CNN AP | CNN sens. | CNN spec. | Diameter AUC | Diameter AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_trainval | all | 828 | 286 | 0.836 | 0.738 | 0.510 | 0.928 | 0.913 | 0.840 |
| label_consensus | all_rules_agree | 663 | 183 | 0.882 | 0.748 | 0.601 | 0.938 | 0.949 | 0.874 |
| label_consensus | ambiguous | 165 | 103 | 0.673 | 0.780 | 0.350 | 0.855 | 0.745 | 0.795 |
| padding | positive_high | 202 | 58 | 0.721 | 0.518 | 0.328 | 0.903 | 0.879 | 0.689 |
| padding | positive_low | 239 | 87 | 0.824 | 0.753 | 0.437 | 0.947 | 0.914 | 0.886 |
| padding | positive_mid | 193 | 56 | 0.874 | 0.764 | 0.518 | 0.949 | 0.924 | 0.842 |
| padding | zero | 194 | 85 | 0.901 | 0.838 | 0.706 | 0.908 | 0.946 | 0.888 |
| size_bin | large | 276 | 218 | 0.656 | 0.863 | 0.638 | 0.603 | 0.742 | 0.897 |
| size_bin | medium | 267 | 55 | 0.700 | 0.360 | 0.127 | 0.962 | 0.762 | 0.425 |
| size_bin | small | 285 | 13 | 0.556 | 0.082 | 0.000 | 0.971 | 0.770 | 0.144 |
| thickness | 1.25 | 229 | 76 | 0.807 | 0.719 | 0.434 | 0.941 | 0.918 | 0.872 |
| thickness | thick (>=1.8) | 319 | 129 | 0.861 | 0.793 | 0.581 | 0.921 | 0.929 | 0.859 |
| thickness | thin (<=1.0) | 280 | 81 | 0.815 | 0.667 | 0.469 | 0.925 | 0.892 | 0.796 |

## Interpretation limits

Padding groups are zero padding, then tertiles among positive padding (cut points 0.136 and 0.261; determined from trainval only). These are descriptive subgroup results from a single fixed trainval OOF run. The fold models can have different score calibration; subgroup AUC/AP rank within the pooled OOF scores and should be interpreted cautiously. Subgroups were not used for tuning. The ambiguity flag marks a nodule when either the majority-vote or mean-rating alternative label differs from the dataset label. This analysis does not estimate clinical utility and does not replace the locked test evaluation.

The CNN and diameter scores are compared on the same nodules. The test set remains untouched.

Reproduce with `python -m src.analyze_cnn_errors_trainval`. Aggregate CSV: `results/cnn/cnn_trainval_subgroup_metrics.csv`.
