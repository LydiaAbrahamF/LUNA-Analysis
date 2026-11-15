# CNN input ablation plan (trainval only)

## Question

How do the planned field of view, number of fixed-mm slices, and HU upper clipping limit affect the exploratory CNN under the shared patient-level trainval folds?

## Fixed comparisons

All variants use the existing small 2D CNN, optimizer, batch size, early-stopping rule, inner patient-grouped validation split, outer folds, seed, and score metrics from `CNN_TRAINING_PLAN.md`.

| Variant | Field of view | Input shape | Slice offsets | HU window |
|---|---:|---:|---|---|
| `cnn_32mm_1slice_hu400` | 32 mm | 1 × 64 × 64 | 0 mm | [-1000, 400] |
| `cnn_40mm_1slice_hu400` | 40 mm | 1 × 80 × 80 | 0 mm | [-1000, 400] |
| `cnn_32mm_3slice_hu400` | 32 mm | 3 × 64 × 64 | -1, 0, +1 mm | [-1000, 400] |
| `cnn_32mm_1slice_hu600` | 32 mm | 1 × 64 × 64 | 0 mm | [-1000, 600] |

The pixel size remains 0.5 mm in-plane. The mask is used only for centring and QA, never as a model input. Padding fraction is not a model input. A diameter-only logistic model is recomputed on the same outer folds as a reference.

## Evaluation boundary

Use exactly the 828 shared trainval nodules and outer folds 0–4. Each variant selects its epoch on one fixed patient-grouped inner split of the outer training data, refits on that outer training partition, and predicts its held-out trainval fold. Report mean and standard deviation across outer folds for ROC-AUC and average precision; threshold-0.5 measures are descriptive. Do not read or score the held-out test set. No hyperparameter search is performed.

This is a single-seed exploratory input ablation. Padding and resolution are related, the 40 mm VOI crop may still contain air outside the available VOI, and only one training seed is used. Results cannot establish a universally best input or replace independent validation.
