# Real-data preprocessing QA (trainval only)

- Inputs: 828 shared trainval VOIs; held-out test rows and images were not processed.
- Patch array: `(828, 1, 64, 64)`, dtype `float32`, finite values, range 0.0–1.0.
- Empty patch masks: 0; mask area min/median/max: 43/172.5/2902 pixels.
- Mask-centroid offset from patch centre: mean 0.81px, median 0.54px, max 6.00px.
- Input orientations: {'LPS': 818, 'RAS': 10}; image-mask shape/affine mismatches: 0. All image headers canonicalized to RAS by the preprocessing routine.
- Padding: any padding in 76.6%; median 0.136, 90th percentile 0.377, maximum 0.629.
- Raw `pad_fraction` ROC-AUC for malignant label: 0.419; reverse-direction AUC: 0.581.
- Spearman correlation of padding with: diameter_mm -0.317, sx -0.942, sz -0.285.
- Mean padding by slice-thickness group: 1.25: 0.148, thick (>=1.8): 0.129, thin (<=1.0): 0.205.

## Visual review

The deterministic class montage and three hard-case panels show nodules located within the 32 mm patches, with masks aligned to the visible nodule in the reviewed samples. No obvious orientation or centring failure was seen in these previews. This is a visual spot-check, not a per-sample clinical review.

## Interpretation and limits

The pipeline produced finite, correctly shaped trainval patches with no empty masks or image-mask geometry mismatches. However, padding is strongly associated with native in-plane spacing (|Spearman rho| about 0.94) and appears in most patches. This suggests a possible acquisition-resolution shortcut. The raw padding score alone has weak label discrimination (AUC 0.419; reversed 0.581), but the spacing association means preprocessing QA is not a blanket approval of the representation. Do not alter the configuration based on the held-out test set; any padding/window ablation must be specified and compared using trainval only.

Patch arrays and preview images are local working data; do not upload them with raw images or masks. Preview files: `data/processed/qa_previews/`.

## Planned 32 mm vs 40 mm field-of-view check

The comparison used the same 0.5 mm per pixel scale: 32 mm → 64 × 64 and 40 mm → 80 × 80. Only the 828 trainval VOIs were processed; no classifier was fitted and no test data were read.

| Quantity | 32 mm | 40 mm |
|---|---:|---:|
| Patches with any padding | 76.6% | 97.6% |
| Median padding fraction | 0.136 | 0.447 |
| 90th percentile padding fraction | 0.377 | 0.602 |
| Maximum padding fraction | 0.629 | 0.762 |
| Raw padding ROC-AUC for malignant label | 0.419 | 0.383 |
| Padding vs. native in-plane spacing (Spearman ρ) | −0.942 | −0.921 |
| Masks touching the patch edge | 3.6% | 1.6% |

The central 32 mm of the 40 mm output was pixel-identical to the 32 mm output for every nodule. The wider field captured additional mask pixels beyond the 32 mm boundary in 27 of 828 nodules, reducing edge-touching masks, but greatly increased air padding because these are cropped VOIs rather than full CT scans. This is a geometry and padding comparison only; it does not establish which window gives better classification performance. Do not change the model input based on these summaries alone. Any model comparison must use a pre-specified trainval-only procedure, with the locked test set untouched.
