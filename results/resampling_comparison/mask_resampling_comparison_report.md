# Three-sample mask resampling check

## Method

For each of the three prespecified VOIs, the image and same-named mask were loaded and checked for matching size, spacing, origin, and direction (tolerance 1e-6); masks were also checked for binary values and presence of label 1. Archived results from the earlier separately calculated manual output grid were compared with a fresh run of PyRadiomics 3.0.1 `imageoperations.resampleImage` at 1 mm isotropic spacing, label 1, B-spline image interpolation, and 5-voxel padding. The label map is resampled with nearest-neighbour interpolation by PyRadiomics. The fresh PyRadiomics mask is measured on its own ROI-aligned output grid. Volume is voxel count × voxel spacing product.

## Results

All three image-mask pairs passed the geometry and label checks. Manual-grid versus PyRadiomics ROI-grid mask voxels were: **0292_R_1, 124 vs 126** (volume change −10.02% vs −8.57%); **0470_R_2, 6456 vs 6516** (+0.19% vs +1.12%); and **0144_R_2, 503 vs 524** (−2.86% vs +1.20%). Thus the two grid definitions produce different voxel counts and can even change the direction of the measured volume change, as in 0144_R_2.

## Limitations

This is a three-case geometry check, not a comparison over all nodules and not evidence that either grid is globally more accurate. The older manual-grid output was read from its archived CSV; its original grid-generation script/settings were not available for an independent rerun. These measurements describe binary-mask volume only and do not establish which extraction protocol is preferable for downstream prediction. Do not generalize the result beyond these three samples.

Reproduce with `python -m src.compare_mask_resampling_grids`. Aggregate check table: `results/resampling_comparison/mask_resampling_comparison_updated.csv`.
