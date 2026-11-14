#!/usr/bin/env bash
# Run the whole pipeline in order. Expects the dataset in data/ (or LUNA_DATA).
# Stages 1-9 use cross-validation only; the sealed test set is scored once, by stage 10:
#     python pipeline/10_final_test.py --confirm
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q
python pipeline/01_build_dataset.py --qa
python pipeline/02_make_split.py
python pipeline/03_extract_features.py
python pipeline/04_run_models.py
python pipeline/05_robustness.py
python pipeline/06_repeated_cv.py
python pipeline/07_compare.py
python pipeline/08_model_comparison.py
python pipeline/09_threshold_calibration.py
python pipeline/09b_subgroups_decision_curve.py
python pipeline/09c_figures.py
# Stage 10 is deliberately not run here: it uses the sealed test set once.
echo "Pipeline finished. Tables are in results/tables/"
