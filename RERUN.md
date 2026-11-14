# Rerun under the dataset label rule (PowerShell, from the repo root)

1. Copy the files in this zip over your repo (same paths). `report_main.tex` replaces `report/paper/main.tex`.
2. Archive everything produced under the old label, so nothing mixes and the test seal resets:

   mkdir results_mean_label
   Move-Item results\tables results_mean_label\tables
   Move-Item results\figures results_mean_label\figures
   mkdir results\tables, results\figures
   Move-Item data\processed\splits.csv data\processed\splits_mean_label.csv
   Move-Item data\processed\nodules.csv data\processed\nodules_mean_label.csv
   Move-Item data\processed\label_audit.csv data\processed\label_audit_mean_label.csv
   (features.csv stays: features do not depend on the label.)

3. python -m pytest -q
4. python pipeline/01_build_dataset.py --qa      -> must print: label rule: dataset | 994 | malignant 344 | benign 650
5. python pipeline/02_make_split.py              -> note the test chunk size and patients (group report: 166 / 91) and the sha256
6. Stages 4, 5, 6, 7, 8, 9 as before (stage 3 is NOT needed again).
   Stage 9b (python pipeline/09b_subgroups_decision_curve.py) must wait until step 7 is done: it reads PRIMARY_MODEL and SECONDARY_MODEL from config.py.
7. Decide the primary and secondary models from the stage 8 CV tables (PRIMARY_MODEL / SECONDARY_MODEL in config.py are still the
   OLD choices; confirm or change them from CV only), copy the stage 9 recall-95% cutoffs into FINAL_THRESHOLDS, and record all of it in
   docs/decisions.md BEFORE stage 10.
8. python pipeline/10_final_test.py --confirm --expect-sha256 <hash from step 5>
