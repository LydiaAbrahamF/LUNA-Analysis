# Trainval-only CNN baseline plan

## Question

Can a small CNN using the fixed 32 mm central-slice patches rank malignant-labelled nodules above the existing diameter baseline under the shared patient-level trainval folds?

## Data boundary

- Use `data/processed/patches_v1_trainval_X.npy` and its trainval index only (828 nodules).
- Use only the shared trainval folds 0–4. No test images, labels, predictions, or metrics are read.
- Do not feed the nodule mask or `pad_fraction` to the model. The mask is used only to create the patch.
- Keep per-nodule out-of-fold predictions under ignored `data/processed/`.

## Fixed model and training procedure

- Input: one channel, 64 × 64, values scaled to [0, 1].
- Small four-block 2D CNN with channels 16, 32, 64, 96, global average pooling, dropout 0.25, and one logit output.
- Binary cross-entropy with logits, AdamW (learning rate 0.001, weight decay 0.0001), batch size 32.
- At most 80 epochs; patience 10 based on ROC-AUC in one patient-grouped inner validation split drawn from each outer training partition.
- Refit on the full outer training partition for the chosen epoch count; score the held-out trainval fold once.
- Repeat for the five shared outer folds. Compare with a diameter-only logistic-regression baseline fitted on the same outer folds.
- Report ROC-AUC and average precision as primary metrics; threshold-0.5 metrics are descriptive. No hyperparameter or preprocessing search is performed in this baseline.

This is an exploratory trainval baseline, not a final model result. The 32 mm patch padding is associated with native in-plane spacing. A subsequent four-input ablation, including a 40 mm CNN input, is specified in `CNN_INPUT_ABLATION_PLAN.md` and uses trainval only. The held-out test set has already been used for the pre-specified Radiomics baseline comparison; do not use it to tune or select CNN inputs. Any additional test analysis requires an explicit documented decision by the team.
