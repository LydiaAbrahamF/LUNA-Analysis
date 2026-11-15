"""Run the fixed CNN input ablation on the shared 828 trainval VOIs only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import PreprocessConfig, preprocess_images
from src.train import SmallNoduleCNN, choose_epoch, fit_epochs, metrics, predict


VARIANTS = {
    "cnn_32mm_1slice_hu400": PreprocessConfig(
        name="cnn_32mm_1slice_hu400", window_mm=32.0, out_px=64,
        offsets_mm=(0.0,), hu_min=-1000.0, hu_max=400.0),
    "cnn_40mm_1slice_hu400": PreprocessConfig(
        name="cnn_40mm_1slice_hu400", window_mm=40.0, out_px=80,
        offsets_mm=(0.0,), hu_min=-1000.0, hu_max=400.0),
    "cnn_32mm_3slice_hu400": PreprocessConfig(
        name="cnn_32mm_3slice_hu400", window_mm=32.0, out_px=64,
        offsets_mm=(-1.0, 0.0, 1.0), hu_min=-1000.0, hu_max=400.0),
    "cnn_32mm_1slice_hu600": PreprocessConfig(
        name="cnn_32mm_1slice_hu600", window_mm=32.0, out_px=64,
        offsets_mm=(0.0,), hu_min=-1000.0, hu_max=600.0),
}
BASELINE = "cnn_32mm_1slice_hu400"


def load_trainval(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    proc = root / "data" / "processed"
    index = pd.read_csv(proc / "patches_v1_trainval_index.csv")
    if len(index) != 828 or set(index["split"]) != {"trainval"}:
        raise ValueError("Expected exactly 828 trainval patch-index rows")
    if index["voi_name"].duplicated().any():
        raise ValueError("Repeated VOI IDs in trainval index")
    split = pd.read_csv(proc / "splits.csv", usecols=["voi_name", "patient_id", "split", "fold"])
    shared = split.loc[split["split"] == "trainval", ["voi_name", "patient_id", "fold"]]
    if len(shared) != 828 or set(shared["voi_name"]) != set(index["voi_name"]):
        raise ValueError("Trainval index does not match the shared split")
    index = index.drop(columns=["patient_id", "fold"]).merge(
        shared, on="voi_name", validate="one_to_one")
    if index.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ValueError("A patient crosses trainval folds")
    if sorted(index["fold"].unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected shared outer folds 0–4")
    metadata = pd.read_csv(proc / "nodules.csv", usecols=["voi_name", "diameter_mm"])
    index = index.merge(metadata, on="voi_name", validate="one_to_one")
    y = index["malignant"].to_numpy(dtype=int)
    if len(y) != 828 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid trainval labels")
    return index.reset_index(drop=True), y


def create_variant_arrays(root: Path, voi_root: Path, index: pd.DataFrame,
                          cache: Path) -> dict[str, Path]:
    proc = root / "data" / "processed"
    existing = proc / "patches_v1_trainval_X.npy"
    arrays = {BASELINE: existing}
    cache.mkdir(parents=True, exist_ok=True)
    for name, cfg in VARIANTS.items():
        if name == BASELINE:
            continue  # use the already QA-reviewed, exact baseline input array
        out = cache / f"{name}_X.npy"
        if out.is_file():
            arr = np.load(out, mmap_mode="r")
            if arr.shape != (828, len(cfg.offsets_mm), cfg.out_px, cfg.out_px) or not np.isfinite(arr).all():
                raise ValueError(f"Invalid cached input array: {out}")
            arrays[name] = out
            continue
        arr = np.empty((828, len(cfg.offsets_mm), cfg.out_px, cfg.out_px), dtype=np.float32)
        for i, voi in enumerate(index["voi_name"]):
            image_path = voi_root / "image" / voi
            mask_path = voi_root / "nodule_mask" / voi
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing trainval image-mask pair for {voi}")
            image = nib.load(str(image_path))
            mask = nib.load(str(mask_path))
            patch = preprocess_images(image, mask, cfg)
            if patch.x.shape != arr.shape[1:] or not np.isfinite(patch.x).all():
                raise ValueError(f"Invalid patch for {voi} under {name}")
            arr[i] = patch.x
            if (i + 1) % 100 == 0 or i + 1 == len(index):
                print(f"{name}: processed {i + 1}/828 trainval VOIs")
        np.save(out, arr)
        arrays[name] = out
    baseline = np.load(existing, mmap_mode="r")
    if baseline.shape != (828, 1, 64, 64) or not np.isfinite(baseline).all():
        raise ValueError("Invalid existing baseline input array")
    return arrays


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--voi-root", type=Path,
                        default=Path.home() / "Desktop" / "LUNA_Dataset" / "VOIs")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, voi_root = args.project_root.resolve(), args.voi_root.resolve()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)

    index, y = load_trainval(root)
    arrays = create_variant_arrays(root, voi_root, index,
                                   root / "data" / "processed" / "cnn_input_ablations")
    # All image loading above iterates only the shared trainval index.
    folds = sorted(index["fold"].unique().tolist())
    diameter = index["diameter_mm"].to_numpy(dtype=float).reshape(-1, 1)
    results_dir = root / "results" / "cnn"
    results_dir.mkdir(parents=True, exist_ok=True)
    score_records: list[dict] = []
    fold_records: list[dict] = []
    oof_rows: list[dict] = []

    for variant, array_path in arrays.items():
        X = np.load(array_path, mmap_mode="r")
        cnn_scores = np.full(828, np.nan, dtype=float)
        print(f"\nTraining {variant} ({X.shape[1:]} input)")
        for fold in folds:
            val_i = np.flatnonzero(index["fold"].to_numpy() == fold)
            train_i = np.flatnonzero(index["fold"].to_numpy() != fold)
            if set(index.iloc[train_i]["patient_id"]) & set(index.iloc[val_i]["patient_id"]):
                raise ValueError(f"Patient leakage in fold {fold}")
            epochs = choose_epoch(X[train_i], y[train_i],
                                  index.iloc[train_i]["patient_id"].to_numpy(),
                                  args.seed + fold, device)
            model = fit_epochs(X[train_i], y[train_i], epochs,
                               args.seed + 100 + fold, device)
            cnn_scores[val_i] = predict(model, X[val_i], device)
            row = {"variant": variant, "fold": fold, "n_train": len(train_i),
                   "n_validation": len(val_i), "selected_epochs": epochs,
                   **metrics(y[val_i], cnn_scores[val_i])}
            fold_records.append(row)
            print(f"fold {fold}: epochs={epochs}, ROC-AUC={row['roc_auc']:.3f}, AP={row['average_precision']:.3f}")
            for i in val_i:
                oof_rows.append({"variant": variant, "voi_name": index.iloc[i]["voi_name"],
                                 "fold": fold, "y_true": int(y[i]),
                                 "y_score": float(cnn_scores[i])})
        if np.isnan(cnn_scores).any():
            raise RuntimeError(f"Missing OOF scores for {variant}")
        folds_for_variant = [r for r in fold_records if r["variant"] == variant]
        pooled = metrics(y, cnn_scores)
        score_records.append({"variant": variant, "n_trainval": 828,
                              "oof_roc_auc": pooled["roc_auc"],
                              "oof_average_precision": pooled["average_precision"],
                              **{f"fold_mean_{key}": float(np.mean([r[key] for r in folds_for_variant]))
                                 for key in ("roc_auc", "average_precision", "accuracy",
                                             "balanced_accuracy", "sensitivity", "specificity")},
                              **{f"fold_sd_{key}": float(np.std([r[key] for r in folds_for_variant], ddof=1))
                                 for key in ("roc_auc", "average_precision", "accuracy",
                                             "balanced_accuracy", "sensitivity", "specificity")}})

    # Same-fold diameter-only logistic reference, fitted without the validation fold.
    ref_scores = np.full(828, np.nan, dtype=float)
    for fold in folds:
        val_i = np.flatnonzero(index["fold"].to_numpy() == fold)
        train_i = np.flatnonzero(index["fold"].to_numpy() != fold)
        baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=5000,
                                                    random_state=args.seed))
        baseline.fit(diameter[train_i], y[train_i])
        ref_scores[val_i] = baseline.predict_proba(diameter[val_i])[:, 1]
        row = {"variant": "diameter_logistic_reference", "fold": fold,
               "n_train": len(train_i), "n_validation": len(val_i),
               **metrics(y[val_i], ref_scores[val_i])}
        fold_records.append(row)
        for i in val_i:
            oof_rows.append({"variant": "diameter_logistic_reference",
                             "voi_name": index.iloc[i]["voi_name"], "fold": fold,
                             "y_true": int(y[i]), "y_score": float(ref_scores[i])})
    if np.isnan(ref_scores).any():
        raise RuntimeError("Missing diameter reference OOF scores")
    ref_rows = [r for r in fold_records if r["variant"] == "diameter_logistic_reference"]
    pooled_ref = metrics(y, ref_scores)
    score_records.append({"variant": "diameter_logistic_reference", "n_trainval": 828,
                          "oof_roc_auc": pooled_ref["roc_auc"],
                          "oof_average_precision": pooled_ref["average_precision"],
                          **{f"fold_mean_{key}": float(np.mean([r[key] for r in ref_rows]))
                             for key in ("roc_auc", "average_precision", "accuracy",
                                         "balanced_accuracy", "sensitivity", "specificity")},
                          **{f"fold_sd_{key}": float(np.std([r[key] for r in ref_rows], ddof=1))
                             for key in ("roc_auc", "average_precision", "accuracy",
                                         "balanced_accuracy", "sensitivity", "specificity")}})
    pd.DataFrame(fold_records).to_csv(results_dir / "cnn_input_ablation_by_fold_trainval.csv", index=False)
    pd.DataFrame(score_records).to_csv(results_dir / "cnn_input_ablation_summary_trainval.csv", index=False)
    # This row-level file includes case identifiers and stays local under data/processed.
    pd.DataFrame(oof_rows).to_csv(root / "data" / "processed" / "cnn_input_ablation_oof_trainval.csv", index=False)
    (root / "data" / "processed" / "cnn_input_ablations" / "configs.json").write_text(
        json.dumps({k: {**v.__dict__, "offsets_mm": list(v.offsets_mm)} for k, v in VARIANTS.items()}, indent=2))
    write_report(results_dir / "cnn_input_ablation_report_trainval.md", score_records)


def write_report(path: Path, rows: list[dict]) -> None:
    by_name = {r["variant"]: r for r in rows}
    order = ["cnn_32mm_1slice_hu400", "cnn_40mm_1slice_hu400",
             "cnn_32mm_3slice_hu400", "cnn_32mm_1slice_hu600",
             "diameter_logistic_reference"]
    labels = {
        "cnn_32mm_1slice_hu400": "CNN: 32 mm, 1 slice, HU max 400",
        "cnn_40mm_1slice_hu400": "CNN: 40 mm, 1 slice, HU max 400",
        "cnn_32mm_3slice_hu400": "CNN: 32 mm, 3 slices, HU max 400",
        "cnn_32mm_1slice_hu600": "CNN: 32 mm, 1 slice, HU max 600",
        "diameter_logistic_reference": "Diameter logistic reference",
    }
    lines = ["# CNN input ablation (trainval only)", "",
             "Four fixed CNN input configurations were compared under the five shared patient-level trainval folds (828 nodules). Epoch selection used one patient-grouped inner validation split within each outer training partition; the same architecture and optimization settings were used for each variant. The diameter logistic reference was refit on the same outer folds. Only trainval image-mask pairs were opened; no test images, labels, predictions, or metrics were used.", "",
             "| Input | Fold-mean ROC-AUC ± SD | Fold-mean AP ± SD | Fold-mean sensitivity | Fold-mean specificity | Pooled OOF ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name in order:
        r = by_name[name]
        lines.append(f"| {labels[name]} | {r['fold_mean_roc_auc']:.3f} ± {r['fold_sd_roc_auc']:.3f} | {r['fold_mean_average_precision']:.3f} ± {r['fold_sd_average_precision']:.3f} | {r['fold_mean_sensitivity']:.3f} | {r['fold_mean_specificity']:.3f} | {r['oof_roc_auc']:.3f} |")
    lines += ["", "## Interpretation", "",
              "This is a single-seed, exploratory comparison. Use the outer-fold means as the primary comparison; pooled OOF scores are secondary because separately fitted fold models may have different score scales. No subgroup or test-set result was used to select an input. The CNN variants were not evaluated on the test set.", "",
              "The 40 mm view contains more air padding for some VOIs and may not add real surrounding lung tissue beyond the stored crop. Three-slice sampling uses fixed physical offsets (-1, 0, +1 mm), but native slice thickness varies substantially. A single seed and five folds do not establish a universally best input; repeated seeds and independent validation would be needed for a stronger claim.", "",
              "Reproduce with `python -m src.run_cnn_input_ablations_trainval --voi-root /path/to/LUNA_Dataset/VOIs --device cpu`. Aggregate outputs: `cnn_input_ablation_summary_trainval.csv` and `cnn_input_ablation_by_fold_trainval.csv`. Per-nodule OOF scores and generated patch arrays remain local under `data/processed/` and must not be uploaded.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
