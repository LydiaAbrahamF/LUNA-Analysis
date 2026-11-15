"""Exploratory 5-fold CNN baseline using shared trainval folds only.

The held-out test set is never opened by this script. Hyperparameters and the
fold-local early-stopping rule are recorded in CNN_TRAINING_PLAN.md.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, roc_auc_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallNoduleCNN(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for out_channels in (16, 32, 64, 96):
            blocks.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            current_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool, seed: int) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y, dtype=np.float32)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def fit_epochs(x: np.ndarray, y: np.ndarray, epochs: int, seed: int,
               device: torch.device) -> nn.Module:
    seed_everything(seed)
    model = SmallNoduleCNN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = make_loader(x, y, batch_size=32, shuffle=True, seed=seed)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def choose_epoch(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 seed: int, device: torch.device) -> int:
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_i, val_i = next(inner.split(np.zeros(len(y)), y, groups))
    seed_everything(seed)
    model = SmallNoduleCNN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = make_loader(x[train_i], y[train_i], 32, True, seed)
    best_auc, best_epoch, stale = -np.inf, 1, 0
    for epoch in range(1, 81):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_scores = predict(model, x[val_i], device)
        auc = roc_auc_score(y[val_i], val_scores)
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
            if stale >= 10:
                break
    return best_epoch


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    root = args.project_root.resolve()
    proc = root / "data" / "processed"
    results = root / "results" / "cnn"
    results.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(proc / "patches_v1_trainval_index.csv")
    splits = pd.read_csv(proc / "splits.csv", usecols=["voi_name", "patient_id", "split", "fold"])
    trainval_splits = splits.loc[splits["split"] == "trainval"]
    expected = set(trainval_splits["voi_name"])
    if set(index["split"]) != {"trainval"} or set(index["voi_name"]) != expected:
        raise ValueError("CNN input must contain exactly the shared trainval set")
    if index["voi_name"].duplicated().any() or index["patient_id"].isna().any():
        raise ValueError("Duplicate VOIs or missing patient groups in the trainval index")
    folds = sorted(index["fold"].unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected shared folds 0–4, got {folds}")
    if index.groupby("patient_id")["fold"].nunique().max() != 1:
        raise ValueError("A patient appears in multiple trainval folds")
    y = index["malignant"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected both binary classes in trainval")
    X = np.load(proc / "patches_v1_trainval_X.npy", mmap_mode="r")
    if X.shape != (len(index), 1, 64, 64) or not np.isfinite(X).all():
        raise ValueError(f"Unexpected or invalid CNN array: {X.shape}")
    meta = pd.read_csv(proc / "nodules.csv", usecols=["voi_name", "diameter_mm"])
    meta = meta.loc[meta["voi_name"].isin(expected)]
    index = index.merge(meta, on="voi_name", validate="one_to_one")
    diameter = index["diameter_mm"].to_numpy(dtype=float).reshape(-1, 1)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
    print(f"Device: {device}; trainval only: {len(index)} nodules")

    cnn_scores = np.full(len(index), np.nan)
    diameter_scores = np.full(len(index), np.nan)
    fold_rows = []
    for fold in folds:
        val_i = np.flatnonzero(index["fold"].to_numpy() == fold)
        train_i = np.flatnonzero(index["fold"].to_numpy() != fold)
        if set(index.iloc[train_i]["patient_id"]) & set(index.iloc[val_i]["patient_id"]):
            raise ValueError(f"Patient leakage in outer fold {fold}")
        best_epoch = choose_epoch(X[train_i], y[train_i],
                                  index.iloc[train_i]["patient_id"].to_numpy(),
                                  args.seed + fold, device)
        model = fit_epochs(X[train_i], y[train_i], best_epoch,
                           args.seed + 100 + fold, device)
        cnn_scores[val_i] = predict(model, X[val_i], device)

        baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=5000, random_state=args.seed))
        baseline.fit(diameter[train_i], y[train_i])
        diameter_scores[val_i] = baseline.predict_proba(diameter[val_i])[:, 1]
        fold_rows.append({
            "fold": fold,
            "n_train": len(train_i),
            "n_validation": len(val_i),
            "cnn_selected_epochs": best_epoch,
            **{f"cnn_{k}": v for k, v in metrics(y[val_i], cnn_scores[val_i]).items()},
            **{f"diameter_{k}": v for k, v in metrics(y[val_i], diameter_scores[val_i]).items()},
        })
        print(f"fold {fold}: selected epochs={best_epoch}; "
              f"CNN AUC={fold_rows[-1]['cnn_roc_auc']:.3f}; "
              f"diameter AUC={fold_rows[-1]['diameter_roc_auc']:.3f}")

    if np.isnan(cnn_scores).any() or np.isnan(diameter_scores).any():
        raise RuntimeError("Missing out-of-fold predictions")
    pd.DataFrame(fold_rows).to_csv(results / "cnn_trainval_by_fold.csv", index=False)
    summary = []
    fold_table = pd.DataFrame(fold_rows)
    for method, scores in (("CNN_v1_32mm", cnn_scores), ("Diameter_logistic", diameter_scores)):
        prefix = "cnn" if method == "CNN_v1_32mm" else "diameter"
        pooled = metrics(y, scores)
        row = {
            "method": method,
            "n_trainval": len(y),
            "oof_roc_auc": pooled["roc_auc"],
            "oof_average_precision": pooled["average_precision"],
        }
        for metric_name in ("roc_auc", "average_precision", "accuracy",
                            "balanced_accuracy", "sensitivity", "specificity"):
            values = fold_table[f"{prefix}_{metric_name}"]
            row[f"fold_mean_{metric_name}"] = float(values.mean())
            row[f"fold_sd_{metric_name}"] = float(values.std(ddof=1))
        summary.append(row)
    pd.DataFrame(summary).to_csv(results / "cnn_trainval_summary.csv", index=False)
    predictions = index[["voi_name", "patient_id", "fold"]].copy()
    predictions["y_true"] = y
    predictions["cnn_score"] = cnn_scores
    predictions["diameter_score"] = diameter_scores
    predictions.to_csv(proc / "cnn_oof_predictions_trainval.csv", index=False)
    print(f"Saved aggregate results to {results}")
    print("Per-nodule predictions saved locally under data/processed/; do not upload them.")


if __name__ == "__main__":
    main()
