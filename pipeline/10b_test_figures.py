"""Stage 10b: figures of the test-set result. Draws only; nothing is fitted or scored again.

Reads   results/tables/final_test_predictions.csv (written by the single stage-10 run)
Writes  results/figures/test_roc.png, test_confusion.png and results/tables/test_confusion.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from lunarad import config as cfg

INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"
STYLE = {"primary": dict(color=INK, ls="-", lw=2.0), "secondary": dict(color=INK, ls="--", lw=1.4),
         "reference": dict(color=MUTED, ls=":", lw=1.8)}
NAME = {"primary": "primary (radiomics)", "secondary": "secondary (ensemble)", "reference": "diameter only"}


def main():
    predictions = pd.read_csv(cfg.TABLES / "final_test_predictions.csv")
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    for role, part in predictions.groupby("role", sort=False):
        y, p = part["malignant"], part["probability_malignant"]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{NAME.get(role, role)}, AUC {roc_auc_score(y, p):.3f}", **STYLE.get(role, {}))
        threshold = cfg.FINAL_THRESHOLDS.get(role)
        if threshold is not None:                      # mark the pre-declared operating point
            flagged = p >= threshold
            ax.plot((flagged & (y == 0)).sum() / (y == 0).sum(), (flagged & (y == 1)).sum() / (y == 1).sum(),
                    "o", ms=6, mec=INK, mfc="white" if role == "secondary" else INK)
    ax.set_xlabel("1 - specificity", fontsize=9, color=MUTED)
    ax.set_ylabel("sensitivity", fontsize=9, color=MUTED)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "test_roc.png", dpi=300)
    plt.close(fig)

    rows = []
    for role in ("primary", "secondary"):
        part = predictions[predictions["role"] == role]
        threshold = cfg.FINAL_THRESHOLDS.get(role)
        if part.empty or threshold is None:
            continue
        y, flagged = part["malignant"] == 1, part["probability_malignant"] >= threshold
        rows.append({"role": role, "threshold": threshold, "tp": int((y & flagged).sum()), "fn": int((y & ~flagged).sum()),
                     "fp": int((~y & flagged).sum()), "tn": int((~y & ~flagged).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLES / "test_confusion.csv", index=False)

    fig, axes = plt.subplots(1, len(table), figsize=(3.1 * len(table), 3.0), squeeze=False)
    for ax, row in zip(axes[0], table.itertuples()):
        cells = [[row.tp, row.fn], [row.fp, row.tn]]
        ax.imshow(cells, cmap="Greys", vmin=0, vmax=max(max(c) for c in cells) * 1.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cells[i][j]), ha="center", va="center", fontsize=13, color=INK)
        ax.set_xticks([0, 1], ["flagged", "not flagged"], fontsize=8)
        ax.set_yticks([0, 1], ["malignant", "benign"], fontsize=8)
        ax.set_title(f"{row.role}, t = {row.threshold}", fontsize=9, color=INK)
        ax.tick_params(length=0, colors=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "test_confusion.png", dpi=300)
    plt.close(fig)
    print(table.to_string(index=False))
    print(f"wrote {cfg.FIGURES / 'test_roc.png'} and test_confusion.png")


if __name__ == "__main__":
    main()
