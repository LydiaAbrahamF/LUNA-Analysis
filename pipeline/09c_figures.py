"""Stage 9c: figures for the report, from cross-validation tables only (no test data).

1. model_auc_forest.png      cross-validated AUC of every model with its 95% patient-bootstrap interval.
2. paired_difference_forest.png   paired AUC differences with 95% intervals and a line at zero.

How to read figure 2: an interval that crosses zero means the difference is NOT statistically
significant at the 5% level (two-sided); a filled marker means the interval excludes zero.
Overlap of two separate intervals in figure 1 is NOT a test: use figure 2 for any claim about a difference.

Reads  results/tables/model_comparison.csv, model_comparison_paired.csv, paired_differences.csv
Writes results/figures/model_auc_forest.png, paired_difference_forest.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lunarad import config as cfg

INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, labelsize=8, colors=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def model_forest(table, path):
    inputs = list(dict.fromkeys(table["input"]))
    fig, axes = plt.subplots(1, len(inputs), figsize=(3.6 * len(inputs), 0.32 * table["model"].nunique() + 1.3),
                             sharex=True, squeeze=False)
    for ax, name in zip(axes[0], inputs):
        part = table[table["input"] == name].iloc[::-1]
        y = range(len(part))
        ax.hlines(y, part["roc_auc_ci_low"], part["roc_auc_ci_high"], color=INK, lw=1.6)
        ax.plot(part["roc_auc_mean"], y, "o", color=INK, ms=5)
        ax.set_yticks(list(y))
        ax.set_yticklabels(part["model"])
        ax.set_ylim(-0.6, len(part) - 0.4)
        ax.set_title(f"input: {name} ({int(part['n_features'].iloc[0])} features)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("cross-validated AUC (95% interval)", fontsize=8, color=MUTED)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def difference_forest(groups, path):
    """groups: list of (title, DataFrame with label, auc_difference, ci_low, ci_high)."""
    heights = [0.32 * len(g) + 0.9 for _, g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(6.2, sum(heights)), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, (title, part) in zip(axes[:, 0], groups):
        part = part.iloc[::-1]
        y = list(range(len(part)))
        excludes_zero = (part["ci_low"] > 0) | (part["ci_high"] < 0)
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.hlines(y, part["ci_low"], part["ci_high"], color=INK, lw=1.6)
        for yi, value, solid in zip(y, part["auc_difference"], excludes_zero):
            ax.plot(value, yi, "o", ms=5.5, mec=INK, mew=1.4, mfc=INK if solid else "white")
        ax.set_yticks(y)
        ax.set_yticklabels(part["label"])
        ax.set_ylim(-0.6, len(part) - 0.4)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    axes[-1, 0].set_xlabel("paired AUC difference, A minus B (95% interval)", fontsize=8, color=MUTED)
    axes[0, 0].plot([], [], "o", ms=5.5, mec=INK, mew=1.4, mfc="white", label="interval includes 0 (not significant)")
    axes[0, 0].plot([], [], "o", ms=5.5, mec=INK, mew=1.4, mfc=INK, label="interval excludes 0")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    models = pd.read_csv(cfg.TABLES / "model_comparison.csv")
    models = models[models["protocol"] == cfg.FINAL_PROTOCOL]
    model_forest(models, cfg.FIGURES / "model_auc_forest.png")

    groups = []
    inputs = pd.read_csv(cfg.TABLES / "paired_differences.csv").rename(columns={"comparison": "label"})
    groups.append(("Feature sets (logistic regression)", inputs))
    paired = pd.read_csv(cfg.TABLES / "model_comparison_paired.csv")
    for name, part in paired.groupby("input", sort=False):
        groups.append((f"Model families, input: {name}", part.rename(columns={"comparison": "label"})))
    difference_forest(groups, cfg.FIGURES / "paired_difference_forest.png")

    every = pd.concat([g for _, g in groups])
    crossing = int(((every["ci_low"] <= 0) & (every["ci_high"] >= 0)).sum())
    print(f"{crossing} of {len(every)} paired differences have a 95% interval that includes zero")
    print(f"wrote {cfg.FIGURES / 'model_auc_forest.png'} and paired_difference_forest.png")


if __name__ == "__main__":
    main()
