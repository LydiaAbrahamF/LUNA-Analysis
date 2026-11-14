"""Create aggregate trainval-only HU and reader-agreement figures.

Uses only summary CSVs; no image, mask, patient-level, or test-set data is read.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "results" / "eda"
OUT = ROOT / "results" / "figures" / "eda"
WIDTH, HEIGHT = 960, 560


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def svg_start(title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, sans-serif" fill="#202124">',
        f'<text x="{WIDTH/2}" y="34" text-anchor="middle" font-size="21" font-weight="600">{esc(title)}</text>',
    ]


def svg_end(parts: list[str], path: Path) -> None:
    parts += ['</g>', '</svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def hu_plot(path: Path) -> None:
    rows = read_csv(EDA / "hu_histogram_trainval.csv")
    series = {}
    for label in ("benign", "malignant"):
        vals = sorted((float(r["hu_bin_midpoint"]), float(r["mean_nodule_normalized_density"]))
                      for r in rows if r["label"] == label)
        if not vals:
            raise ValueError(f"No histogram data for {label}")
        series[label] = vals

    left, right, top, bottom = 92, 900, 80, 455
    ymin, ymax = 1e-7, 3e-3
    xmap = lambda x: left + (x + 1500) / 4600 * (right - left)
    ymap = lambda y: bottom - (math.log10(max(y, ymin)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (bottom - top)
    p = svg_start("Trainval masked-region HU distribution",
                  "Equal-weight mean of per-nodule normalized histogram densities, shown on a logarithmic vertical scale. Benign n=542; malignant n=286. Trainval only.")
    for exponent in range(-7, -2):
        y = ymap(10.0 ** exponent)
        p.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dadce0" stroke-width="1"/>')
        p.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="13">10^{exponent}</text>')
    for tick in range(-1500, 3101, 500):
        x = xmap(tick)
        p.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#eceff1" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{bottom+24}" text-anchor="middle" font-size="13">{tick}</text>')
    p.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#5f6368"/>')
    p.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#5f6368"/>')
    for label, color, n in (("benign", "#1a73e8", 542), ("malignant", "#d93025", 286)):
        pts = " ".join(f'{xmap(x):.1f},{ymap(y):.1f}' for x, y in series[label] if y >= ymin)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    p.append(f'<text x="{(left+right)/2}" y="{HEIGHT-34}" text-anchor="middle" font-size="15">HU in nodule mask</text>')
    p.append(f'<text transform="translate(25 {(top+bottom)/2}) rotate(-90)" text-anchor="middle" font-size="15">Mean per-nodule density (log scale)</text>')
    for i, (label, color, n) in enumerate((("Benign (n=542)", "#1a73e8", 542), ("Malignant (n=286)", "#d93025", 286))):
        y = 87 + i * 25
        p.append(f'<line x1="690" y1="{y}" x2="722" y2="{y}" stroke="{color}" stroke-width="3"/>')
        p.append(f'<text x="730" y="{y+5}" font-size="14">{label}</text>')
    p.append('<text x="92" y="510" font-size="12" fill="#5f6368">Each nodule is normalized before averaging; zeros below plotting floor are omitted.</text>')
    svg_end(p, path)


def agreement_plot(path: Path) -> None:
    rows = read_csv(EDA / "annotator_feature_agreement_trainval.csv")
    features = ["Calcification", "InternalStructure", "Lobulation", "Margin", "Sphericity", "Spiculation", "Subtlety", "Texture"]
    vals = {}
    for feature in features:
        vals[feature] = {}
        for label in ("benign", "malignant"):
            match = [r for r in rows if r["feature"] == feature and r["label"] == label]
            if len(match) != 1:
                raise ValueError(f"Expected one agreement row for {feature}/{label}")
            vals[feature][label] = float(match[0]["fraction_unanimous"])
    left, right, top, bottom = 230, 900, 76, 455
    p = svg_start("Trainval radiologist-feature unanimity",
                  "Fraction of nodules with unanimous ratings, shown separately by dataset malignancy label for eight radiologist-rated features. Trainval only.")
    for tick in (0, .2, .4, .6, .8, 1):
        x = left + tick * (right-left)
        p.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#dadce0" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{bottom+24}" text-anchor="middle" font-size="13">{tick:.0%}</text>')
    group_h = (bottom-top) / len(features)
    for i, feature in enumerate(features):
        y0 = top + i * group_h
        p.append(f'<text x="{left-12}" y="{y0+group_h/2+5:.1f}" text-anchor="end" font-size="14">{esc(feature)}</text>')
        for j, label in enumerate(("benign", "malignant")):
            y = y0 + 7 + j * 20
            width = vals[feature][label] * (right-left)
            color = "#1a73e8" if label == "benign" else "#d93025"
            p.append(f'<rect x="{left}" y="{y:.1f}" width="{width:.1f}" height="13" fill="{color}"/>')
            p.append(f'<text x="{min(left+width+6,right+3):.1f}" y="{y+11:.1f}" font-size="12">{vals[feature][label]:.1%}</text>')
    p.append(f'<text x="{(left+right)/2}" y="{HEIGHT-34}" text-anchor="middle" font-size="15">Nodules with unanimous ratings</text>')
    for i, (label, color) in enumerate((("Benign (n=542)", "#1a73e8"), ("Malignant (n=286)", "#d93025"))):
        x = 655 + i * 150
        p.append(f'<rect x="{x}" y="490" width="14" height="14" fill="{color}"/>')
        p.append(f'<text x="{x+20}" y="502" font-size="13">{label}</text>')
    p.append('<text x="230" y="535" font-size="12" fill="#5f6368">Descriptive annotation agreement; malignancy labels and feature ratings come from the same reader process.</text>')
    svg_end(p, path)


def main() -> None:
    hu_plot(OUT / "hu_distribution_trainval.svg")
    agreement_plot(OUT / "annotator_agreement_trainval.svg")
    print(f"Saved aggregate figures to {OUT}")


if __name__ == "__main__":
    main()
