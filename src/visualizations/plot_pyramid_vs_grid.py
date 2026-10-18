"""QSD1 mAP@5 of spatial pyramids (equal level weights) against the grid of their finest level,
with and without photometric normalisation (YCbCr 64-64-64, L1).

Reads the QSD1 sweeps in results/week1/ and results/week2/.

Run: python src/visualizations/plot_pyramid_vs_grid.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SWEEPS = ["results/week2/qsd1_pred_with_norm.csv", "results/week1/1d_none.csv", "results/week1/1d_photometric.csv"]
PAIRS = [((1, 2), 2), ((1, 2, 3), 3), ((1, 2, 4), 4)]
NORM_TITLES = {"photometric": "Photometric normalisation", "none": "No normalisation"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a66", "#e5e4e0"


def ycbcr_rows(root):
    """YCbCr 64-64-64, L1, default ranges (no overflow bins)."""
    d = pd.concat([pd.read_csv(root / f) for f in SWEEPS], ignore_index=True)
    params = d["parameters"].fillna("{}").map(json.loads)
    d["norm"] = params.map(lambda p: p.get("normalization", "none"))
    default = params.map(lambda p: p.get("y_range", [0, 256]) == [0, 256]
                         and p.get("chroma_range", [0, 256]) == [0, 256] and not p.get("chroma_overflow", False))
    return d[(d.descriptor == "ycbcr") & (d.bins == "64-64-64") & (d.measure == "l1") & default]


def bar_labels(ax, bars):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8.5, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "pyramid_vs_grid.png")
    args = ap.parse_args()
    d = ycbcr_rows(ROOT)

    x = np.arange(len(PAIRS))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for ax, norm in zip(axes, ("photometric", "none")):
        rows = d[d.norm == norm]
        grid_vals = [rows[(rows.spatial_mode == "grid") & (rows.spatial_grids == f"[{g}]")].map5.max()
                     for _, g in PAIRS]
        pyr = rows[(rows.spatial_mode == "pyramid")
                   & rows.level_weights.map(lambda w: len(set(np.round(json.loads(w), 6))) == 1)]
        pyr_vals = [pyr[pyr.spatial_grids == "[" + ",".join(map(str, p)) + "]"].map5.max() for p, _ in PAIRS]
        bar_labels(ax, ax.bar(x - 0.2, grid_vals, 0.38, color=BLUE, label="finest grid only"))
        bar_labels(ax, ax.bar(x + 0.2, pyr_vals, 0.38, color=ORANGE, label="pyramid, equal level weights"))
        ax.set_xticks(x, [f"{g}×{g} grid\nvs pyramid {'-'.join(map(str, p))}" for p, g in PAIRS], fontsize=8.5)
        ax.set_title(NORM_TITLES[norm], color=INK, fontsize=11)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("mAP@5", color=MUTED)
    axes[0].legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
