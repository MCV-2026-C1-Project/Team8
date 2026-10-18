"""QSD1 mAP@5 against the number of blocks per side (YCbCr 64-64-64, L1), with and without
photometric normalisation.

Reads the QSD1 sweeps in results/week1/ and results/week2/.

Run: python src/visualizations/plot_grid_size_sweep.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SWEEPS = ["results/week2/qsd1_pred_with_norm.csv", "results/week1/1d_none.csv", "results/week1/1d_photometric.csv"]
GRIDS = [1, 2, 3, 4, 5]
NORMS = [("photometric", "photometric normalisation", "#eb6834", 7), ("none", "no normalisation", "#2a78d6", -13)]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"


def ycbcr_rows(root):
    """YCbCr 64-64-64, L1, default ranges (no overflow bins)."""
    d = pd.concat([pd.read_csv(root / f) for f in SWEEPS], ignore_index=True)
    params = d["parameters"].fillna("{}").map(json.loads)
    d["norm"] = params.map(lambda p: p.get("normalization", "none"))
    default = params.map(lambda p: p.get("y_range", [0, 256]) == [0, 256]
                         and p.get("chroma_range", [0, 256]) == [0, 256] and not p.get("chroma_overflow", False))
    return d[(d.descriptor == "ycbcr") & (d.bins == "64-64-64") & (d.measure == "l1") & default]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "grid_size_sweep.png")
    args = ap.parse_args()
    d = ycbcr_rows(ROOT)

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.axvspan(3.65, 4.35, color="#f3f2ef", zorder=0)  # chosen grid
    for norm, label, color, dy in NORMS:
        rows = d[d.norm == norm]
        y = [rows[(rows.spatial_mode == ("global" if g == 1 else "grid")) &
                  (rows.spatial_grids == f"[{g}]")].map5.max() for g in GRIDS]
        ax.plot(GRIDS, y, color=color, lw=2, marker="o", ms=5, label=label)
        for g, v in zip(GRIDS, y):
            ax.annotate(f"{v:.2f}", (g, v), textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks(GRIDS, [f"{g}×{g}\n{192 * g * g}" for g in GRIDS], fontsize=8.5)
    ax.set_xlabel("blocks per side · descriptor length", color=MUTED)
    ax.set_ylabel("mAP@5", color=MUTED)
    ax.set_ylim(0.4, 1.05)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
