"""Best global mAP@5 of each descriptor type on QSD1, without and with photometric normalisation.

Reads the QSD1 sweeps in results/week1/ and results/week2/ (global rows only: no blocks).
HSV is not normalised, so its normalised bar is marked n/a.

Run: python src/visualizations/plot_descriptor_types.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SWEEPS = ["results/week2/qsd1_pred_with_norm.csv", "results/week1/1d_none.csv", "results/week1/1d_photometric.csv"]
TYPES = [("Lab 1D", "lab"), ("YCbCr 1D", "ycbcr"), ("Lab 2D\na-b + L", "lab_ab2d"),
         ("HSV 2D\nH-S + V", "hsv_hs2d"), ("Lab 3D", "lab_3d"), ("HSV 3D", "hsv_3d")]
NORMS = [("none", "no normalisation", "#2a78d6"), ("photometric", "photometric normalisation", "#eb6834")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"


def load_sweeps(root):
    d = pd.concat([pd.read_csv(root / f) for f in SWEEPS], ignore_index=True)
    d["norm"] = d["parameters"].fillna("{}").map(lambda p: json.loads(p).get("normalization", "none"))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "descriptor_types_map5.png")
    args = ap.parse_args()
    d = load_sweeps(ROOT)
    d = d[d.spatial_mode == "global"]

    x = np.arange(len(TYPES))
    fig, ax = plt.subplots(figsize=(9.2, 2.55))
    for offset, (norm, label, color) in zip((-0.2, 0.2), NORMS):
        values = []
        for _, desc in TYPES:
            rows = d[(d.descriptor == desc) & (d.norm == norm)]
            values.append(rows.map5.max() if len(rows) else np.nan)
        bars = ax.bar(x + offset, np.nan_to_num(values), 0.38, color=color, label=label)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03, "n/a", ha="center", fontsize=8.5, color=MUTED)
            else:
                ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                            xytext=(0, 2), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [name for name, _ in TYPES], fontsize=9)
    ax.set_ylabel("mAP@5", color=MUTED)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5)
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
