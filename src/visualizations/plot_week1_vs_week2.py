"""QSD1 mAP@1 and mAP@5 of the Week 1 best descriptor, the Week 1 descriptor with photometric
normalisation and the Week 2 best. Retrieval is rerun from the exported best configurations.

Run: python src/visualizations/plot_week1_vs_week2.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptor_cli import load_descriptor_document
from metrics import mapk
from retrieval import DATA_DIR, load_gt, retrieve_config

ROOT = SRC.parent
SYSTEMS = [
    ("W1 best: HSV 16³, global", "results/week1/1d_none_best.json", "#2a78d6"),
    ("W1 + photometric: Lab 16-64-64, global", "results/week1/1d_photometric_best.json", "#a9a8a3"),
    ("W2 best: YCbCr 64³, norm, 4×4", "results/week2/qsd1_best_with_norm.json", "#eb6834"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd1_w1")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "week1_vs_week2_qsd1.png")
    args = ap.parse_args()
    gt = load_gt(args.query_dir)

    x = np.arange(2)
    width = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for k, (label, path, color) in enumerate(SYSTEMS):
        cfg, measure = load_descriptor_document(ROOT / path)
        _, pred = retrieve_config(cfg, measure, query_dir=args.query_dir, k=10)
        values = [mapk(gt, pred, 1), mapk(gt, pred, 5)]
        print(f"{label}: mAP@1 {values[0]:.3f}  mAP@5 {values[1]:.3f}")
        bars = ax.bar(x + (k - 1) * width, values, width * 0.95, color=color, label=label)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, ["mAP@1", "mAP@5"], fontsize=10)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
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
