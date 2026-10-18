"""A QSD1 query the Week 1 best misses in its top 10 and the Week 2 best gets at rank 1:
the query, then the top 3 of each descriptor (green = correct painting).

Run: python src/visualizations/plot_week1_vs_week2_example.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptor_cli import load_descriptor_document
from descriptors import list_images, load_image
from retrieval import BBDD_DIR, DATA_DIR, load_gt, retrieve_config

ROOT = SRC.parent
W1 = ROOT / "results" / "week1" / "1d_none_best.json"
W2 = ROOT / "results" / "week2" / "qsd1_best_with_norm.json"
INK, GOOD, BAD, BLUE, ORANGE = "#1f1f1e", "#008300", "#e34948", "#2a78d6", "#eb6834"
TOP = 3


def predictions(path, query_dir):
    cfg, measure = load_descriptor_document(path)
    return retrieve_config(cfg, measure, query_dir=query_dir, k=10)[1]


def show(ax, image_bgr, title, border=None):
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        if border is not None:
            spine.set_edgecolor(border)
            spine.set_linewidth(4)
    ax.set_title(title, fontsize=9.5, color=border or INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd1_w1")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "week1_vs_week2_example.png")
    args = ap.parse_args()
    gt = load_gt(args.query_dir)
    w1, w2 = predictions(W1, args.query_dir), predictions(W2, args.query_dir)
    paths, _ = list_images(args.query_dir)
    q = next(i for i in range(len(gt)) if not set(w1[i]) & set(gt[i]) and w2[i][0] in gt[i])

    fig = plt.figure(figsize=(9.2, 2.35))
    cell = 1 / (2 * TOP + 1.6)
    show(fig.add_axes([0.0, 0.03, cell * 1.05, 0.8]), load_image(paths[q]), f"query {paths[q].stem}")
    for g, (pred, name, color) in enumerate(((w1, "W1 best", BLUE), (w2, "W2 best", ORANGE))):
        x0 = cell * (1.6 + g * TOP)
        fig.text(x0 + cell * TOP / 2, 0.95, name, ha="center", va="top", fontsize=10.5, fontweight="bold",
                 color=color)
        for r in range(TOP):
            ok = pred[q][r] in gt[q]
            ax = fig.add_axes([x0 + r * cell + 0.006, 0.03, cell - 0.012, 0.72])
            show(ax, load_image(BBDD_DIR / f"bbdd_{pred[q][r]:05d}.jpg"), f"#{r + 1}" + (" ✓" if ok else ""),
                 border=GOOD if ok else BAD)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Query {paths[q].stem}. Saved {args.output}")


if __name__ == "__main__":
    main()
