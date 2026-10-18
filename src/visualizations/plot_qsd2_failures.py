"""Every QSD2 query the QSD2 best descriptor (with predicted masks) gets wrong at rank 1:
the query with its mask outline, the wrong first result, and the correct painting's rank.

Run: python src/visualizations/plot_qsd2_failures.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptor_cli import load_descriptor_document
from descriptors import load_image
from retrieval import BBDD_DIR, DATA_DIR, load_gt, query_masks, retrieve_config

ROOT = SRC.parent
INK, GOOD, BAD, AQUA = "#1f1f1e", "#008300", "#e34948", "#1baf7a"


def show(ax, image_bgr, title, border=None):
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        if border is not None:
            spine.set_edgecolor(border)
            spine.set_linewidth(4)
    ax.set_title(title, fontsize=8.5, color=border or INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "results" / "week2" / "qsd2_best.json")
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd2_w2")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "qsd2_failures.png")
    args = ap.parse_args()
    gt = load_gt(args.query_dir)
    paths, ids, images, masks = query_masks(args.query_dir, "pred")
    cfg, measure = load_descriptor_document(args.config)
    _, pred = retrieve_config(cfg, measure, query_dir=args.query_dir, k=10, masks=(ids, images, masks))
    wrong = [q for q in range(len(gt)) if pred[q][0] not in gt[q]]
    if not wrong:
        print("No rank-1 errors; nothing to plot.")
        return

    fig, axes = plt.subplots(len(wrong), 3, figsize=(3.6, 1.45 * len(wrong)), squeeze=False)
    for row, q in zip(axes, wrong):
        show(row[0], images[q], f"query {paths[q].stem}")
        row[0].contour(masks[q] > 0, levels=[0.5], colors=[AQUA], linewidths=1.2)
        show(row[1], load_image(BBDD_DIR / f"bbdd_{pred[q][0]:05d}.jpg"), "#1 (wrong)", border=BAD)
        rank = next((r for r, i in enumerate(pred[q], 1) if i in gt[q]), None)
        show(row[2], load_image(BBDD_DIR / f"bbdd_{gt[q][0]:05d}.jpg"),
             f"correct: #{rank}" if rank else "correct: > 10", border=GOOD)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrong at rank 1: {[paths[q].stem for q in wrong]}. Saved {args.output}")


if __name__ == "__main__":
    main()
