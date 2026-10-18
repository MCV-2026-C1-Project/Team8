"""QSD2 mAP@1 and mAP@5 with and without background removal: the whole photo, the QSD1 best
descriptor on our masks, and the QSD2 best descriptor on our masks and on the GT masks.

Run: python src/visualizations/plot_qsd2_retrieval.py
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
from retrieval import DATA_DIR, load_gt, query_masks, retrieve_config

ROOT = SRC.parent
QSD2_BEST = ROOT / "results" / "week2" / "qsd2_best.json"
QSD1_BEST = ROOT / "results" / "week2" / "qsd1_best_with_norm.json"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a66", "#e5e4e0"


def run(cfg_path, query_dir, masks=None):
    """masks: query_masks output, or None to describe the whole photo."""
    cfg, measure = load_descriptor_document(cfg_path)
    return retrieve_config(cfg, measure, query_dir=query_dir, k=10,
                           masks=None if masks is None else masks[1:])[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd2_w2")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "qsd2_retrieval.png")
    args = ap.parse_args()
    gt = load_gt(args.query_dir)
    pred_masks, gt_masks = query_masks(args.query_dir, "pred"), query_masks(args.query_dir, "gt")
    results = [
        ("whole photo\n(no removal)", run(QSD2_BEST, args.query_dir)),
        ("QSD1 best\n+ our masks", run(QSD1_BEST, args.query_dir, pred_masks)),
        ("QSD2 best\n+ our masks", run(QSD2_BEST, args.query_dir, pred_masks)),
        ("QSD2 best\n+ GT masks", run(QSD2_BEST, args.query_dir, gt_masks)),
    ]

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for offset, k, color in ((-0.2, 1, BLUE), (0.2, 5, ORANGE)):
        values = [mapk(gt, pred, k) for _, pred in results]
        bars = ax.bar(x + offset, values, 0.38, color=color, label=f"mAP@{k}")
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8, color=INK)
    for name, pred in results:
        print(f"{name.replace(chr(10), ' '):28s} mAP@1 {mapk(gt, pred, 1):.3f}  mAP@5 {mapk(gt, pred, 5):.3f}")
    ax.set_xticks(x, [name for name, _ in results], fontsize=8.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
