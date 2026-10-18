"""One query split into G x G blocks and the concatenated block descriptor it produces.

Uses the QSD1 best configuration (results/week2/qsd1_best_with_norm.json: YCbCr 64-64-64,
photometric normalisation, 4x4 grid).

Run: python src/visualizations/plot_block_grid_example.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptor_cli import load_descriptor_document
from descriptors import compute_descriptor, load_image
from retrieval import DATA_DIR

ROOT = SRC.parent
INK, MUTED, RED, BLUE = "#1f1f1e", "#6b6a66", "#e34948", "#2a78d6"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "results" / "week2" / "qsd1_best_with_norm.json")
    ap.add_argument("--image", type=Path, default=DATA_DIR / "qsd1_w1" / "00003.jpg")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "block_grid_example.png")
    args = ap.parse_args()
    cfg, _ = load_descriptor_document(args.config)
    image = load_image(args.image)
    desc = compute_descriptor(image, cfg["descriptor"], **cfg["params"])
    grid = cfg["params"]["spatial"]["grids"][0]
    per_block = len(desc) // (grid * grid)
    bins = cfg["params"]["bins"]

    fig = plt.figure(figsize=(9.2, 2.5))
    ax0 = fig.add_axes([0.0, 0.02, 0.3, 0.96])
    ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax0.set_axis_off()
    h, w = image.shape[:2]
    ys, xs = np.linspace(0, h, grid + 1), np.linspace(0, w, grid + 1)
    for y in ys:
        ax0.axhline(y, color="white", lw=1.2)
    for x in xs:
        ax0.axvline(x, color="white", lw=1.2)
    for i in range(grid):
        for j in range(grid):
            ax0.text((xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2, str(i * grid + j + 1), ha="center",
                     va="center", fontsize=7.5, color=INK,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

    ax1 = fig.add_axes([0.34, 0.2, 0.65, 0.62])
    for b in range(1, grid * grid, 2):
        ax1.axvspan(b * per_block, (b + 1) * per_block, color="#f3f2ef", lw=0)
    start = 0
    for length, color, label in ((bins[0], INK, f"Y: {bins[0]} bins"), (bins[1], RED, f"Cr: {bins[1]} bins"),
                                 (bins[2], BLUE, f"Cb: {bins[2]} bins")):
        idx = np.concatenate([np.arange(b * per_block + start, b * per_block + start + length)
                              for b in range(grid * grid)])
        ax1.vlines(idx, 0, desc[idx], color=color, lw=0.6, label=label)
        start += length
    ax1.set_xlim(0, len(desc))
    ax1.set_yticks([])
    ticks = range(0, grid * grid, 3)
    ax1.set_xticks([(b + 0.5) * per_block for b in ticks], [f"block {b + 1}" for b in ticks], fontsize=8.5)
    ax1.set_title(f"{grid}×{grid} blocks × {per_block} values = {len(desc)}-value descriptor", color=INK,
                  loc="left", pad=18, fontsize=11)
    ax1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=3, fontsize=8.5)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(colors=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
