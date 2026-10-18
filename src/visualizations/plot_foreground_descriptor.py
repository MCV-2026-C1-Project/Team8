"""How a QSD2 query descriptor uses the mask: query, predicted mask, and the block grid laid on
the (block-trimmed) image where only painting pixels are counted (wall greyed out).
The grid size comes from the QSD2 best configuration (results/week2/qsd2_best.json).

Run: python src/visualizations/plot_foreground_descriptor.py
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
from descriptors import load_image
from remove_background import foreground_mask
from retrieval import DATA_DIR
from spatial import trim_background_blocks

ROOT = SRC.parent
INK = "#1f1f1e"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "results" / "week2" / "qsd2_best.json")
    ap.add_argument("--image", type=Path, default=DATA_DIR / "qsd2_w2" / "00013.jpg")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "foreground_descriptor.png")
    args = ap.parse_args()
    cfg, _ = load_descriptor_document(args.config)
    grid = max(cfg["params"]["spatial"]["grids"])
    image = load_image(args.image)
    mask = foreground_mask(image)
    trimmed, tmask = trim_background_blocks(image, mask, grid)

    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.15))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("query", fontsize=9, color=INK)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("predicted mask", fontsize=9, color=INK)
    shown = cv2.cvtColor(trimmed, cv2.COLOR_BGR2RGB).astype(np.float32)
    shown[tmask == 0] = shown[tmask == 0] * 0.15 + 220 * 0.85
    axes[2].imshow(shown.astype(np.uint8))
    h, w = tmask.shape
    for v in np.linspace(0, h, grid + 1)[1:-1]:
        axes[2].axhline(v, color="white", lw=1.5)
    for v in np.linspace(0, w, grid + 1)[1:-1]:
        axes[2].axvline(v, color="white", lw=1.5)
    axes[2].set_title(f"{grid}×{grid} blocks, painting\npixels only", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
