"""Wall colour model of the background removal on one QSD2 query: the sampled border pixels'
a and b against lightness L, with the robust quadratic fit used by
remove_background._distribution_model (method="color_distribution").

Run: python src/visualizations/plot_wall_model.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from evaluate_background import load_image
from retrieval import DATA_DIR

ROOT = SRC.parent
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#6b6a66", "#e5e4e0"
# Defaults of foreground_mask(method="color_distribution").
BORDER_RATIO, SAMPLE_STEP = 0.04, 4


def fit_wall(image):
    """Sampled border Lab pixels and the robust fit of (a, b) as a quadratic in L."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bh, bw = max(2, int(h * BORDER_RATIO)), max(2, int(w * BORDER_RATIO))
    border = np.zeros((h, w), bool)
    border[:bh], border[-bh:], border[:, :bw], border[:, -bw:] = True, True, True, True
    sampled = np.zeros((h, w), bool)
    sampled[::SAMPLE_STEP, ::SAMPLE_STEP] = True
    colors = lab.reshape(-1, 3)[np.flatnonzero(sampled & border)]
    lightness = colors[:, 0] / 255.0
    design = np.column_stack([np.ones_like(lightness), lightness, lightness ** 2])
    target, weights = colors[:, 1:], np.ones(len(colors))
    for _ in range(8):
        sw = np.sqrt(weights)[:, None]
        coef = np.linalg.lstsq(design * sw, target * sw, rcond=None)[0]
        res = np.linalg.norm(target - design @ coef, axis=1)
        med = np.median(res)
        scale = max(0.5, 1.4826 * np.median(np.abs(res - med)))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(res, 1e-6))
    center = np.median(target - design @ coef, axis=0)
    return colors, coef, center


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DATA_DIR / "qsd2_w2" / "00020.jpg")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "wall_model.png")
    args = ap.parse_args()
    colors, coef, center = fit_wall(load_image(args.image))
    L = colors[:, 0] / 255.0
    grid = np.linspace(L.min(), L.max(), 100)

    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    for k, (name, color) in enumerate((("a (green–red)", BLUE), ("b (blue–yellow)", ORANGE))):
        ax.scatter(colors[:, 0], colors[:, k + 1], s=3, color=color, alpha=0.25, lw=0)
        fit = coef[0, k] + grid * coef[1, k] + grid ** 2 * coef[2, k] + center[k]
        ax.plot(grid * 255, fit, color=color, lw=2, label=f"wall {name}")
    ax.set_xlabel("border pixel lightness L", color=MUTED, fontsize=10)
    ax.set_ylabel("chroma (OpenCV Lab)", color=MUTED, fontsize=10)
    lo, hi = np.percentile(colors[:, 1:], [1, 99])
    ax.set_ylim(lo - 6, hi + 10)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8, handlelength=1.2, columnspacing=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
