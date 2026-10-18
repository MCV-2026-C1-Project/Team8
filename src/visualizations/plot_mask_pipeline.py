"""Steps of the background removal (foreground_mask, method="color_distribution") on one QSD2 query:
border sample, distance to the wall colour model, colour mask, final mask after filling and
row/column cleanup. F1 is against the ground-truth mask.

Run: python src/visualizations/plot_mask_pipeline.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from evaluate_background import evaluate_mask, load_image
from remove_background import _color_distribution_mask, foreground_mask
from retrieval import DATA_DIR

ROOT = SRC.parent
INK, GREEN = "#1f1f1e", "#1baf7a"
# Defaults of foreground_mask(method="color_distribution").
BORDER_RATIO, SAMPLE_STEP, COLOR_SIGMA = 0.04, 4, 5.5


def wall_distance(image):
    """Border mask and per-pixel Mahalanobis distance to the wall colour, as in
    remove_background._distribution_model (inf for pixels darker than the wall allows)."""
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
    errors = target - design @ coef
    center = np.median(errors, axis=0)
    cov = ((errors - center) * weights[:, None]).T @ (errors - center) / max(weights.sum(), 1.0) + np.eye(2)
    L = np.clip(lab[..., 0] / 255.0, lightness.min(), lightness.max())
    pred = coef[0] + L[..., None] * coef[1] + L[..., None] ** 2 * coef[2]
    delta = lab[..., 1:] - pred - center
    dist = np.sqrt(np.maximum(0, np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov), delta)))
    dist[lab[..., 0] < max(0.0, np.percentile(colors[:, 0], 2) - 30.0)] = np.inf
    return border, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=DATA_DIR / "qsd2_w2" / "00020.jpg")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "mask_pipeline.png")
    args = ap.parse_args()
    image = load_image(args.image)
    gt = cv2.imread(str(args.image.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    border, dist = wall_distance(image)
    raw = _color_distribution_mask(image, BORDER_RATIO, SAMPLE_STEP, COLOR_SIGMA)
    final = foreground_mask(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.65))
    shown = rgb.astype(np.float32)
    shown[~border] *= 0.3
    axes[0].imshow(shown.astype(np.uint8))
    axes[0].set_title("1. Sample the border (4 %)", color=INK, fontsize=10.5)
    im = axes[1].imshow(np.minimum(dist, 3 * COLOR_SIGMA), cmap="magma")
    axes[1].set_title("2. Distance to wall colour", color=INK, fontsize=10.5)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03, ticks=[0, COLOR_SIGMA, 3 * COLOR_SIGMA])
    cbar.ax.set_yticklabels(["0", f"{COLOR_SIGMA:g}", f"≥{3 * COLOR_SIGMA:g}"], fontsize=8)
    cbar.ax.axhline(COLOR_SIGMA, color="white", lw=1.5)
    axes[2].imshow(raw, cmap="gray")
    axes[2].set_title(f"3. Colour mask (white = painting)\nF1 {evaluate_mask(raw, gt)['f1']:.2f}",
                      color=INK, fontsize=10.5)
    axes[3].imshow(rgb)
    axes[3].contour(final > 0, levels=[0.5], colors=[GREEN], linewidths=2)
    axes[3].set_title(f"4. Fill, clean rows/cols\nF1 {evaluate_mask(final, gt)['f1']:.2f}", color=INK, fontsize=10.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(w_pad=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
