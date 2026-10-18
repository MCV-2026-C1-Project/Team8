"""Predicted background masks on harder QSD2 queries, compared with the ground truth:
predicted painting outline, wall kept as painting (red) and painting removed (blue).

Run: python src/visualizations/plot_mask_examples.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from evaluate_background import evaluate_mask, load_image
from remove_background import foreground_mask
from retrieval import DATA_DIR

ROOT = SRC.parent
INK, GREEN, RED, BLUE = "#1f1f1e", "#1baf7a", "#e34948", "#2a78d6"
PICKS = [("00007", "labels left out"), ("00010", "dark painting"),
         ("00017", "gaps between panels stay wall"), ("00014", "fails: frame shadow")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd2_w2")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "mask_examples.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(PICKS), figsize=(9.2, 2.55))
    for ax, (stem, caption) in zip(axes, PICKS):
        image = load_image(args.query_dir / f"{stem}.jpg")
        gt = cv2.imread(str(args.query_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        pred = foreground_mask(image)
        p, g = pred > 0, gt > 0
        shown = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shown[p & ~g] = (227, 73, 72)
        shown[~p & g] = (42, 120, 214)
        ax.imshow(shown)
        ax.contour(p, levels=[0.5], colors=[GREEN], linewidths=1.6)
        ax.set_title(f"{caption}\n{stem} · F1 {evaluate_mask(pred, gt)['f1']:.2f}", color=INK, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=GREEN, label="predicted painting outline"),
                        Patch(color=RED, label="wall kept as painting"),
                        Patch(color=BLUE, label="painting removed")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
