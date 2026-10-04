"""Show which pixels contribute to the Hue histogram and the resulting HSV descriptor.

hsv_hist only counts a pixel in the H histogram if S > s_min and V > v_min
(hue is unreliable for dark / grey pixels). S and V use every pixel.

Row 1: original image | pixels counted in H (excluded ones in black).
Row 2: the H, S, V histograms, each normalised to sum 1 (no valid-Hue weighting).
Row 3: the concatenated descriptor.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptors import HSV_DEFAULTS, hsv_hist, load_image
from plot_hsv_histograms import CHANNELS, COLORS, find_image, hue_colors

ROOT = SRC.parent


def plot(img_bgr, s_min, v_min, bins, title):
    h, s, v = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    mask = (s > s_min) & (v > v_min)
    n_valid, n_total = int(mask.sum()), mask.size
    desc = hsv_hist(img_bgr, bins=bins, s_min=s_min, v_min=v_min,
                    hue_valid_weight=False, hue_smoothing=False)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shown = rgb.copy()
    shown[~mask] = 0  # excluded pixels shown in black

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 1, 1])

    ax = fig.add_subplot(gs[0, :3])
    ax.imshow(rgb)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3:])
    ax.imshow(shown)
    ax.set_title(f"Pixels counted in H (S > {s_min:g}, V > {v_min:g})\n"
                 f"{n_valid:,} / {n_total:,} = {100 * n_valid / n_total:.1f}%",
                 fontweight="bold")
    ax.axis("off")

    starts = np.cumsum([0, *bins])
    colors = [hue_colors(bins[0]), COLORS["S"], COLORS["V"]]
    for i, ((key, name, rng), color) in enumerate(zip(CHANNELS, colors)):
        x = np.linspace(rng[0], rng[1], bins[i], endpoint=False)
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        ax.bar(x, desc[starts[i]:starts[i + 1]], width=(rng[1] - rng[0]) / bins[i],
               color=color, align="edge", edgecolor="k", linewidth=0.3)
        ax.set_title(f"{key}: {name} ({bins[i]} bins)", fontweight="bold")
        ax.set_xlim(*rng)
        ax.set_xlabel(f"{key} value")
        ax.set_ylabel("Normalised count")

    ax = fig.add_subplot(gs[2, :])
    for i, ((key, name, _), color) in enumerate(zip(CHANNELS, colors)):
        x = np.arange(starts[i], starts[i + 1])
        ax.bar(x, desc[starts[i]:starts[i + 1]], width=1.0, color=color, label=f"{key}: {name}",
               edgecolor="k", linewidth=0.3)
        if i:
            ax.axvline(starts[i] - 0.5, color="#B22222", linestyle="--", linewidth=1)
    ax.set_title(f"Concatenated descriptor (D = {len(desc)}, each channel sums to 1)",
                 fontweight="bold")
    ax.set_xlabel("Descriptor index")
    ax.set_ylabel("Normalised count")
    ax.set_xlim(-0.5, len(desc) - 0.5)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def parse_bins(text):
    bins = tuple(int(b) for b in text.split("-"))
    if len(bins) != 3 or min(bins) <= 0:
        raise argparse.ArgumentTypeError("bins must be three positive integers, e.g. 16-16-16")
    return bins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", type=int, help="Image ID (e.g. 7 for bbdd_00007.jpg)")
    ap.add_argument("--set", default="bbdd", help="'bbdd' (default) or a query set, e.g. qsd1")
    ap.add_argument("--bins", type=parse_bins, default=HSV_DEFAULTS["bins"],
                    help="H-S-V bins, e.g. 16-16-16 (default: %(default)s)")
    ap.add_argument("--s-min", type=float, default=HSV_DEFAULTS["s_min"])
    ap.add_argument("--v-min", type=float, default=HSV_DEFAULTS["v_min"])
    ap.add_argument("--out", default=str(ROOT / "plots"), help="Output folder")
    ap.add_argument("--show", action="store_true", help="Also open the figure in a window")
    args = ap.parse_args()

    path = find_image(args.id, args.set)
    fig = plot(load_image(path), args.s_min, args.v_min, args.bins, path.name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, args.bins))
    out_path = out / f"hue_mask_{tag}_{args.set}_{args.id:05d}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
