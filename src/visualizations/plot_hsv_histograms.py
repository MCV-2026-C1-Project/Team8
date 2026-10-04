"""Plot the H, S, V histograms of one image and their concatenated descriptor.

Uses the hsv_baseline baseline (180-256-256, one bin per value, no S/V mask on Hue),
i.e. exactly the vector used for retrieval.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptors import channel_sizes, compute_descriptor, list_images, load_image
from retrieval import BBDD_DIR, query_folder

ROOT = SRC.parent
METHOD = "hsv_baseline"
CHANNELS = [("H", "Hue", (0, 180)), ("S", "Saturation", (0, 256)), ("V", "Value", (0, 256))]
COLORS = {"S": "#377EB8", "V": "#4D4D4D"}


def find_image(image_id, image_set):
    folder = BBDD_DIR if image_set == "bbdd" else query_folder(image_set)
    paths, ids = list_images(folder)
    if image_id not in ids:
        raise SystemExit(f"ID {image_id} not found in {folder}")
    return paths[ids.index(image_id)]


def hue_colors(n):
    """RGB colour of each Hue bin centre (full S and V), for colouring H bars."""
    h = ((np.arange(n) + 0.5) * 180 / n).astype(np.uint8)
    hsv = np.stack([h, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], axis=1)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0] / 255


def plot(img_bgr, desc, sizes, title):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontweight="bold")
    ax.axis("off")

    starts = np.cumsum([0, *sizes])
    colors = [hue_colors(sizes[0]), COLORS["S"], COLORS["V"]]
    for i, ((key, name, rng), color) in enumerate(zip(CHANNELS, colors)):
        hist = desc[starts[i]:starts[i + 1]]
        x = np.linspace(rng[0], rng[1], sizes[i], endpoint=False)
        ax = fig.add_subplot(gs[0, i + 1])
        ax.bar(x, hist, width=(rng[1] - rng[0]) / sizes[i], color=color, align="edge")
        ax.set_title(f"{key}: {name} ({sizes[i]} bins)", fontweight="bold")
        ax.set_xlim(*rng)
        ax.set_xlabel(f"{key} value")
        ax.set_ylabel("Normalised count")

    ax = fig.add_subplot(gs[1, :])
    for i, ((key, name, _), color) in enumerate(zip(CHANNELS, colors)):
        x = np.arange(starts[i], starts[i + 1])
        ax.bar(x, desc[starts[i]:starts[i + 1]], width=1.0, color=color, label=f"{key}: {name}")
        if i:
            ax.axvline(starts[i] - 0.5, color="#B22222", linestyle="--", linewidth=1)
    ax.set_title(f"Concatenated descriptor (D = {len(desc)}, each channel sums to 1)", fontweight="bold")
    ax.set_xlabel("Descriptor index")
    ax.set_ylabel("Normalised count")
    ax.set_xlim(-0.5, len(desc) - 0.5)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", type=int, help="Image ID (e.g. 7 for bbdd_00007.jpg)")
    ap.add_argument("--set", default="bbdd", help="'bbdd' (default) or a query set, e.g. qsd1")
    ap.add_argument("--out", default=str(ROOT / "plots"), help="Output folder")
    ap.add_argument("--show", action="store_true", help="Also open the figure in a window")
    args = ap.parse_args()

    path = find_image(args.id, args.set)
    img = load_image(path)
    desc = compute_descriptor(img, METHOD)
    fig = plot(img, desc, channel_sizes(METHOD), path.name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{METHOD}_{args.set}_{args.id:05d}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
