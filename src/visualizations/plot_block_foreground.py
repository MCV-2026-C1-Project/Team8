"""Percentage of foreground pixels per block of the 5x5 grid, without trimming.

Figure 1 (block_foreground_examples.png): example queries with the background
dimmed, the grid drawn and each block's foreground %. The margins give the
foreground % of each block row/column; those below the trim limit would
be removed by trim_background_blocks.
Figure 2 (block_foreground_summary.png): mean foreground % per block over the
whole query set, and the distribution of block-row/column foreground %.

Run: python src/visualizations/plot_block_foreground.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptors import list_images, load_image
from remove_background import foreground_mask
from spatial import TRIM_MIN_FRACTION

ROOT = SRC.parent
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, FLAG = "#1f1f1e", "#6b6a66", "#c4352b"
TRIM_LIMIT = TRIM_MIN_FRACTION


def block_fractions(mask, grid):
    """Foreground fraction of every block, and of every block row and column."""
    fg = mask > 0
    ys = np.linspace(0, fg.shape[0], grid + 1, dtype=int)
    xs = np.linspace(0, fg.shape[1], grid + 1, dtype=int)
    blocks = np.array([[fg[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(grid)]
                       for i in range(grid)])
    rows = np.array([fg[ys[i]:ys[i + 1]].mean() for i in range(grid)])
    cols = np.array([fg[:, xs[j]:xs[j + 1]].mean() for j in range(grid)])
    return blocks, rows, cols, ys, xs


def get_mask(path, image, source):
    if source == "gt":
        return cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    return foreground_mask(image)


def plot_examples(items, grid, output):
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, image, mask) in zip(axes.flat, items):
        blocks, row_f, col_f, ys, xs = block_fractions(mask, grid)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        shown = np.where(mask[..., None] > 0, rgb, rgb * 0.25 + 255 * 0.05).astype(np.uint8)
        h, w = mask.shape
        ax.imshow(shown, extent=(0, w, h, 0))
        for y in ys:
            ax.axhline(y, color="white", lw=1)
        for x in xs:
            ax.axvline(x, color="white", lw=1)
        for i in range(grid):
            for j in range(grid):
                ax.text((xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2, f"{100 * blocks[i, j]:.0f}",
                        ha="center", va="center", fontsize=8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        # Row % on the right, column % below; red = would be trimmed (< 1 %).
        for i, value in enumerate(row_f):
            ax.text(w * 1.02, (ys[i] + ys[i + 1]) / 2, f"{100 * value:.1f}", ha="left", va="center",
                    fontsize=8, color=FLAG if value < TRIM_LIMIT else MUTED, clip_on=False)
        for j, value in enumerate(col_f):
            ax.text((xs[j] + xs[j + 1]) / 2, h * 1.04, f"{100 * value:.1f}", ha="center", va="top",
                    fontsize=8, color=FLAG if value < TRIM_LIMIT else MUTED, clip_on=False)
        removable = int((row_f < TRIM_LIMIT).sum() + (col_f < TRIM_LIMIT).sum())
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f"{path.name}  ({100 * (mask > 0).mean():.0f}% foreground, "
                     f"{removable} row/col < {100 * TRIM_LIMIT:g}%)", fontsize=10, color=INK)
    fig.suptitle(f"Foreground % per block ({grid}x{grid}, no trimming). "
                 f"Margins: % per block row/column; red = below the {100 * TRIM_LIMIT:g}% trim limit",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_blocks, all_lines, grid, name, output):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    mean = 100 * np.mean(all_blocks, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1, 1.3]))
    im = ax1.imshow(mean, cmap=cmap, vmin=0, vmax=100)
    for i in range(grid):
        for j in range(grid):
            ax1.text(j, i, f"{mean[i, j]:.0f}", ha="center", va="center", fontsize=10,
                     color="white" if mean[i, j] > 55 else INK)
    ax1.set_xticks(range(grid), [f"col {j + 1}" for j in range(grid)], fontsize=8, color=MUTED)
    ax1.set_yticks(range(grid), [f"row {i + 1}" for i in range(grid)], fontsize=8, color=MUTED)
    ax1.tick_params(length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_title(f"Mean foreground % per block ({len(all_blocks)} images)", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04).set_label("% foreground", color=MUTED)

    lines = 100 * np.asarray(all_lines)
    ax2.hist(lines, bins=np.arange(0, 102, 2.5), color=BLUES[3], edgecolor="white", linewidth=1)
    ax2.axvline(100 * TRIM_LIMIT, color=FLAG, lw=1.5)
    below = int((lines < 100 * TRIM_LIMIT).sum())
    ax2.text(100 * TRIM_LIMIT + 1.5, ax2.get_ylim()[1] * 0.92,
             f"{100 * TRIM_LIMIT:g}% trim limit: {below} of {len(lines)} rows/cols below", color=FLAG, fontsize=9)
    ax2.set_xlabel("% foreground in block row / column", color=MUTED)
    ax2.set_ylabel("count", color=MUTED)
    ax2.set_title("Foreground % of every block row and column", fontsize=11, color=INK)
    ax2.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    fig.suptitle(f"{name}: foreground per block, {grid}x{grid} grid without trimming",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=ROOT / "data" / "qsd2_w2")
    ap.add_argument("--mask-source", choices=("pred", "gt"), default="pred")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--examples", nargs="+", default=["00000", "00004", "00007", "00014", "00026", "00029"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "plots")
    args = ap.parse_args()

    paths, _ = list_images(args.query_dir)
    all_blocks, all_lines, examples = [], [], []
    for path in paths:
        image = load_image(path)
        mask = get_mask(path, image, args.mask_source)
        blocks, rows, cols, _, _ = block_fractions(mask, args.grid)
        all_blocks.append(blocks)
        all_lines.extend(rows)
        all_lines.extend(cols)
        if path.stem in args.examples:
            examples.append((path, image, mask))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.query_dir.name}_{args.mask_source}"
    plot_examples(examples, args.grid, args.out_dir / f"block_foreground_examples_{suffix}.png")
    plot_summary(all_blocks, all_lines, args.grid, f"{args.query_dir.name} ({args.mask_source} masks)",
                 args.out_dir / f"block_foreground_summary_{suffix}.png")
    print(f"Saved plots to {args.out_dir} ({suffix})")


if __name__ == "__main__":
    main()
