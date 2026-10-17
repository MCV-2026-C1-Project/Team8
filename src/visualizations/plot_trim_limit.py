"""Which trim limit and mode to use: retrieval quality vs painting cut off.

Two ways to remove the background around the painting before the block grid:
- block: delete block rows/columns with less than `limit` foreground (trim_background_blocks)
- bbox:  crop to the pixel rows/columns with at least `limit` foreground (crop_to_foreground)
For each mode and limit, QSD2 query descriptors are computed on foreground pixels
and retrieval is scored with the exported best configuration, using predicted and
ground-truth masks. The ground-truth masks also give how much real painting is
cut off when the predicted masks drive the cropping. Results go to a CSV too.

Run: python src/visualizations/plot_trim_limit.py
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptor_cli import load_descriptor_document
from descriptors import compute_descriptor, list_images, load_image
from metrics import mapk
from remove_background import foreground_mask
from retrieval import BBDD_DIR, load_gt, load_or_compute, retrieve
from spatial import (BBOX_MIN_FRACTION, TRIM_MIN_FRACTION, crop_to_foreground, resolve_spatial,
                     trim_background_blocks)

ROOT = SRC.parent
COLORS = {"block": "#2a78d6", "bbox": "#eb6834"}
LABELS = {"block": "block trimming", "bbox": "bounding-box crop"}
CURRENT = {"block": TRIM_MIN_FRACTION, "bbox": BBOX_MIN_FRACTION}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"
LIMITS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def cropper(mode, limit, grid):
    """Function (image, mask) -> cropped (image, mask); limit 0 means no cropping."""
    if limit == 0:
        return lambda image, mask: (image, mask)
    if mode == "block":
        return lambda image, mask: trim_background_blocks(image, mask, grid, min_fraction=limit / 100)
    return lambda image, mask: crop_to_foreground(image, mask, min_fraction=limit / 100)


def evaluate(crop, images, masks, cfg, measure, db, gt):
    desc = []
    for image, mask in zip(images, masks):
        image, mask = crop(image, mask)
        desc.append(compute_descriptor(image, cfg["descriptor"], mask=mask, trim=None, **cfg["params"]))
    pred = retrieve(np.stack(desc).astype(np.float32), db[1], db[0], measure, 10)
    return mapk(gt, pred, 1), mapk(gt, pred, 5)


def painting_lost(crop, images, pred_masks, gt_masks):
    """Fraction of ground-truth painting removed when cropping with the predicted masks."""
    lost = []
    for image, pred, gt in zip(images, pred_masks, gt_masks):
        cropped, _ = crop(np.dstack([image, gt]), pred)
        lost.append(1 - np.count_nonzero(cropped[..., 3]) / max(np.count_nonzero(gt), 1))
    return np.array(lost)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "week2" / "extended_best.json")
    ap.add_argument("--query-dir", type=Path, default=ROOT / "data" / "qsd2_w2")
    ap.add_argument("--limits", type=float, nargs="+", default=LIMITS, help="Limits in %%")
    ap.add_argument("--output", type=Path, default=ROOT / "plots" / "trim_limit_analysis.png")
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "trim_limit_sweep.csv")
    args = ap.parse_args()

    cfg, measure = load_descriptor_document(args.config)
    grid = max(resolve_spatial(cfg["params"]["spatial"])["grids"])
    db = load_or_compute("bbdd", BBDD_DIR, cfg["descriptor"], cfg["params"])
    gt = load_gt(args.query_dir)
    paths, _ = list_images(args.query_dir)
    images = [load_image(p) for p in paths]
    pred_masks = [foreground_mask(image) for image in images]
    gt_masks = [cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_GRAYSCALE) for p in paths]

    rows = []
    for mode in ("block", "bbox"):
        for limit in args.limits:
            crop = cropper(mode, limit, grid)
            p1, p5 = evaluate(crop, images, pred_masks, cfg, measure, db, gt)
            g1, g5 = evaluate(crop, images, gt_masks, cfg, measure, db, gt)
            lost = painting_lost(crop, images, pred_masks, gt_masks)
            rows.append(dict(mode=mode, limit=limit, pred_map1=p1, pred_map5=p5, gt_map1=g1, gt_map5=g5,
                             lost_mean=100 * lost.mean(), lost_max=100 * lost.max(),
                             images_lost_over_2pct=int((lost > 0.02).sum())))
            print(f"{mode:<5} {limit:>4g}%  pred {p1:.3f}/{p5:.3f}  gt {g1:.3f}/{g5:.3f}  "
                  f"lost mean {100 * lost.mean():.2f}% max {100 * lost.max():.2f}%")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    panels = [("pred_map1", "mAP@1, predicted masks"), ("pred_map5", "mAP@5, predicted masks"),
              ("lost_mean", "Painting cut off: mean over images (%)"),
              ("gt_map1", "mAP@1, ground-truth masks"), ("gt_map5", "mAP@5, ground-truth masks"),
              ("lost_max", "Painting cut off: worst image (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, title) in zip(axes.flat, panels):
        for mode in ("block", "bbox"):
            data = [r for r in rows if r["mode"] == mode]
            x = np.array([r["limit"] for r in data])
            y = np.array([r[key] for r in data])
            ax.plot(x, y, color=COLORS[mode], lw=2, marker="o", ms=5, label=LABELS[mode])
            ax.axvline(100 * CURRENT[mode], color=COLORS[mode], lw=1, ls=":")
            if key.endswith(("map1", "map5")):
                best = int(np.argmax(y))
                ax.annotate(f"{y[best]:.3f}", (x[best], y[best]), textcoords="offset points",
                            xytext=(0, 8 if mode == "bbox" else -14), ha="center", fontsize=8, color=INK)
        if key.endswith(("map1", "map5")):
            ax.set_ylim(0.4, 0.9)
        ax.set_title(title, fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("limit (% foreground in row/column)", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"{args.query_dir.name}: block trimming vs bounding-box crop "
                 "(limit 0 = no cropping; dotted = current default)", fontsize=12, color=INK)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
