"""Week 3 Task 3: background removal, Week 2 method vs Week 3 multi-painting detection.

W2 = foreground_mask with its Week 2 defaults (one region, 50 % row/column cleanup).
W3 = detect_paintings with configs/week3/task3_selected.json (up to 2 instances).
Pixel precision / recall / F1 per image (evaluate_background.evaluate_mask), averaged,
plus painting-count accuracy, on QSD2-W3 and on QSD2-W2 (regression check).

Run: python src/visualizations/plot_week3_masks.py
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from descriptors import list_images, load_image
from evaluate_background import evaluate_mask
from painting_detection import detect_paintings, resolve_detection_config
from remove_background import foreground_mask
from retrieval import DATA_DIR

ROOT = SRC.parent
INK, MUTED = "#1f1f1e", "#6b6a66"
CONFIG = ROOT / "configs/week3/task3_selected.json"


def count_regions(mask, min_ratio=0.003):
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return int(sum(stats[i, cv2.CC_STAT_AREA] >= min_ratio * mask.size for i in range(1, n)))


def run(folder, config):
    paths, _ = list_images(folder)
    with open(folder / "gt_corresps.pkl", "rb") as stream:
        gt_counts = [len(x) if isinstance(x, list) else 1 for x in pickle.load(stream)]
    aug_path = folder / "augmentations.pkl"
    labels = pickle.load(open(aug_path, "rb")) if aug_path.exists() else ["None"] * len(paths)
    rows = []
    for path, expected, label in zip(paths, gt_counts, labels):
        image = load_image(path)
        gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        w2 = foreground_mask(image)
        w3, paintings = detect_paintings(image, config)
        rows.append(dict(image=path.stem, expected=expected, noisy="Impulse" in label,
                         w2=evaluate_mask(w2, gt), w3=evaluate_mask(w3, gt),
                         w2_count=count_regions(w2), w3_count=len(paintings)))
    return rows


def summarise(rows):
    out = {}
    subsets = dict(all=lambda r: True, two=lambda r: r["expected"] == 2, one=lambda r: r["expected"] == 1,
                   noisy=lambda r: r["noisy"])
    for method in ("w2", "w3"):
        for name, keep in subsets.items():
            chosen = [r for r in rows if keep(r)]
            if not chosen:
                continue
            out[f"{method}_{name}"] = dict(
                n=len(chosen), precision=float(np.mean([r[method]["precision"] for r in chosen])),
                recall=float(np.mean([r[method]["recall"] for r in chosen])),
                f1=float(np.mean([r[method]["f1"] for r in chosen])),
                count_ok=int(sum(r[f"{method}_count"] == r["expected"] for r in chosen)))
    return out


def overlay(image, mask, gt):
    """Prediction tinted orange, ground-truth outline green."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    tint = np.array([235, 104, 52], np.float32)
    m = mask > 0
    rgb[m] = 0.55 * rgb[m] + 0.45 * tint
    rgb[~m] *= 0.45
    contours, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = rgb.clip(0, 255).astype(np.uint8)
    cv2.drawContours(out, contours, -1, (27, 175, 122), max(3, image.shape[1] // 250))
    return out


def example(folder, rows, config, out):
    """The two-painting image where Week 2 does worst."""
    two = [r for r in rows if r["expected"] == 2]
    worst = min(two, key=lambda r: r["w2"]["f1"])
    path = folder / f"{worst['image']}.jpg"
    image = load_image(path)
    gt = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
    w2 = foreground_mask(image)
    w3, _ = detect_paintings(image, config)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
    for ax, mask, title, row in [(axes[0], w2, "Week 2 method", worst["w2"]), (axes[1], w3, "Week 3 method", worst["w3"])]:
        ax.imshow(overlay(image, mask, gt))
        ax.set_title(f"{title}: F1 {row['f1']:.2f}", fontsize=9.5, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return worst["image"]


def ablation(config, out):
    """F1 as each Week 3 change is added to the Week 2 method, on QSD2-W3 and QSD2-W2."""
    base = dict(min_row_foreground=0, min_col_foreground=0)
    strict = dict(base, color_sigma=3.0)
    dark = dict(strict, dark_mode="robust", dark_margin=40.0)
    steps = [("Week 2 method", lambda im: foreground_mask(im)),
             ("+ drop 50 % row/col cleanup", lambda im: foreground_mask(im, **base)),
             ("+ stricter wall colour (σ 5.5→3)", lambda im: foreground_mask(im, **strict)),
             ("+ robust dark-wall cutoff", lambda im: foreground_mask(im, **dark)),
             ("+ ≤2 instances, convex hull (W3)", lambda im: detect_paintings(im, config)[0])]
    scores = {}
    for name in ("qsd2_w3", "qsd2_w2"):
        paths, _ = list_images(DATA_DIR / name)
        data = [(load_image(p), cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)) for p in paths]
        scores[name] = [float(np.mean([evaluate_mask(fn(im), gt)["f1"] for im, gt in data])) for _, fn in steps]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    y = np.arange(len(steps))[::-1]
    for k, (name, color, label) in enumerate([("qsd2_w3", "#eb6834", "QSD2-W3 (this week)"),
                                              ("qsd2_w2", "#a9a8a3", "QSD2-W2 (last week)")]):
        bars = ax.barh(y + (0.5 - k) * 0.38, scores[name], 0.36, color=color, label=label)
        for bar, v in zip(bars, scores[name]):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [s for s, _ in steps], fontsize=8.5)
    ax.set_xlim(0.5, 1.08)
    ax.set_xlabel("mean pixel F1", fontsize=9, color=MUTED)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#eb6834", label="QSD2-W3 (this week)"),
                        Patch(color="#a9a8a3", label="QSD2-W2 (last week)")],
               frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    ax.grid(axis="x", color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {name: dict(zip([s for s, _ in steps], v)) for name, v in scores.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", type=Path, default=ROOT / "plots")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "week3" / "masks")
    args = ap.parse_args()
    args.plots.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    config = resolve_detection_config(json.loads(CONFIG.read_text(encoding="utf-8")))
    report = {}
    for name in ("qsd2_w3", "qsd2_w2"):
        rows = run(DATA_DIR / name, config)
        report[name] = summarise(rows)
        if name == "qsd2_w3":
            report["example"] = example(DATA_DIR / name, rows, config, args.plots / "masks_w2_vs_w3.png")
            report["w3_worst"] = sorted(((r["image"], round(r["w3"]["f1"], 3)) for r in rows), key=lambda t: t[1])[:3]
    report["ablation"] = ablation(config, args.plots / "masks_ablation.png")
    (args.results / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
