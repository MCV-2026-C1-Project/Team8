"""Week 3 Task 1 figures and numbers: noise detection, filter comparison and retrieval impact.

1. Detection: MAD noise score (noise_detection.compute_noise_score) of every query,
   against the augmentations.pkl labels, for QSD1-W3 and QSD2-W3.
2. Filters: PSNR/SSIM against non_augmented/ on the noisy queries, and the damage
   each filter does to the clean queries (why filtering must be selective).
3. Retrieval: QSD1-W3 mAP with the best texture (DCT) and best Week 2 colour
   descriptor, without filtering, always filtering and selective filtering.

Run: python src/visualizations/plot_week3_noise.py
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from descriptor_cli import load_descriptor_document
from descriptors import list_images, load_image
from filter_noise import make_filter, query_preprocessor
from metrics import mapk
from noise_detection import compute_noise_score
from retrieval import DATA_DIR, load_gt, retrieve_config

ROOT = SRC.parent
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a3"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"
THRESHOLD = 6.0
FILTERS = [("none", {}), ("average", dict(ksize=3)), ("average", dict(ksize=5)),
           ("gaussian", dict(ksize=3, sigma=1.0)), ("gaussian", dict(ksize=5, sigma=1.0)),
           ("median", dict(ksize=3)), ("median", dict(ksize=5)), ("median", dict(ksize=7)),
           ("ideal", dict(cutoff=40.0)), ("ideal", dict(cutoff=80.0))]


def label(name, params):
    if name == "none":
        return "no filter"
    if name == "ideal":
        return f"ideal low-pass r={params['cutoff']:.0f}"
    return f"{name} {params['ksize']}×{params['ksize']}"


def psnr(image, reference):
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def ssim(image, reference):
    """Mean SSIM on grey levels (11×11 Gaussian window, σ 1.5)."""
    x = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())


def load_set(folder):
    paths, _ = list_images(folder)
    with open(folder / "augmentations.pkl", "rb") as stream:
        labels = pickle.load(stream)
    noisy = np.array(["Impulse" in l for l in labels])
    hue = np.array(["Hue" in l for l in labels])
    images = [load_image(p) for p in paths]
    clean = [load_image(folder / "non_augmented" / p.name) for p in paths]
    return paths, images, clean, noisy, hue


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def detection(sets, out):
    """Per-query noise score bars for each set, threshold line; returns scores."""
    fig, axes = plt.subplots(1, len(sets), figsize=(10, 3.2), sharey=True)
    summary = {}
    for ax, (name, (paths, images, _, noisy, _)) in zip(axes, sets.items()):
        scores = []
        for image in images:
            s = compute_noise_score(image)
            if s["score"] is None:  # too few flat pixels: same relaxed retry as the pipeline would need
                s = compute_noise_score(image, min_smooth_fraction=0.01)
            scores.append(s["score"])
        scores = np.array(scores)
        colors = [ORANGE if n else BLUE for n in noisy]
        ax.bar(np.arange(len(scores)), scores, 0.75, color=colors)
        ax.axhline(THRESHOLD, color=INK, lw=1, ls="--")
        ax.set_title(f"{name.upper().replace('_', '-')}", fontsize=10, color=INK)
        ax.set_xlabel("query", fontsize=9, color=MUTED)
        style(ax)
        flagged = scores > THRESHOLD
        summary[name] = dict(noisy=int(noisy.sum()), detected=int((flagged & noisy).sum()),
                             false_alarms=int((flagged & ~noisy).sum()), clean=int((~noisy).sum()),
                             max_clean=float(scores[~noisy].max()), min_noisy=float(scores[noisy].min()))
    axes[0].set_ylabel("noise score (grey levels)", fontsize=9, color=MUTED)
    axes[0].set_ylim(0, 15)
    fig.legend(handles=[Patch(color=ORANGE, label="impulse noise (GT label)"),
                        Patch(color=BLUE, label="no noise (clean or hue change)"),
                        Line2D([], [], color=INK, lw=1, ls="--", label=f"threshold {THRESHOLD:g}")],
               frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return summary


def noise_example(paths, images, clean, noisy, out):
    """Central zoom of one noisy query: as given | clean | difference."""
    i = int(np.flatnonzero(noisy)[0])
    image, reference = images[i], clean[i]
    h, w = image.shape[:2]
    s = min(h, w) // 4
    crop = lambda a: a[h // 2 - s // 2:h // 2 + s // 2, w // 2 - s // 2:w // 2 + s // 2]
    diff = np.abs(image.astype(int) - reference.astype(int)).max(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    panels = [(cv2.cvtColor(crop(image), cv2.COLOR_BGR2RGB), f"query {paths[i].stem} (as given)"),
              (cv2.cvtColor(crop(reference), cv2.COLOR_BGR2RGB), "clean version"),
              (crop(diff) > 30, "pixels changed by > 30")]
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(data, cmap="gray" if data.ndim == 2 else None, interpolation="nearest")
        ax.set_title(title, fontsize=9, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    changed = [float((np.abs(images[j].astype(int) - clean[j].astype(int)).max(axis=2) > 30).mean())
               for j in np.flatnonzero(noisy)]
    return paths[i].stem, float(np.mean(changed))


def filter_quality(images, clean, noisy):
    rows = []
    for name, params in FILTERS:
        apply = (lambda im: im) if name == "none" else make_filter(name, **params)
        out = [apply(im) for im in images]
        p = np.array([psnr(o, c) for o, c in zip(out, clean)])
        q = np.array([ssim(o, c) for o, c in zip(out, clean)])
        untouched = ~noisy & np.isfinite(p)  # identical clean copies have infinite PSNR
        rows.append(dict(filter=label(name, params), psnr_noisy=p[noisy].mean(), ssim_noisy=q[noisy].mean(),
                         psnr_clean=p[untouched].mean(), ssim_clean=q[~noisy].mean()))
    return rows


def filter_bars(rows, out):
    """PSNR on noisy queries (gain) next to PSNR on clean queries (damage)."""
    order = sorted(rows, key=lambda r: r["psnr_noisy"])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key, title in [(axes[0], "psnr_noisy", "7 noisy queries: higher = noise removed"),
                           (axes[1], "psnr_clean", "clean queries: lower = detail destroyed")]:
        colors = [GREY if r["filter"] == "no filter" else ORANGE if r["filter"] == "median 3×3" else BLUE
                  for r in order]
        ax.barh(y, [r[key] for r in order], 0.7, color=colors)
        for k, r in enumerate(order):
            ax.text(r[key] + 0.3, k, f"{r[key]:.1f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("PSNR vs clean version (dB)", fontsize=9, color=MUTED)
        ax.set_xlim(20, max(r[key] for r in order) + 4)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    axes[0].set_yticks(y, [r["filter"] for r in order])
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def zoom_grid(paths, images, clean, noisy, out):
    """Same central zoom of one noisy query under several filters, with PSNR."""
    i = int(np.flatnonzero(noisy)[2])
    image, reference = images[i], clean[i]
    h, w = image.shape[:2]
    s = min(h, w) // 5
    crop = lambda a: cv2.cvtColor(a[h // 2 - s // 2:h // 2 + s // 2, w // 2 - s // 2:w // 2 + s // 2],
                                  cv2.COLOR_BGR2RGB)
    shown = [("noisy", image), ("median 3×3", make_filter("median", ksize=3)(image)),
             ("gaussian 5×5", make_filter("gaussian", ksize=5, sigma=1.0)(image)),
             ("average 3×3", make_filter("average", ksize=3)(image)),
             ("ideal low-pass r=80", make_filter("ideal", cutoff=80.0)(image)), ("clean (reference)", reference)]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.4))
    for ax, (title, im) in zip(axes.ravel(), shown):
        ax.imshow(crop(im), interpolation="nearest")
        sub = "" if title.startswith("clean") else f"\n{psnr(im, reference):.1f} dB"
        ax.set_title(title + sub, fontsize=9, color=ORANGE if title == "median 3×3" else INK,
                     fontweight="bold" if title == "median 3×3" else "normal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return paths[i].stem


def retrieval(query_dir, noisy, hue, out):
    systems = [("Texture: DCT 64 coef, 4×4, L2", ROOT / "results/week3/week3_100_best.json"),
               ("Colour (W2 best): YCbCr 64³ norm, 4×4, L1", ROOT / "results/week2/qsd1_best_with_norm.json")]
    policies = [("no filter", None, GREY), ("median 3×3 on all", dict(name="median", ksize=3), BLUE),
                ("median 3×3 if score > 6", dict(name="median", ksize=3, noise_threshold=THRESHOLD), ORANGE)]
    gt = load_gt(query_dir)
    rows = []
    for system, path in systems:
        cfg, measure = load_descriptor_document(path)
        for policy, filt, _ in policies:
            _, pred = retrieve_config(cfg, measure, query_dir=query_dir, k=10,
                                      query_preprocess=query_preprocessor(filt))
            subset = lambda m: [x for x, keep in zip(pred, m) if keep]
            sub_gt = lambda m: [x for x, keep in zip(gt, m) if keep]
            rows.append(dict(system=system, policy=policy, map1=mapk(gt, pred, 1), map5=mapk(gt, pred, 5),
                             map1_noisy=mapk(sub_gt(noisy), subset(noisy), 1),
                             map1_clean=mapk(sub_gt(~noisy & ~hue), subset(~noisy & ~hue), 1),
                             map1_hue=mapk(sub_gt(hue), subset(hue), 1)))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(systems))
    width = 0.26
    for k, (policy, _, color) in enumerate(policies):
        values = [r["map1"] for r in rows if r["policy"] == policy]
        bars = ax.bar(x + (k - 1) * width, values, width * 0.92, color=color, label=policy)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [s for s, _ in systems], fontsize=8.5)
    ax.set_ylabel("mAP@1 on QSD1-W3", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rows


def wavelet_filters(query_dir, out):
    """Wavelet texture (high-frequency detail statistics): every filter applied to all
    queries vs only to the queries the detector flags."""
    cfg = dict(descriptor="wavelet", params=dict(levels=3, resize=256, spatial=dict(mode="grid", grids=[4])))
    gt = load_gt(query_dir)
    shown = [("median", dict(ksize=3)), ("median", dict(ksize=5)), ("gaussian", dict(ksize=5, sigma=1.0)),
             ("average", dict(ksize=3)), ("ideal", dict(cutoff=80.0)), ("ideal", dict(cutoff=40.0))]
    _, pred = retrieve_config(cfg, "euclidean", query_dir=query_dir, k=10)
    baseline = mapk(gt, pred, 1)
    rows = []
    for name, params in shown:
        values = []
        for threshold in (None, THRESHOLD):
            filt = dict(name=name, **params, **({} if threshold is None else dict(noise_threshold=threshold)))
            _, pred = retrieve_config(cfg, "euclidean", query_dir=query_dir, k=10,
                                      query_preprocess=query_preprocessor(filt))
            values.append(mapk(gt, pred, 1))
        rows.append(dict(filter=label(name, params), always=values[0], selective=values[1], none=baseline))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(rows))
    for k, (key, color, name) in enumerate([("always", BLUE, "filter all 30 queries"),
                                            ("selective", ORANGE, "filter only if score > 6")]):
        bars = ax.bar(x + (k - 0.5) * 0.38, [r[key] for r in rows], 0.36, color=color, label=name)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8, color=INK)
    ax.axhline(baseline, color=GREY, lw=1.2, ls="--", label=f"no filter ({baseline:.2f})")
    ax.set_xticks(x, [r["filter"].replace("ideal low-pass", "ideal LP") for r in rows], fontsize=8.5)
    ax.set_ylabel("mAP@1 on QSD1-W3", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", type=Path, default=ROOT / "plots")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "week3" / "noise")
    args = ap.parse_args()
    args.plots.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    sets = {name: load_set(DATA_DIR / name) for name in ("qsd1_w3", "qsd2_w3")}
    paths, images, clean, noisy, hue = sets["qsd1_w3"]

    summary = detection(sets, args.plots / "noise_detection.png")
    example, density = noise_example(paths, images, clean, noisy, args.plots / "noise_example.png")
    rows = filter_quality(images, clean, noisy)
    zoom = zoom_grid(paths, images, clean, noisy, args.plots / "filter_zoom.png")
    retrieval_rows = retrieval(DATA_DIR / "qsd1_w3", noisy, hue, args.plots / "noise_retrieval.png")
    wavelet_rows = wavelet_filters(DATA_DIR / "qsd1_w3", args.plots / "noise_retrieval_wavelet.png")
    for r in wavelet_rows:
        print(f"wavelet {r['filter']:<22} always {r['always']:.3f} selective {r['selective']:.3f} none {r['none']:.3f}")

    for name, data in (("filter_quality", rows), ("retrieval", retrieval_rows), ("wavelet_filters", wavelet_rows)):
        with (args.results / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = dict(detection=summary, example_query=example, mean_changed_pixels=density, zoom_query=zoom)
    (args.results / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for r in rows:
        print(f"{r['filter']:<22} noisy {r['psnr_noisy']:.2f} dB / {r['ssim_noisy']:.3f}   "
              f"clean {r['psnr_clean']:.2f} dB / {r['ssim_clean']:.3f}")
    for r in retrieval_rows:
        print(f"{r['system']:<44} {r['policy']:<24} mAP@1 {r['map1']:.3f} mAP@5 {r['map5']:.3f} | "
              f"noisy {r['map1_noisy']:.3f} clean {r['map1_clean']:.3f} hue {r['map1_hue']:.3f}")


if __name__ == "__main__":
    main()
