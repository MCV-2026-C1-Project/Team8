"""Week 3 Task 2: texture descriptors (DCT, LBP, wavelet) on QSD1-W3, texture only.

Every configuration is evaluated with and without the Task 1 noise removal
(median 3x3 only on queries whose noise score exceeds 6), and per query group
(impulse noise / hue change / clean) from augmentations.pkl. The Week 2 best
colour descriptor is the reference.

Writes results/week3/texture/sweep.csv and the figures in plots/week3/.
Run: python src/visualizations/plot_week3_texture.py
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from descriptor_cli import load_descriptor_document
from descriptors import list_images, load_image
from filter_noise import query_preprocessor
from metrics import mapk
from retrieval import BBDD_DIR, DATA_DIR, load_gt, retrieve_config

ROOT = SRC.parent
BLUE, ORANGE, GREY, AQUA = "#2a78d6", "#eb6834", "#a9a8a3", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"
DENOISE = dict(name="median", ksize=3, noise_threshold=6.0)
COLOUR_DOC = ROOT / "results/week2/qsd1_best_with_norm.json"
SUBMITTED_DCT = ("DCT 4×4, 64 coef", "euclidean")  # results/week3/week3_100_best.json, used for QST1


def configurations():
    """(family, label, config, measures). Block layouts: g x g grid."""
    grid = lambda g: dict(mode="grid", grids=[g])
    out = []
    for g in (1, 2, 4, 8):
        for n in (8, 16, 32, 64, 128):
            out.append(("DCT", f"DCT {g}×{g}, {n} coef", dict(descriptor="dct", params=dict(
                n_coeffs=n, resize=256, spatial=grid(g))), ["l1", "euclidean"]))
    for g in (1, 2, 4):
        for mapping in ("riu2", "u2", "none"):
            for radius in (1, 2):
                # LBP compares neighbouring pixels: sample every block at the native
                # 256x256 image scale (256 / g per block) instead of upsampling it.
                out.append(("LBP", f"LBP {mapping} R{radius} {g}×{g}", dict(descriptor="lbp", params=dict(
                    radius=radius, points=8, mapping=mapping, resize=256 // g, spatial=grid(g))), ["l1", "chi2"]))
        out.append(("Multiscale LBP", f"MS-LBP u2 R1-3 {g}×{g}", dict(descriptor="multiscale_lbp", params=dict(
            radii=[1, 2, 3], points=[8, 8, 8], mapping="u2", resize=256 // g, spatial=grid(g))), ["l1", "chi2"]))
        for levels in (2, 3, 4):
            out.append(("Wavelet", f"Haar wavelet L{levels} {g}×{g}", dict(descriptor="wavelet", params=dict(
                levels=levels, resize=256, spatial=grid(g))), ["l1", "euclidean"]))
    return out


def evaluate(config, measure, query_dir, gt, groups, denoise):
    _, pred = retrieve_config(config, measure, query_dir=query_dir, k=10,
                              query_preprocess=query_preprocessor(DENOISE if denoise else None))
    row = dict(map1=mapk(gt, pred, 1), map5=mapk(gt, pred, 5))
    for name, mask in groups.items():
        row[f"map1_{name}"] = mapk([g for g, m in zip(gt, mask) if m], [p for p, m in zip(pred, mask) if m], 1)
    return row, pred


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def family_bars(rows, colour, out):
    """Best configuration of each family (by mAP@1, then mAP@5), with denoising."""
    fams = ["DCT", "LBP", "Multiscale LBP", "Wavelet"]
    # Ties at the top break towards the DCT configuration submitted for QST1.
    best = {f: max((r for r in rows if r["family"] == f and r["denoise"]),
                   key=lambda r: (r["map1"], r["map5"],
                                  (r["label"], r["measure"]) == SUBMITTED_DCT)) for f in fams}
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = np.arange(len(fams) + 1)
    unit = lambda m: m.replace("euclidean", "L2").replace("l1", "L1").replace("chi2", "χ²")
    grid_of = lambda label: label.split()[-1] if label.split()[-1][0].isdigit() else label.split(",")[0].split()[-1]
    short = {"DCT": lambda r: f"DCT\n{r['label'][4:]}",
             "LBP": lambda r: f"LBP {r['label'].split()[1].replace('none', '256').replace('u2', '59').replace('riu2', '10')} bins"
                              f"\n{r['label'].split()[2]}, {grid_of(r['label'])}",
             "Multiscale LBP": lambda r: f"MS-LBP 59 bins\nR1-3, {grid_of(r['label'])}",
             "Wavelet": lambda r: f"Haar wavelet\n{r['label'].split()[2][1:]} levels, {grid_of(r['label'])}"}
    names = [f"{short[f](best[f])}, {unit(best[f]['measure'])}" for f in fams]
    names.append("Colour (W2 best)\nYCbCr 4×4, L1")
    for k, (key, color, label) in enumerate([("map1", ORANGE, "mAP@1"), ("map5", BLUE, "mAP@5")]):
        values = [best[f][key] for f in fams] + [colour[key]]
        bars = ax.bar(x + (k - 0.5) * 0.38, values, 0.36, color=[color] * len(fams) + [GREY], label=label)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return best


def dct_heatmap(rows, out):
    grids, coefs = [1, 2, 4, 8], [8, 16, 32, 64, 128]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, measure, title in [(axes[0], "l1", "L1"), (axes[1], "euclidean", "L2 (Euclidean)")]:
        m = np.array([[next(r["map1"] for r in rows if r["family"] == "DCT" and r["denoise"] and r["measure"] == measure
                            and r["label"] == f"DCT {g}×{g}, {n} coef") for n in coefs] for g in grids])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(grids)):
            for j in range(len(coefs)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if m[i, j] > 0.6 else INK, fontweight="bold" if m[i, j] == 1 else "normal")
        ax.set_xticks(range(len(coefs)), coefs)
        ax.set_yticks(range(len(grids)), [f"{g}×{g}" for g in grids])
        ax.set_xlabel("coefficients per block", fontsize=9, color=MUTED)
        ax.set_title(f"mAP@1 · {title}", fontsize=9.5, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("grid", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def lbp_bars(rows, out):
    """Best mAP@1 per LBP mapping and grid (best radius / measure), with denoising."""
    mappings = [("riu2", "rotation-invariant\nuniform (10 bins)"), ("u2", "uniform\n(59 bins)"),
                ("none", "all codes\n(256 bins)")]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    x = np.arange(len(mappings))
    colors = [GREY, BLUE, ORANGE]
    for k, g in enumerate((1, 2, 4)):
        values = [max(r["map1"] for r in rows if r["family"] == "LBP" and r["denoise"]
                      and r["label"].startswith(f"LBP {m} ") and r["label"].endswith(f"{g}×{g}")) for m, _ in mappings]
        bars = ax.bar(x + (k - 1) * 0.27, values, 0.25, color=colors[k], label=f"{g}×{g} grid")
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [m[1] for m in mappings], fontsize=8)
    ax.set_ylabel("mAP@1 on QSD1-W3", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def dct_pipeline(query_dir, out, grid=4, n=64):
    """Grey 256x256 with the grid, one block's log|DCT| with the kept zig-zag coefficients."""
    image = load_image(list_images(query_dir)[0][0])
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (256, 256), interpolation=cv2.INTER_AREA)
    s = 256 // grid
    block = cv2.resize(gray[s:2 * s, 2 * s:3 * s], (256, 256)).astype(np.float32)
    block = (block - block.mean()) / max(block.std(), 1e-6)
    coeff = np.log1p(np.abs(cv2.dct(block)))[:16, :16]
    kept = np.zeros((16, 16), bool)
    count = 0
    for d in range(1, 31):
        pairs = [(y, d - y) for y in range(max(0, d - 15), min(15, d) + 1)]
        for y, x in (pairs if d % 2 else pairs[::-1]):
            if count < n:
                kept[y, x] = True
                count += 1
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("query (colour ignored)", fontsize=9, color=INK)
    axes[1].imshow(gray, cmap="gray")
    for k in range(1, grid):
        axes[1].axhline(k * s - 0.5, color=ORANGE, lw=1)
        axes[1].axvline(k * s - 0.5, color=ORANGE, lw=1)
    axes[1].add_patch(plt.Rectangle((2 * s - 0.5, s - 0.5), s, s, fill=False, ec=BLUE, lw=2))
    axes[1].set_title(f"grey 256×256, {grid}×{grid} blocks", fontsize=9, color=INK)
    axes[2].imshow(coeff, cmap="Blues")
    ys, xs = np.nonzero(kept)
    axes[2].scatter(xs, ys, s=6, c=ORANGE, marker="s")
    axes[2].set_title(f"log|DCT| of the blue block\norange: first {n} zig-zag coefs", fontsize=9, color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def qualitative(query_dir, colour_pred, texture_pred, gt, hue, out):
    """A hue-changed query that colour misses and texture finds."""
    cases = [i for i in np.flatnonzero(hue) if colour_pred[i][0] not in gt[i] and texture_pred[i][0] in gt[i]]
    if not cases:
        return None
    i = int(cases[0])
    paths, _ = list_images(query_dir)
    bbdd = lambda k: load_image(BBDD_DIR / f"bbdd_{k:05d}.jpg")
    rank = colour_pred[i].index(gt[i][0]) + 1 if gt[i][0] in colour_pred[i] else ">10"
    panels = [(load_image(paths[i]), f"query {paths[i].stem}\n(hue changed)", INK),
              (bbdd(colour_pred[i][0]), f"colour top-1: bbdd {colour_pred[i][0]}\nwrong (correct at rank {rank})", ORANGE),
              (bbdd(texture_pred[i][0]), f"DCT top-1: bbdd {texture_pred[i][0]}\ncorrect", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (im, title, color) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9, color=color)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return paths[i].stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd1_w3")
    ap.add_argument("--plots", type=Path, default=ROOT / "plots")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "week3" / "texture")
    ap.add_argument("--from-csv", action="store_true", help="Only redraw the sweep charts from sweep.csv")
    args = ap.parse_args()
    if args.from_csv:
        with (args.results / "sweep.csv").open(encoding="utf-8") as stream:
            rows = [dict(r, denoise=r["denoise"] == "True", **{k: float(v) for k, v in r.items() if k.startswith("map")})
                    for r in csv.DictReader(stream)]
        colour = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))["colour"]["denoise"]
        family_bars(rows, colour, args.plots / "texture_families.png")
        dct_heatmap(rows, args.plots / "texture_dct_grid.png")
        lbp_bars(rows, args.plots / "texture_lbp_mapping.png")
        return
    args.plots.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    gt = load_gt(args.query_dir)
    with open(args.query_dir / "augmentations.pkl", "rb") as stream:
        labels = pickle.load(stream)
    noise = np.array(["Impulse" in l for l in labels])
    hue = np.array(["Hue" in l for l in labels])
    groups = dict(noise=noise, hue=hue, clean=~noise & ~hue)

    rows, preds = [], {}
    for family, label, config, measures in configurations():
        for measure in measures:
            for denoise in (False, True):
                row, pred = evaluate(config, measure, args.query_dir, gt, groups, denoise)
                rows.append(dict(family=family, label=label, measure=measure, denoise=denoise, **row,
                                 config_json=json.dumps(config, sort_keys=True)))
                preds[(label, measure, denoise)] = pred
        best_here = max((r for r in rows if r["label"] == label), key=lambda r: r["map1"])
        print(f"{label:<28} best mAP@1 {best_here['map1']:.3f} ({best_here['measure']})", flush=True)

    cfg, measure = load_descriptor_document(COLOUR_DOC)
    colour = {}
    for denoise in (False, True):
        row, pred = evaluate(cfg, measure, args.query_dir, gt, groups, denoise)
        colour[denoise] = (row, pred)

    with (args.results / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = family_bars(rows, colour[True][0], args.plots / "texture_families.png")
    dct_heatmap(rows, args.plots / "texture_dct_grid.png")
    lbp_bars(rows, args.plots / "texture_lbp_mapping.png")
    dct_pipeline(args.query_dir, args.plots / "texture_dct_pipeline.png")
    top = best["DCT"]
    stem = qualitative(args.query_dir, colour[True][1], preds[(top["label"], top["measure"], True)], gt, hue,
                       args.plots / "texture_vs_colour.png")

    summary = dict(best={f: {k: r[k] for k in ("label", "measure", "map1", "map5", "map1_noise", "map1_hue",
                                                 "map1_clean")} for f, r in best.items()},
                   best_no_denoise={f: max((r for r in rows if r["family"] == f and not r["denoise"]),
                                           key=lambda r: (r["map1"], r["map5"]))["map1"] for f in best},
                   colour={("denoise" if d else "raw"): r for d, (r, _) in colour.items()},
                   dct_perfect=[f"{r['label']} {r['measure']}" for r in rows
                                if r["family"] == "DCT" and r["denoise"] and r["map1"] == 1.0],
                   qualitative_query=stem)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
