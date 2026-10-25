"""Week 3 Task 4: multi-painting retrieval on QSD2-W3.

1. Ablation from the QSD1 winner to the final DCT + Lab fusion, read from
   results/week3/qsd2_search.csv (search_multi.py, no noise filter).
2. Qualitative results of the final model (results/week3/qsd2_best.json): a
   correctly retrieved two-painting photo and every missed painting.

Run: python src/visualizations/plot_week3_retrieval.py
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
from evaluate_background import instance_diagnostics, load_instance_annotations, polygon_masks
from metrics import evaluate_painting_retrieval, normalize_painting_gt
from descriptors import list_images, load_image
from retrieval import BBDD_DIR, DATA_DIR, retrieve_multi_config

ROOT = SRC.parent
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#a9a8a3", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e5e4e0"
FS = 15  # example titles: the figure is shown at ~4" wide on the slide
SEGMENTATION = ROOT / "configs/week3/task3_selected.json"
BEST = ROOT / "results/week3/qsd2_best.json"

# (label, descriptor, params subset, measure, inset, fusion weight or None)
STEPS = [("QSD1 best: DCT 64 coef, 4×4, L2", "dct", dict(n_coeffs=64, grids=4), "euclidean", 0.0, None),
         ("coarser DCT: 32 coef, 2×2, L1", "dct", dict(n_coeffs=32, grids=2), "l1", 0.0, None),
         ("+ trim 3 % of each box side", "dct", dict(n_coeffs=32, grids=2), "l1", 0.03, None),
         ("colour only: Lab 8-64-64, 5×5, L1", "lab", dict(grids=5), "l1", 0.03, None),
         ("DCT + Lab fused 50/50 (final)", "fusion", dict(n_coeffs=32, grids=2), "l1+l1", 0.03, 0.5)]


def ablation(search_csv, out):
    with open(search_csv, encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if json.loads(r["query_filter"]) == {"name": "none"}]

    def find(descriptor, params, measure, inset, weight):
        for r in rows:
            p = json.loads(r["config_json"])["params"]
            same = (r["descriptor"] == descriptor and r["measure"] == measure and float(r["inset"]) == inset
                    and p["spatial"]["grids"][0] == params["grids"]
                    and ("n_coeffs" not in params or p.get("n_coeffs") == params["n_coeffs"])
                    and (weight is None or float(r["fusion_weight"]) == weight))
            if same:
                return float(r["qsd2_map1"]), float(r["qsd2_map5"])
        raise ValueError(f"No search row for {descriptor} {params} {measure} {inset} {weight}")

    values = [find(*step[1:]) for step in STEPS]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(STEPS))[::-1]
    colors = [GREY, BLUE, BLUE, GREY, ORANGE]
    ax.barh(y, [v[0] for v in values], 0.62, color=colors)
    for k, (v1, _) in enumerate(values):
        ax.text(v1 + 0.01, y[k], f"{v1:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [s[0] for s in STEPS], fontsize=8.5)
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("painting mAP@1 on QSD2-W3", fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {s[0]: v for s, v in zip(STEPS, values)}


def thumb(image, height=220):
    scale = height / image.shape[0]
    return cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def qualitative(query_dir, out):
    """Run the final model; show one correct two-painting photo and every miss."""
    doc = json.loads(BEST.read_text(encoding="utf-8"))
    config, measure = load_descriptor_document(BEST)
    prov = doc["provenance"]
    paths, _ = list_images(query_dir)
    annotations = load_instance_annotations(query_dir / "frames.pkl", paths)
    with open(query_dir / "gt_corresps.pkl", "rb") as stream:
        gt = normalize_painting_gt(pickle.load(stream))
    with open(query_dir / "augmentations.pkl", "rb") as stream:
        labels = pickle.load(stream)
    matches, records = [], []

    def observe(path, mask, paintings):
        truth = polygon_masks(annotations[path.name], mask.shape)
        matches.append(instance_diagnostics([p.mask for p in paintings], truth, 0.5, return_matches=True)["matches"])

    _, results = retrieve_multi_config(config, measure, json.loads(SEGMENTATION.read_text(encoding="utf-8")),
                                       query_dir=query_dir, k=10, query_filter=prov.get("query_filter"),
                                       diagnostics=records, on_segmented=observe,
                                       crop_inset=prov.get("crop_inset", 0.0), fusion=prov.get("fusion"))
    evaluation = evaluate_painting_retrieval(gt, results, matches=matches, ks=(1, 5))
    misses, success = [], None
    for q, (truth, match, record) in enumerate(zip(gt, matches, records)):
        # Each GT painting is a list of acceptable BBDD IDs.
        ok = all(m is not None and results[q][m][0] in t for t, m in zip(truth, match))
        if ok and len(truth) == 2 and success is None:
            success = q
        for t, m in zip(truth, match):
            if m is None or results[q][m][0] not in t:
                ranking = [] if m is None else results[q][m]
                rank = next((i for i, v in enumerate(ranking, 1) if v in t), None)
                misses.append(dict(query=q, gt=t[0], painting=m, top=ranking[:1], rank=rank, label=labels[q]))

    bbdd = lambda k: load_image(BBDD_DIR / f"bbdd_{k:05d}.jpg")
    rows = 1 + len(misses)
    fig = plt.figure(figsize=(7.4, 1.9 * rows))
    grid = fig.add_gridspec(rows, 3, width_ratios=[1.6, 1, 1])
    # Row 0: the correct two-painting photo with its boxes and top-1 matches.
    image = load_image(paths[success])
    shown = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    for item in records[success]["paintings"]:
        x0, y0, x1, y1 = item["bbox"]
        cv2.rectangle(shown, (x0, y0), (x1, y1), (27, 175, 122), max(4, image.shape[1] // 150))
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(shown)
    ax.set_title(f"{paths[success].stem}: 2 found", fontsize=FS, color=INK)
    for j, item in enumerate(records[success]["paintings"][:2]):
        ax = fig.add_subplot(grid[0, 1 + j])
        ax.imshow(thumb(bbdd(item["ranking"][0])))
        side = "left" if j == 0 else "right"
        ax.set_title(f"{side} → {item['ranking'][0]} ✓", fontsize=FS, color=GREEN)
    # One row per miss: crop | correct painting | retrieved top-1.
    for r, miss in enumerate(misses, start=1):
        image = load_image(paths[miss["query"]])
        x0, y0, x1, y1 = records[miss["query"]]["paintings"][miss["painting"]]["bbox"]
        noise = "noisy" if "Impulse" in miss["label"] else "hue" if "Hue" in miss["label"] else "clean"
        panels = [(image[y0:y1, x0:x1], f"{paths[miss['query']].stem} crop ({noise})", INK),
                  (bbdd(miss["gt"]), f"correct: {miss['gt']}", GREEN),
                  (bbdd(miss["top"][0]), f"top-1: {miss['top'][0]} ✗", ORANGE)]
        for c, (im, title, color) in enumerate(panels):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(thumb(im))
            ax.set_title(title, fontsize=FS, color=color)
    for ax in fig.axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return dict(map1=evaluation["map1"], map5=evaluation["map5"], count_accuracy=evaluation["count_accuracy"],
                success=paths[success].stem, misses=[dict(m, query=paths[m["query"]].stem) for m in misses])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query-dir", type=Path, default=DATA_DIR / "qsd2_w3")
    ap.add_argument("--plots", type=Path, default=ROOT / "plots")
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "week3" / "retrieval")
    args = ap.parse_args()
    args.plots.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    report = dict(ablation=ablation(ROOT / "results/week3/qsd2_search.csv", args.plots / "retrieval_ablation.png"),
                  final=qualitative(args.query_dir, args.plots / "retrieval_examples.png"))
    (args.results / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
