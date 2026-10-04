"""Compare every distance / similarity measure for the best configuration of each descriptor.

Descriptors (defaults from descriptors.py unless overridden below):
  HSV base (hsv_baseline 180-256-256), HSV improved (hsv 16-16-16, S > 10, V > 40,
  H histogram not scaled by the valid-Hue fraction), CIELab (lab), YCbCr (ycbcr).

Evaluates all measures on QSD1, saves the scores to results/best_comparison.csv and
draws one grouped bar chart for mAP@1 and another for mAP@5 in plots/.

Example:
    py src/visualizations/plot_best_comparison.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))  # project modules (descriptors, retrieval, ...) live in src/

from descriptors import channel_sizes, full_params, method_tag
from distances import MEASURES
from metrics import mapk
from retrieval import BBDD_DIR, DATA_DIR, load_gt, load_or_compute, retrieve

ROOT = SRC.parent
# (method, label, parameters overriding the defaults)
DESCRIPTORS = [("hsv_baseline", "HSV base", {}),
               ("hsv", "HSV improved", dict(bins=(16, 16, 16), s_min=10, v_min=40,
                                            hue_valid_weight=False, hue_smoothing=False)),
               ("lab", "CIELab", {}),
               ("ycbcr", "YCbCr", {})]
# Bold suffix shown in the chart legend
LEGEND_SUFFIX = {"HSV base": r" $\bf{(method\ 1)}$", "HSV improved": r" $\bf{(method\ 2)}$"}
COLORS = ["#9E9E9E", "#E66101", "#5E3C99", "#1B9E77"]


def evaluate():
    query_dir = DATA_DIR / "qsd1_w1"
    gt = load_gt(query_dir)
    rows = []
    for method, label, overrides in DESCRIPTORS:
        params = full_params(method, **overrides)
        db_ids, db_desc = load_or_compute("bbdd", BBDD_DIR, method, params)
        _, q_desc = load_or_compute("qsd1", query_dir, method, params)
        sizes = channel_sizes(method, **params)
        for measure in MEASURES:
            pred = retrieve(q_desc, db_desc, db_ids, measure, 10, channel_sizes=sizes)
            rows.append(dict(descriptor=label, tag=method_tag(method, **params), measure=measure,
                             map1=float(mapk(gt, pred, 1)), map5=float(mapk(gt, pred, 5))))
            print(f"{label:13s} {measure:12s} mAP@1={rows[-1]['map1']:.3f} "
                  f"mAP@5={rows[-1]['map5']:.3f}")
    return pd.DataFrame(rows)


def grouped_bars(df, metric, title, out_path):
    measures = list(MEASURES)
    labels = [label for _, label, _ in DESCRIPTORS]
    width = 0.8 / len(labels)
    x = np.arange(len(measures))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (label, color) in enumerate(zip(labels, COLORS)):
        sub = df[df["descriptor"] == label].set_index("measure").reindex(measures)
        bars = ax.bar(x + (i - (len(labels) - 1) / 2) * width, sub[metric], width,
                      color=color, label=label + LEGEND_SUFFIX.get(label, ""))
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x, [m.capitalize() for m in measures])
    ax.set_ylabel(title)
    ax.set_ylim(0, min(1.0, df[metric].max() + 0.12))
    ax.set_title(f"{title} on QSD1", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = evaluate()
    csv_path = ROOT / "results" / "best_comparison.csv"
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    grouped_bars(df, "map1", "mAP@1", plots / "best_comparison_map1.png")
    grouped_bars(df, "map5", "mAP@5", plots / "best_comparison_map5.png")


if __name__ == "__main__":
    main()
