"""Evaluate a small, editable descriptor grid on QSD1 and write a CSV.

Edit EXPERIMENTS below: each grid value is a list of choices; bins are tuples.
Run: python src/run_experiments.py --output results/experiments.csv
An existing output is only replaced when --overwrite is supplied.
"""
import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from descriptors import DEFAULTS, channel_sizes, full_params, method_tag
from distances import MEASURES, rank
from metrics import mapk
from retrieval import BBDD_DIR, DATA_DIR, DESC_DIR, ROOT, load_gt, load_or_compute, retrieve
from rank_fusion import best_relevant_ranks, fuse_rankings, normalise_weights


# Cartesian products within each entry; entries and choices retain their order.
# To expand HSV, e.g.: "bins": [(32, 16, 8), (16, 16, 16)], "s_min": [20, 40].
# Use [False, True] for either Hue option to compare enabled/disabled variants.
COMMON_MEASURES = list(MEASURES)

# EXPERIMENTS = [
#     dict(
#         descriptor="rgb",
#         grid={
#             "bins": [
#                 (8, 8, 8),
#                 (8, 16, 16),
#                 (16, 8, 8),
#                 (16, 16, 8),
#                 (16, 16, 16),
#                 (16, 16, 32),
#                 (16, 32, 16),
#                 (32, 16, 8),
#                 (32, 16, 16),
#                 (32, 32, 16),
#                 (64, 32, 16),
#                 (64, 64, 32),
#             ]
#         },
#         measures=COMMON_MEASURES,
#     ),
# ]

EXPERIMENTS = [
    dict(
        descriptor="hsv",
        grid={
            "bins": [
                (16, 16, 8),
            ],
            "s_min": [20],
            "v_min": [40],
            "hue_valid_weight": [True],
            "hue_smoothing": [False],
        },
        measures=COMMON_MEASURES,
    ),
]


# EXPERIMENTS = [
#     dict(
#         descriptor="hsv",
#         grid={
#             "bins": [
#                 (16, 16, 8),
#                 (32, 16, 8),
#                 (32, 32, 16),
#                 (64, 32, 16),
#             ],
#             "s_min": [20, 40, 60],
#             "v_min": [20, 40, 60],
#             "hue_valid_weight": [False, True],
#             "hue_smoothing": [False, True],
#         },
#         measures=COMMON_MEASURES,
#     ),
# ]

FIELDS = ["descriptor", "bins", "parameters", "measure", "map1", "map5"]

# Separate opt-in study: descriptor parameters stay fixed across every row.
FUSION_DESCRIPTOR_PARAMS = {
    "hsv": dict(DEFAULTS["hsv"]),
    "rgb": dict(DEFAULTS["rgb"]),
    "lab": dict(DEFAULTS["lab"]),
    "ycbcr": dict(DEFAULTS["ycbcr"]),
}
FUSION_MEASURES = [
    "euclidean",
    "l1",
    "chi2",
    "intersection",
    "hellinger",
    "wasserstein",
]

FUSION_PAIRS = [
    ("euclidean", "l1"),
    ("euclidean", "chi2"),
    ("euclidean", "intersection"),
    ("euclidean", "hellinger"),
    ("euclidean", "wasserstein"),
    ("l1", "chi2"),
    ("l1", "intersection"),
    ("l1", "hellinger"),
    ("l1", "wasserstein"),
    ("chi2", "intersection"),
    ("chi2", "hellinger"),
    ("chi2", "wasserstein"),
    ("intersection", "hellinger"),
    ("intersection", "wasserstein"),
    ("hellinger", "wasserstein"),
]

FUSION_TRIPLES = [
    ("euclidean", "l1", "chi2"),
    ("euclidean", "l1", "intersection"),
    ("euclidean", "l1", "hellinger"),
    ("euclidean", "l1", "wasserstein"),
    ("euclidean", "chi2", "intersection"),
    ("euclidean", "chi2", "hellinger"),
    ("euclidean", "chi2", "wasserstein"),
    ("euclidean", "intersection", "hellinger"),
    ("euclidean", "intersection", "wasserstein"),
    ("euclidean", "hellinger", "wasserstein"),
    ("l1", "chi2", "intersection"),
    ("l1", "chi2", "hellinger"),
    ("l1", "chi2", "wasserstein"),
    ("l1", "intersection", "hellinger"),
    ("l1", "intersection", "wasserstein"),
    ("l1", "hellinger", "wasserstein"),
    ("chi2", "intersection", "hellinger"),
    ("chi2", "intersection", "wasserstein"),
    ("chi2", "hellinger", "wasserstein"),
    ("intersection", "hellinger", "wasserstein"),
]

PAIR_WEIGHTS = [
    (0.10, 0.90),
    (0.20, 0.80),
    (0.30, 0.70),
    (0.40, 0.60),
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.90, 0.10),
]

TRIPLE_WEIGHTS = [
    (1/3, 1/3, 1/3),

    (0.60, 0.20, 0.20),
    (0.20, 0.60, 0.20),
    (0.20, 0.20, 0.60),

    (0.50, 0.30, 0.20),
    (0.50, 0.20, 0.30),
    (0.30, 0.50, 0.20),
    (0.20, 0.50, 0.30),
    (0.30, 0.20, 0.50),
    (0.20, 0.30, 0.50),

    (0.40, 0.40, 0.20),
    (0.40, 0.20, 0.40),
    (0.20, 0.40, 0.40),
]
FUSION_FIELDS = ["descriptor", "bins", "parameters", "experiment_type", "measures", "weights",
                 "map1", "map5", "queries_improved_vs_l1", "queries_same_vs_l1",
                 "queries_worsened_vs_l1"]


def fusion_configurations(smoke=False):
    if smoke:
        return [
            ("single", ["l1"], [1.0]),
            ("single", ["wasserstein"], [1.0]),
            ("rank_fusion", ["l1", "wasserstein"], [0.8, 0.2]),
        ]

    return (
        [("single", [m], [1.0]) for m in FUSION_MEASURES]
        + [
            ("rank_fusion", list(pair), weights)
            for pair in FUSION_PAIRS
            for weights in PAIR_WEIGHTS
        ]
        + [
            ("rank_fusion", list(triple), weights)
            for triple in FUSION_TRIPLES
            for weights in TRIPLE_WEIGHTS
        ]
    )

def run_fusion_experiments(smoke=False, desc_dir=DESC_DIR, descriptor="hsv"):
    """QSD1 rank fusion with descriptors and complete measure rankings reused."""
    configs = fusion_configurations(smoke)
    params = full_params(descriptor, **FUSION_DESCRIPTOR_PARAMS[descriptor])
    query_dir = DATA_DIR / "qsd1_w1"
    gt = load_gt(query_dir)
    db_ids, db_desc = load_or_compute("bbdd", BBDD_DIR, descriptor, params, desc_dir)
    q_ids, q_desc = load_or_compute("qsd1", query_dir, descriptor, params, desc_dir)
    if len(q_desc) != len(gt) or q_ids.tolist() != list(range(len(gt))):
        raise ValueError("QSD1 queries must match ground-truth length and order (0, 1, ...)")
    sizes = channel_sizes(descriptor, **params)
    measures = list(dict.fromkeys(m for _, selected, _ in configs for m in selected))
    rankings = {m: np.asarray([rank(q, db_desc, m, sizes) for q in q_desc]) for m in measures}
    l1_ranks = best_relevant_ranks(db_ids[rankings["l1"]], gt)
    rows = []
    for kind, selected, weights in configs:
        weights = normalise_weights(weights, len(selected))
        orders = (rankings[selected[0]] if kind == "single" else np.asarray([
            fuse_rankings([rankings[m][i] for m in selected], weights)
            for i in range(len(q_desc))
        ]))
        ranked_ids = db_ids[orders]
        predictions = ranked_ids[:, :10].tolist()
        relevant_ranks = best_relevant_ranks(ranked_ids, gt)
        row = dict(descriptor=descriptor, bins="-".join(map(str, params["bins"])),
                   parameters=json.dumps({k: v for k, v in params.items() if k != "bins"}, sort_keys=True),
                   experiment_type=kind, measures="+".join(selected),
                   weights="+".join(format(float(w), ".15g") for w in weights),
                   map1=float(mapk(gt, predictions, 1)), map5=float(mapk(gt, predictions, 5)),
                   queries_improved_vs_l1=int(np.sum(relevant_ranks < l1_ranks)),
                   queries_same_vs_l1=int(np.sum(relevant_ranks == l1_ranks)),
                   queries_worsened_vs_l1=int(np.sum(relevant_ranks > l1_ranks)))
        rows.append(row)
        print(f"{kind} {row['measures']} ({row['weights']}): "
              f"mAP@1={row['map1']:.4f} mAP@5={row['map5']:.4f}")
    return rows


def iter_configurations(experiments):
    """Expand parameter grids and merge repeated descriptor configurations."""
    configurations = {}
    for experiment in experiments:
        method = experiment["descriptor"]
        grid = experiment.get("grid", {})
        measures = experiment["measures"]
        if not measures or any(m not in MEASURES for m in measures):
            raise ValueError(f"Invalid measures for {method}: {measures}")
        if any(not choices for choices in grid.values()):
            raise ValueError(f"Empty parameter choices for {method}")
        for values in product(*grid.values()):
            params = full_params(method, **dict(zip(grid, values)))
            if set(params) != set(full_params(method)):
                raise ValueError(f"Unknown descriptor parameters for {method}")
            sizes = params["bins"]
            if len(sizes) != 3 or any(int(b) != b or b <= 0 for b in sizes):
                raise ValueError("bins must contain three positive integers")
            tag = method_tag(method, **params)
            if tag not in configurations:
                configurations[tag] = (method, params, [])
            selected = configurations[tag][2]
            selected.extend(m for m in measures if m not in selected)
    return list(configurations.values())


def run_experiments(experiments, desc_dir=DESC_DIR):
    """Reuse cached descriptors once per configuration; always evaluate QSD1."""
    configurations = iter_configurations(experiments)
    query_dir = DATA_DIR / "qsd1_w1"
    gt = load_gt(query_dir)
    rows = []
    for method, params, measures in configurations:
        db_ids, db_desc = load_or_compute("bbdd", BBDD_DIR, method, params, desc_dir)
        q_ids, q_desc = load_or_compute("qsd1", query_dir, method, params, desc_dir)
        if len(gt) != len(q_desc):
            raise ValueError(f"{len(gt)} GT entries but {len(q_desc)} queries")
        if q_ids.tolist() != list(range(len(gt))):
            raise ValueError("QSD1 query IDs must follow ground-truth order (0, 1, ...)")
        for measure in measures:
            predictions = retrieve(q_desc, db_desc, db_ids, measure, 10,
                                   channel_sizes=channel_sizes(method, **params))
            row = dict(descriptor=method,
                       bins="-".join(str(b) for b in params["bins"]),
                       parameters=json.dumps({k: v for k, v in params.items() if k != "bins"},
                                             sort_keys=True),
                       measure=measure,
                       map1=float(mapk(gt, predictions, 1)),
                       map5=float(mapk(gt, predictions, 5)))
            rows.append(row)
            print(f"{method_tag(method, **params)} / {measure}: "
                  f"mAP@1={row['map1']:.4f} mAP@5={row['map5']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None,
                    help="CSV path (default: results/experiments.csv; fusion: results/<descriptor>_fusion_experiments.csv)")
    ap.add_argument("--fusion", action="store_true", help="Run the fixed-descriptor rank-fusion study")
    ap.add_argument("--fusion-descriptor", choices=list(FUSION_DESCRIPTOR_PARAMS), default="hsv")
    ap.add_argument("--smoke", action="store_true", help="With --fusion: two baselines and one equal-weight fusion")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    args = ap.parse_args()
    if args.smoke and not args.fusion:
        ap.error("--smoke requires --fusion")
    if args.output is None:
        name = f"{args.fusion_descriptor}_fusion_experiments.csv" if args.fusion else "experiments.csv"
        args.output = ROOT / "results" / name
    if args.output.exists() and not args.overwrite:
        ap.error(f"{args.output} already exists; use --overwrite to replace it")
    rows = (run_fusion_experiments(smoke=args.smoke, descriptor=args.fusion_descriptor)
            if args.fusion else run_experiments(EXPERIMENTS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FUSION_FIELDS if args.fusion else FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} experiments to {args.output}")


if __name__ == "__main__":
    main()
