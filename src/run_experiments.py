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

from descriptors import DEFAULTS, channel_sizes, full_params, method_tag
from distances import MEASURES
from metrics import mapk
from retrieval import BBDD_DIR, DATA_DIR, DESC_DIR, ROOT, load_gt, load_or_compute, retrieve


# Cartesian products within each entry; entries and choices retain their order.
# To expand HSV, e.g.: "bins": [(32, 16, 8), (16, 16, 16)], "s_min": [20, 40].
# Use [False, True] for either Hue option to compare enabled/disabled variants.
COMMON_MEASURES = list(MEASURES)

EXPERIMENTS = [
    dict(
        descriptor="rgb",
        grid={
            "bins": [
                (8, 8, 8),
                (8, 16, 16),
                (16, 8, 8),
                (16, 16, 8),
                (16, 16, 16),
                (16, 16, 32),
                (16, 32, 16),
                (32, 16, 8),
                (32, 16, 16),
                (32, 32, 16),
                (64, 32, 16),
                (64, 64, 32),
            ]
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
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "experiments.csv")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV")
    args = ap.parse_args()
    if args.output.exists() and not args.overwrite:
        ap.error(f"{args.output} already exists; use --overwrite to replace it")
    rows = run_experiments(EXPERIMENTS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} experiments to {args.output}")


if __name__ == "__main__":
    main()
