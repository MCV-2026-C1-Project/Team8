"""Evaluate Python, JSON or CLI experiment configurations on cropped QSD1.

Edit EXPERIMENTS below: each grid value is a list of choices; bins are tuples.
Run: python src/run_experiments.py --output results/experiments.csv
Named sweeps: --grid lab or --grid ycbcr (see GRIDS).
JSON suites: --config configs/week2/phase_a.json; inspect with --dry-run.
Direct CLI: --descriptor hsv_hs2d --bins 16 16 --spatial-mode grid --grids 2.
An existing output is only replaced when --overwrite is supplied.
"""
import argparse
import csv
import json
from pathlib import Path

from descriptors import method_tag
from distances import MEASURES
from metrics import mapk
from retrieval import (BBDD_DIR, DATA_DIR, DESC_DIR, ROOT, load_gt, retrieve,
                       load_configured_data, validate_gt_order)
from descriptor_config import canonical_json, descriptor_dim, measure_channels
from descriptor_cli import add_descriptor_arguments, config_from_args
from experiment_config import expand_experiments, suite_settings
from spatial import resolve_spatial


# Cartesian products within each entry; entries and choices retain their order.
# To expand HSV, e.g.: "bins": [(32, 16, 8), (16, 16, 16)], "s_min": [20, 40].
# Use [False, True] for either Hue option to compare enabled/disabled variants.
COMMON_MEASURES = list(MEASURES)

# EDITABLE PYTHON CONFIGURATION: pass these dictionaries directly to run_experiments.
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


# Bin settings of the CIELab sweep (config_results/cielab_experiments.csv),
# reused for YCbCr so both colour spaces are compared on the same grid.
LAB_BINS = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64),
            (4, 32, 32), (8, 32, 32), (16, 32, 32), (8, 64, 64)]

LAB_EXPERIMENTS = [
    dict(descriptor="lab", grid={"bins": LAB_BINS}, measures=COMMON_MEASURES),
]

# Cr/Cb concentrate around 128, so narrower chroma ranges are also tried.
YCBCR_EXPERIMENTS = [
    dict(descriptor="ycbcr", grid={"bins": LAB_BINS,
                                   "chroma_range": [(0, 256), (64, 192), (96, 160)],
                                   "chroma_overflow": [False, True]},
         measures=COMMON_MEASURES),
]

GRIDS = {"default": EXPERIMENTS, "lab": LAB_EXPERIMENTS, "ycbcr": YCBCR_EXPERIMENTS}

FIELDS = ["descriptor", "bins", "parameters", "measure", "map1", "map5",
          "spatial_mode", "spatial_grids", "level_weights", "dimension", "tag",
          "config_json", "experiment_ids"]


def iter_configurations(experiments):
    """Legacy Python API, now also supporting joint/spatial configurations."""
    return [(j["config"]["descriptor"], j["config"]["params"], j["measures"])
            for j in expand_experiments(experiments)]


def dataset_paths(suite, db_dir=None, query_dir=None, gt=None):
    """Suite paths are relative to repository root; explicit Python/CLI paths to cwd."""
    def choose(override, name, default):
        if override is not None:
            return Path(override)
        if name in suite:
            path = Path(suite[name])
            return path if path.is_absolute() else ROOT / path
        return default
    db = choose(db_dir, "db_dir", BBDD_DIR)
    query = choose(query_dir, "query_dir", DATA_DIR / "qsd1_w1")
    return db, query, choose(gt, "gt", query / "gt_corresps.pkl")


def sorted_rows(rows, select_by="map1"):
    if select_by not in ("map1", "map5"):
        raise ValueError("select_by must be map1 or map5")
    secondary = "map5" if select_by == "map1" else "map1"
    return sorted(rows, key=lambda r: (-r[select_by], -r[secondary], r["tag"], r["measure"]))


def run_experiments(experiments, desc_dir=DESC_DIR, *, db_dir=None, query_dir=None,
                    gt=None, select_by="map1", diagnostics=None):
    """Shared Python/JSON/CLI runner. Accept lists of dictionaries directly.

    Each entry has descriptor, optional params, grid, spatial and measures.
    diagnostics can be a caller-owned list populated with per-query records.
    Only one configuration's descriptor matrices are retained at a time.
    """
    jobs = expand_experiments(experiments)
    suite = suite_settings(experiments)
    db_dir, query_dir, gt_path = dataset_paths(suite, db_dir, query_dir, gt)
    if "qst" in query_dir.name.lower():
        raise ValueError("Experiment selection requires development queries, not a QST test set")
    ground_truth = load_gt(query_dir, gt_path)
    rows = []
    for index, job in enumerate(jobs, 1):
        config = job["config"]
        method, params = config["descriptor"], config["params"]
        tag = method_tag(method, **params)
        print(f"[{index}/{len(jobs)}] {tag} ({descriptor_dim(method, **params)} values)")
        db_ids, db_desc, q_ids, q_desc = load_configured_data(config, db_dir, query_dir, desc_dir)
        validate_gt_order(q_ids, ground_truth)
        spatial = resolve_spatial(params.get("spatial"))
        for measure in job["measures"]:
            predictions = retrieve(q_desc, db_desc, db_ids, measure,
                                   len(db_ids) if diagnostics is not None else 10,
                                   channel_sizes=measure_channels(method, params, measure))
            row = dict(descriptor=method, bins="-".join(map(str, params["bins"])),
                       parameters=canonical_json({k: v for k, v in params.items() if k != "bins"}),
                       measure=measure, map1=float(mapk(ground_truth, predictions, 1)),
                       map5=float(mapk(ground_truth, predictions, 5)),
                       spatial_mode=spatial["mode"], spatial_grids=canonical_json(spatial["grids"]),
                       level_weights=canonical_json(spatial["level_weights"]),
                       dimension=descriptor_dim(method, **params), tag=tag,
                       config_json=canonical_json(config), experiment_ids=canonical_json(job["ids"]))
            rows.append(row)
            if diagnostics is not None:
                for qid, actual, predicted in zip(q_ids, ground_truth, predictions):
                    first = next((i for i, value in enumerate(predicted, 1) if value in actual), None)
                    diagnostics.append(dict(tag=tag, measure=measure, query_id=int(qid),
                                            gt=actual, top_ids=predicted[:10], first_relevant_rank=first))
            print(f"  {measure}: mAP@1={row['map1']:.4f} mAP@5={row['map5']:.4f}")
        del db_desc, q_desc
    return sorted_rows(rows, select_by)


def write_json(path, document, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def export_best(rows, path, select_by="map1", provenance=None, overwrite=False):
    if not rows:
        raise ValueError("Cannot export a winner from empty results")
    best = sorted_rows(rows, select_by)[0]
    document = dict(config=json.loads(best["config_json"]), measure=best["measure"],
                    tag=best["tag"], select_by=select_by,
                    development_metrics={key: best[key] for key in ("map1", "map5")},
                    provenance=provenance or {})
    write_json(path, document, overwrite)
    return document


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_descriptor_arguments(ap)
    ap.add_argument("--grid", choices=list(GRIDS), help="Legacy editable Python grid (default: default)")
    ap.add_argument("--measures", nargs="+", choices=list(MEASURES))
    for name in ("db-dir", "query-dir", "gt", "cache-dir", "output", "export-best", "diagnostics"):
        ap.add_argument(f"--{name}", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate/display all configurations without reading images")
    ap.add_argument("--select-by", choices=("map1", "map5"), default="map1")
    args = ap.parse_args(argv)
    try:
        if sum(x is not None for x in (args.config, args.grid, args.descriptor)) > 1:
            raise ValueError("Choose one of --config, --grid or --descriptor")
        if args.config:
            with args.config.open(encoding="utf-8") as stream:
                experiments = json.load(stream)
        elif args.descriptor:
            config = config_from_args(args)
            experiments = [dict(**config, measures=args.measures or ["l1", "chi2", "hellinger"])]
        else:
            experiments = GRIDS[args.grid or "default"]
        # Descriptor flags have a single meaning; do not silently ignore them in suites.
        if not args.descriptor:
            ignored = {key: value for key, value in vars(args).items()
                       if key in {"bins", "hsv_bins", "rgb_bins", "lab_bins", "ycbcr_bins", "s_min", "v_min",
                                  "chroma_weight", "v_bins", "l_bins", "y_range", "chroma_range", "hue_valid_weight",
                                  "hue_smoothing", "filter_hue", "achromatic_bin", "include_v", "include_l",
                                  "chroma_overflow", "spatial_mode", "grids", "level_weights"} and value is not None}
            if ignored:
                raise ValueError("Descriptor CLI parameters require --descriptor; edit suite/grid settings otherwise")
        if args.measures:
            experiments = dict(suite_settings(experiments))
            experiments["experiments"] = [dict(entry, measures=args.measures) for entry in experiments["experiments"]]
        jobs = expand_experiments(experiments)
        if args.dry_run:
            for job in jobs:
                config = job["config"]
                print(f"{method_tag(config['descriptor'], **config['params'])} "
                      f"dim={descriptor_dim(config['descriptor'], **config['params'])} measures={job['measures']}")
                print(canonical_json(config))
            print(f"{len(jobs)} extraction configurations; {sum(len(j['measures']) for j in jobs)} evaluations")
            return
        output = args.output or ROOT / "results" / "experiments.csv"
        targets = [p for p in (output, args.export_best, args.diagnostics) if p is not None]
        if len({p.resolve() for p in targets}) != len(targets):
            raise ValueError("CSV, export and diagnostics must use different output paths")
        for path in targets:
            if path.exists() and not args.overwrite:
                raise ValueError(f"{path} already exists; use --overwrite to replace it")
        diagnostics = [] if args.diagnostics else None
        rows = run_experiments(experiments, args.cache_dir or DESC_DIR, db_dir=args.db_dir,
                               query_dir=args.query_dir, gt=args.gt, select_by=args.select_by,
                               diagnostics=diagnostics)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        if args.diagnostics:
            write_json(args.diagnostics, diagnostics, args.overwrite)
        if args.export_best:
            paths = dataset_paths(suite_settings(experiments), args.db_dir, args.query_dir, args.gt)
            export_best(rows, args.export_best, args.select_by,
                        provenance=dict(source=str(args.config or args.grid or "Python/CLI"),
                                        db_dir=str(paths[0].resolve()), query_dir=str(paths[1].resolve()),
                                        gt=str(paths[2].resolve())), overwrite=args.overwrite)
        print(f"Saved {len(rows)} evaluations to {output}; best by {args.select_by}:")
        for row in rows[:10]:
            print(f"  {row['map1']:.4f} / {row['map5']:.4f}  {row['tag']} / {row['measure']}")
    except (ValueError, TypeError, OSError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
