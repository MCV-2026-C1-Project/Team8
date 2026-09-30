"""Retrieve the top-K BBDD images for every query and (optionally) evaluate mAP@K.

Examples:
    py src/run_retrieval.py --descriptor hsv --measure hellinger
    py src/run_retrieval.py --descriptor ycbcr --measure chi2 --chroma-range 64 192
    py src/run_retrieval.py --descriptor hsv --measure hellinger --query-set qst1 --no-gt --output results/qst1_method1.pkl
"""
import argparse
import pickle
from pathlib import Path

from descriptors import METHODS, DEFAULTS, channel_sizes
from distances import MEASURES
from metrics import mapk
from retrieval import BBDD_DIR, load_gt, load_or_compute, query_folder, retrieve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptor", choices=list(METHODS), required=True)
    ap.add_argument("--measure", choices=list(MEASURES), required=True)
    ap.add_argument("--query-set", default="qsd1",
                    help="Query set name (e.g. qsd1 -> data/qsd1_w1) or a folder path")
    ap.add_argument("--k", type=int, default=10, help="Number of results per query")
    ap.add_argument("--no-gt", action="store_true", help="Query set has no ground truth: skip evaluation")
    ap.add_argument("--output", default=None, help="Save results (list of lists of IDs) to this .pkl")
    # Descriptor parameters (same defaults as compute_descriptors.py)
    ap.add_argument("--ycbcr-bins", type=int, nargs=3, default=DEFAULTS["ycbcr"]["bins"], metavar=("Y", "CR", "CB"))
    ap.add_argument("--chroma-range", type=int, nargs=2, default=DEFAULTS["ycbcr"]["chroma_range"], metavar=("LO", "HI"))
    ap.add_argument("--hsv-bins", type=int, nargs=3, default=DEFAULTS["hsv"]["bins"], metavar=("H", "S", "V"))
    ap.add_argument("--s-min", type=int, default=DEFAULTS["hsv"]["s_min"])
    ap.add_argument("--v-min", type=int, default=DEFAULTS["hsv"]["v_min"])
    ap.add_argument("--rgb-bins", type=int, nargs=3, default=DEFAULTS["rgb"]["bins"], metavar=("R", "G", "B"))
    ap.add_argument("--lab-bins", type=int, nargs=3, default=DEFAULTS["lab"]["bins"], metavar=("L", "A", "B"))
    args = ap.parse_args()

    params = {
        "ycbcr": dict(bins=tuple(args.ycbcr_bins), chroma_range=tuple(args.chroma_range)),
        "hsv": dict(bins=tuple(args.hsv_bins), s_min=args.s_min, v_min=args.v_min),
        "rgb": dict(bins=tuple(args.rgb_bins)),
        "lab": dict(bins=tuple(args.lab_bins)),
    }[args.descriptor]

    q_folder = query_folder(args.query_set)
    q_name = Path(args.query_set).name.split("_")[0].lower()

    # Same descriptor function and parameters for the database and the queries.
    db_ids, db_desc = load_or_compute("bbdd", BBDD_DIR, args.descriptor, params)
    q_ids, q_desc = load_or_compute(q_name, q_folder, args.descriptor, params)

    results = retrieve(q_desc, db_desc, db_ids, args.measure, args.k,
                       channel_sizes=channel_sizes(args.descriptor, **params))
    print(f"{q_name}: {len(results)} queries, {args.descriptor} {params}, {args.measure}, k={args.k}")
    print(f"First 3 results: {results[:3]}")

    if not args.no_gt:
        gt = load_gt(q_folder)
        assert len(gt) == len(results), f"{len(gt)} GT entries but {len(results)} queries"
        for k in (1, 5):
            print(f"mAP@{k}: {mapk(gt, results, k):.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
