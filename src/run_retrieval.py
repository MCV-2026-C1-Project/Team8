"""Top-K retrieval using legacy CLI options or a complete exported configuration."""
import argparse
from pathlib import Path

from descriptor_cli import add_descriptor_arguments, config_from_args, load_descriptor_document
from distances import MEASURES
from metrics import mapk
from retrieval import (BBDD_DIR, DESC_DIR, load_gt, query_folder, retrieve_config,
                       save_results, validate_gt_order)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_descriptor_arguments(ap)
    ap.add_argument("--measure", choices=list(MEASURES))
    ap.add_argument("--query-set", default="qsd1", help="Dataset alias or original image folder")
    ap.add_argument("--query-dir", type=Path, help="Explicit query folder; overrides --query-set")
    ap.add_argument("--db-dir", type=Path, default=BBDD_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DESC_DIR)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--allow-truncate", action="store_true",
                    help="Explicitly allow k > database size and return all available IDs")
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--output", type=Path, help="Save a list of lists of Python integer IDs")
    args = ap.parse_args(argv)
    try:
        base, saved_measure = load_descriptor_document(args.config) if args.config else (None, None)
        config = config_from_args(args, base)
        measure = args.measure or saved_measure
        if measure is None:
            raise ValueError("Specify --measure or use an exported configuration with a measure")
        folder = args.query_dir if args.query_dir is not None else query_folder(args.query_set)
        q_ids, results = retrieve_config(config, measure, args.db_dir, folder, args.cache_dir, args.k,
                                         allow_truncate=args.allow_truncate)
        print(f"{folder.name}: {len(results)} queries, {config}, {measure}, k={args.k}")
        print(f"First 3 results: {results[:3]}")
        if not args.no_gt:
            gt = load_gt(folder, args.gt)
            validate_gt_order(q_ids, gt)
            for k in (1, 5):
                if args.k >= k:
                    print(f"mAP@{k}: {mapk(gt, results, k):.4f}")
        if args.output:
            save_results(args.output, results)
            print(f"Saved results to {args.output}")
        return results
    except (ValueError, TypeError, OSError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
