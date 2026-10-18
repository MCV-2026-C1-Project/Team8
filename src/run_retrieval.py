"""Top-K retrieval using legacy CLI options or a complete exported configuration."""
import argparse
from pathlib import Path

import cv2

from descriptor_cli import add_descriptor_arguments, config_from_args, load_descriptor_document
from distances import MEASURES
from metrics import mapk
from retrieval import (BBDD_DIR, DESC_DIR, load_gt, query_folder, query_masks, retrieve_config,
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
    ap.add_argument("--remove-background", action="store_true",
                    help="Compute query descriptors only on foreground pixels")
    ap.add_argument("--mask-source", choices=("pred", "gt"), default="pred",
                    help="pred: foreground_mask; gt: <stem>.png next to each query")
    ap.add_argument("--trim-mode", choices=("block", "bbox"), default="block",
                    help="block: delete background block rows/columns; bbox: crop to the foreground box")
    ap.add_argument("--no-trim", action="store_true",
                    help="Keep the whole image instead of removing the background around the painting")
    ap.add_argument("--save-masks", type=Path, help="Write the query masks as <stem>.png")
    args = ap.parse_args(argv)
    try:
        base, saved_measure = load_descriptor_document(args.config) if args.config else (None, None)
        config = config_from_args(args, base)
        measure = args.measure or saved_measure
        if measure is None:
            raise ValueError("Specify --measure or use an exported configuration with a measure")
        folder = args.query_dir if args.query_dir is not None else query_folder(args.query_set)
        if (args.no_trim or args.save_masks or args.mask_source != "pred") and not args.remove_background:
            raise ValueError("--mask-source, --no-trim and --save-masks require --remove-background")
        masks = None
        if args.remove_background:
            paths, ids, images, query_mask_list = query_masks(folder, args.mask_source)
            masks = (ids, images, query_mask_list)
            if args.save_masks:
                args.save_masks.mkdir(parents=True, exist_ok=True)
                for path, mask in zip(paths, query_mask_list):
                    cv2.imwrite(str(args.save_masks / f"{path.stem}.png"), mask)
        q_ids, results = retrieve_config(config, measure, args.db_dir, folder, args.cache_dir, args.k,
                                         allow_truncate=args.allow_truncate, masks=masks,
                                         trim=None if args.no_trim else args.trim_mode)
        mode = (f"foreground ({args.mask_source} mask, {'no trim' if args.no_trim else args.trim_mode + ' trim'})"
                if args.remove_background else "full image")
        print(f"{folder.name}: {len(results)} queries, {config}, {measure}, k={args.k}, {mode}")
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
