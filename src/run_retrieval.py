"""Top-K retrieval using legacy CLI options or a complete exported configuration."""
import argparse
import json
import pickle
from pathlib import Path

from descriptor_cli import add_descriptor_arguments, config_from_args, load_descriptor_document
from filter_noise import FILTER_NAMES, query_preprocessor, resolve_filter_config
from distances import MEASURES
from metrics import mapk, normalize_painting_gt, evaluate_painting_retrieval, validate_predictions
from descriptors import list_images
from painting_detection import resolve_detection_config
from evaluate_background import load_instance_annotations, polygon_masks, instance_diagnostics
from retrieval import (BBDD_DIR, DESC_DIR, load_gt, query_folder, query_masks, retrieve_config,
                       save_results, validate_gt_order, retrieve_multi_config, save_query_mask)


def noise_threshold_argument(value):
    """Allow an explicit reset of a saved selective policy to unconditional."""
    if value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("noise threshold must be a number or 'none'") from None


def query_filter_from_args(args, saved=None):
    """Explicit flags override saved values; a new filter type starts from defaults."""
    config = resolve_filter_config(saved)
    if args.noise_filter is not None and args.noise_filter != config["name"]:
        config = {"name": args.noise_filter}
    for key in ("ksize", "sigma", "cutoff", "notch", "radius", "noise_threshold"):
        attribute = "filter_" + key
        if hasattr(args, attribute):
            config["notches" if key == "notch" else key] = getattr(args, attribute)
    return resolve_filter_config(config)


def write_report(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


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
    ap.add_argument("--output", type=Path, help="Save integer-ID rankings as pickle (nested for Week 3 options)")
    ap.add_argument("--multi-painting", action="store_true", help="Detect paintings and rank original-image crops")
    ap.add_argument("--segmentation-config", type=Path, help="Independent Task 3 detection configuration")
    ap.add_argument("--nested-output", action="store_true", help="Week 3 list level for single-painting queries, without segmentation")
    ap.add_argument("--gt-frames", type=Path, help="Trusted frame annotations aligned with painting GT, for evaluation only")
    ap.add_argument("--match-iou", type=float, default=None, help="Development geometric-match threshold (default 0.5)")
    ap.add_argument("--diagnostics", type=Path, help="Save per-query/crop decisions and evaluation as JSON")
    ap.add_argument("--manifest", type=Path, help="Run metadata; Week 3 output defaults to <output>.manifest.json")
    ap.add_argument("--remove-background", action="store_true",
                    help="Compute query descriptors only on foreground pixels")
    ap.add_argument("--mask-source", choices=("pred", "gt"), default=None,
                    help="pred: foreground_mask; gt: <stem>.png next to each query")
    ap.add_argument("--trim-mode", choices=("block", "bbox"), default=None,
                    help="block: delete background block rows/columns; bbox: crop to the foreground box")
    ap.add_argument("--no-trim", action="store_true",
                    help="Keep the whole image instead of removing the background around the painting")
    ap.add_argument("--save-masks", type=Path, help="Write the query masks as <stem>.png")
    ap.add_argument("--crop-inset", type=float, default=None,
                    help="Shrink each detected box by this fraction per side (overrides the exported value)")
    ap.add_argument("--noise-filter", choices=FILTER_NAMES, default=None)
    ap.add_argument("--filter-ksize", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--filter-sigma", type=float, default=argparse.SUPPRESS)
    ap.add_argument("--filter-cutoff", type=float, default=argparse.SUPPRESS)
    ap.add_argument("--filter-notch", type=float, nargs=2, action="append", default=argparse.SUPPRESS,
                    metavar=("DY", "DX"))
    ap.add_argument("--filter-radius", type=float, default=argparse.SUPPRESS)
    ap.add_argument("--filter-noise-threshold", type=noise_threshold_argument, default=argparse.SUPPRESS,
                    metavar="VALUE|none", help="Filter only above this noise score; none filters every query")
    args = ap.parse_args(argv)
    try:
        if args.multi_painting and args.remove_background:
            raise ValueError("--multi-painting cannot be combined with legacy --remove-background")
        if args.multi_painting != (args.segmentation_config is not None):
            raise ValueError("--multi-painting and --segmentation-config must be supplied together")
        if args.multi_painting and args.nested_output:
            raise ValueError("--multi-painting already produces nested output; omit --nested-output")
        if (args.mask_source is not None or args.trim_mode is not None or args.no_trim) and not args.remove_background:
            raise ValueError("--mask-source, --trim-mode and --no-trim require legacy --remove-background")
        if args.no_trim and args.trim_mode is not None:
            raise ValueError("Choose --no-trim or --trim-mode, not both")
        if args.save_masks and not (args.multi_painting or args.remove_background):
            raise ValueError("--save-masks requires --multi-painting or --remove-background")
        if args.no_gt and (args.gt is not None or args.gt_frames is not None):
            raise ValueError("--no-gt cannot be combined with --gt or --gt-frames")
        if args.gt_frames is not None and not args.multi_painting:
            raise ValueError("--gt-frames is only used for --multi-painting evaluation")
        if args.match_iou is not None and (args.gt_frames is None or not 0 < args.match_iou <= 1):
            raise ValueError("--match-iou requires --gt-frames and a value in (0, 1]")
        if args.multi_painting and not args.no_gt and args.gt_frames is None:
            raise ValueError("Multi-painting evaluation requires --gt-frames aligned with correspondence order; use --no-gt for inference only")
        args.mask_source = args.mask_source or "pred"
        args.trim_mode = args.trim_mode or "block"
        base, saved_measure = load_descriptor_document(args.config) if args.config else (None, None)
        config = config_from_args(args, base)
        measure = args.measure or saved_measure
        if measure is None:
            raise ValueError("Specify --measure or use an exported configuration with a measure")
        folder = args.query_dir if args.query_dir is not None else query_folder(args.query_set)
        saved_filter, crop_inset, fusion = None, 0.0, None
        if args.config:
            with args.config.open(encoding="utf-8") as stream:
                provenance = json.load(stream).get("provenance", {})
            if not isinstance(provenance, dict):
                raise ValueError("Exported provenance must be a dictionary")
            saved_filter = provenance.get("query_filter")
            # Written by search_multi.py: replay crop inset and colour/texture fusion.
            crop_inset = float(provenance.get("crop_inset") or 0.0)
            fusion = provenance.get("fusion")
        if args.crop_inset is not None:
            crop_inset = args.crop_inset
        if (crop_inset or fusion) and not args.multi_painting:
            raise ValueError("Crop inset and fusion apply only to --multi-painting retrieval")
        filter_config = query_filter_from_args(args, saved_filter)
        query_preprocess = query_preprocessor(filter_config)
        if query_preprocess is not None and args.remove_background:
            raise ValueError("--noise-filter cannot be combined with --remove-background")
        nested = args.multi_painting or args.nested_output
        paths, expected_ids = list_images(folder)
        if not paths:
            raise ValueError(f"No JPEG images found in {folder}")
        manifest_path = args.manifest or (args.output.with_suffix(".manifest.json") if nested and args.output else None)
        targets = [p for p in (args.output, args.diagnostics, manifest_path) if p is not None]
        if args.save_masks:
            targets.extend(args.save_masks / f"{p.stem}.png" for p in paths)
        if len({p.resolve() for p in targets}) != len(targets):
            raise ValueError("Result, mask and report outputs must use distinct paths")
        for target in targets:
            if target.exists():
                raise ValueError(f"{target} already exists; choose a new output path")
        segmentation = None
        annotations, multi_gt = None, None
        records, matches, geometry = [], [], []
        evaluation = None
        if args.multi_painting:
            segmentation = resolve_detection_config(json.loads(args.segmentation_config.read_text(encoding="utf-8")))
            if not args.no_gt:
                with (args.gt or folder / "gt_corresps.pkl").open("rb") as stream:
                    multi_gt = normalize_painting_gt(pickle.load(stream))
                if expected_ids != list(range(len(multi_gt))):
                    raise ValueError("Positional correspondence GT requires query IDs in 0..N-1 order")
                annotations = load_instance_annotations(args.gt_frames, paths)
                for path, truth in zip(paths, multi_gt):
                    if path.name not in annotations or len(annotations[path.name]) != len(truth):
                        raise ValueError(f"Frame/correspondence counts disagree for {path.name}; supply polygons in correspondence order")

            def observe_segmentation(path, mask, paintings):
                if annotations is not None:
                    truth = polygon_masks(annotations[path.name], mask.shape)
                    info = instance_diagnostics([p.mask for p in paintings], truth,
                                                args.match_iou if args.match_iou is not None else 0.5,
                                                return_matches=True)
                    matches.append(info["matches"])
                    geometry.append(info)
                if args.save_masks:
                    save_query_mask(args.save_masks / f"{path.stem}.png", mask)

            q_ids, results = retrieve_multi_config(
                config, measure, segmentation, args.db_dir, folder, args.cache_dir, args.k,
                allow_truncate=args.allow_truncate, query_filter=filter_config,
                diagnostics=records, on_segmented=observe_segmentation,
                crop_inset=crop_inset, fusion=fusion)
            print("Detected paintings: " + ", ".join(f"{r['image']}={r['detected']}" for r in records))
            for record in records:
                if record["failure"]:
                    print(f"Detection failure: {record['image']}: {record['failure']}")
            if multi_gt is not None:
                cutoffs = tuple(k for k in (1, 5) if args.k >= k)
                evaluation = evaluate_painting_retrieval(multi_gt, results, matches=matches, ks=cutoffs)
                for k in cutoffs:
                    print(f"Painting mAP@{k}: {evaluation[f'map{k}']:.4f}")
                print(f"Painting-count accuracy: {evaluation['count_accuracy']:.4f}; "
                      f"missing: {evaluation['missing']}; extra: {evaluation['extra']}")
                evaluation["geometry"] = geometry
                evaluation["matching"] = "maximum-total polygon IoU, thresholded; labels never used for assignment"
                evaluation["match_iou"] = args.match_iou if args.match_iou is not None else 0.5
        else:
            masks = None
            if args.remove_background:
                paths, ids, images, query_mask_list = query_masks(folder, args.mask_source)
                masks = (ids, images, query_mask_list)
                if args.save_masks:
                    for path, mask in zip(paths, query_mask_list):
                        save_query_mask(args.save_masks / f"{path.stem}.png", mask)
            q_ids, results = retrieve_config(config, measure, args.db_dir, folder, args.cache_dir, args.k,
                                             allow_truncate=args.allow_truncate, masks=masks,
                                             trim=None if args.no_trim else args.trim_mode,
                                             query_preprocess=query_preprocess)
            if not args.no_gt:
                gt = load_gt(folder, args.gt)
                validate_gt_order(q_ids, gt)
                evaluation = {f"map{k}": float(mapk(gt, results, k)) for k in (1, 5) if args.k >= k}
                for k in (1, 5):
                    if args.k >= k:
                        print(f"mAP@{k}: {evaluation[f'map{k}']:.4f}")
            records = [dict(image=p.name, query_id=int(qid), rankings=[ranking])
                       for p, qid, ranking in zip(paths, q_ids, results)]
            if args.nested_output:
                results = [[ranking] for ranking in results]
        if q_ids.tolist() != expected_ids:
            raise ValueError("Retrieval changed the query image order")
        db_ids = list_images(args.db_dir)[1]
        effective_k = min(args.k, len(db_ids)) if args.allow_truncate else args.k
        counts = [r["detected"] for r in records] if args.multi_painting else [1] * len(paths)
        results = validate_predictions(results, nested=nested, k=effective_k,
                                       query_count=len(paths), painting_counts=counts, db_ids=db_ids)
        mode = ("painting crops" if args.multi_painting else
                f"foreground ({args.mask_source} mask, {'no trim' if args.no_trim else args.trim_mode + ' trim'})"
                if args.remove_background else "full image")
        print(f"{folder.name}: {len(results)} queries, {config}, {measure}, k={args.k}, {mode}")
        print(f"Query noise filter: {filter_config['name']}")
        print(f"First 3 results: {results[:3]}")
        manifest = dict(config=config, measure=measure, segmentation=segmentation, query_filter=filter_config,
                        crop_inset=crop_inset, fusion=fusion,
                        crop_strategy="original_rectangular_bbox" if args.multi_painting else
                                      "legacy_foreground_mask" if args.remove_background else "whole_image",
                        format="nested" if nested else "flat", requested_k=args.k, effective_k=effective_k,
                        db_dir=str(args.db_dir), query_dir=str(folder),
                        gt=None if args.no_gt else str(args.gt or folder / "gt_corresps.pkl"),
                        gt_frames=str(args.gt_frames) if args.gt_frames else None,
                        match_iou=(args.match_iou if args.match_iou is not None else 0.5) if args.gt_frames else None,
                        queries=[dict(image=p.name, query_id=int(qid), paintings=count)
                                 for p, qid, count in zip(paths, q_ids, counts)],
                        failures=[r for r in records if r.get("failure")])
        if args.output:
            save_results(args.output, results, nested=nested, k=effective_k,
                         query_count=len(paths), painting_counts=counts, db_ids=db_ids)
            print(f"Saved results to {args.output}")
        if manifest_path:
            write_report(manifest_path, manifest)
        if args.diagnostics:
            write_report(args.diagnostics, dict(manifest=manifest, queries=records, evaluation=evaluation))
        return results
    except (ValueError, TypeError, OSError, KeyError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
