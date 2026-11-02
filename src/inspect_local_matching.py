"""Local retrieval inspection with optional development-only unknown rejection evaluation."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from descriptors import list_images, load_image
from local_feature_cache import DEFAULT_CACHE_DIR, load_or_compute_database_features
from local_feature_queries import prepare_painting_crops
from local_matching_config import ROOT, resolve_run_config
from local_retrieval import retrieve_local, rank_candidates, candidate_report
from painting_detection import resolve_detection_config
from retrieval import BBDD_DIR
from local_rejection import painting_evidence, predict, apply_threshold, evaluate_unknown, select_threshold
from local_development import load_development, align_boxes


def draw_candidate(path, query_image, database_image, candidate, score_mode, show_rejected=False):
    """Display-only aspect-preserving scaling; geometry and features remain native."""
    images, scales = [], []
    for image in (query_image, database_image):
        scale = min(1., 800 / max(image.shape[:2]))
        target = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
        images.append(cv2.resize(image, target))
        scales.append(np.array(target) / [image.shape[1], image.shape[0]])
    banner = 90
    canvas = np.zeros((max(i.shape[0] for i in images)+banner, sum(i.shape[1] for i in images), 3), np.uint8)
    offset = images[0].shape[1]
    canvas[banner:banner+images[0].shape[0], :offset] = images[0]
    canvas[banner:banner+images[1].shape[0], offset:] = images[1]
    v = candidate.verification
    # Cap display clutter, reserving room for both classes when rejection is shown.
    # Counts and JSON evidence always describe ALL correspondences.
    inliers = list(np.flatnonzero(v.inlier_mask))
    rejected = list(np.flatnonzero(~v.inlier_mask)) if show_rejected else []
    inlier_limit = min(len(inliers), 200-min(len(rejected), 100))
    indices = inliers[:inlier_limit] + rejected[:200-inlier_limit]
    for index in indices:
        a = tuple(np.rint(v.matches.query_points[index]*scales[0] + [0, banner]).astype(int))
        b = tuple(np.rint(v.matches.database_points[index]*scales[1] + [offset, banner]).astype(int))
        color = (0, 230, 0) if v.inlier_mask[index] else (50, 50, 245)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, color, 1)
        cv2.circle(canvas, b, 3, color, 1)
    lines = [f"DB {candidate.database_id} | tentative {candidate.tentative.candidate_count} | filtered {v.matches.count}",
             f"inliers {v.inlier_count} | {score_mode}={candidate.scores[score_mode]:.3f}",
             f"green=inlier red=rejected | drawn {len(indices)} | {v.failure_reason or 'valid homography'}"]
    # Ensure labels are readable even for tiny synthetic crop images.
    if canvas.shape[1] < 800:
        canvas = cv2.copyMakeBorder(canvas, 0, 0, 0, 800-canvas.shape[1], cv2.BORDER_CONSTANT)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 23+27*i), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise OSError(f"Cannot encode visualization {path}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week4/local_matching_baseline.json")
    parser.add_argument("--query-dir", type=Path, default=ROOT / "data/qsd1_w4")
    parser.add_argument("--db-dir", type=Path, default=BBDD_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--segmentation-config", type=Path, default=ROOT / "configs/week3/task3_selected.json")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--db-limit", type=int, help="Explicit partial-database diagnostic; full caches remain intact")
    parser.add_argument("--k", type=int)
    parser.add_argument("--method", choices=("sift", "orb", "akaze"))
    parser.add_argument("--filter", choices=("ratio", "ratio_mutual"))
    parser.add_argument("--ratio-threshold", type=float)
    parser.add_argument("--estimator", choices=("ransac", "usac_magsac"))
    parser.add_argument("--score", choices=("inlier_count", "coverage_weighted"))
    parser.add_argument("--output", type=Path, required=True, help="New diagnostics directory")
    parser.add_argument("--visualize-top", type=int, default=2, help="PNG candidates per painting, 0 disables")
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--detail", choices=("summary", "visualized", "all"), default="visualized")
    parser.add_argument("--min-inliers", type=int)
    parser.add_argument("--evaluate-dev", action="store_true", help="Explicitly load trusted QSD1-W4 labels and frames")
    parser.add_argument("--select-threshold", action="store_true", help="Select development F1 threshold; requires --evaluate-dev")
    args = parser.parse_args(argv)
    try:
        start = perf_counter()
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Run config must be a dictionary")
        if args.method:
            raw["feature_method"] = args.method
        for section, key, value in (("matching", "filter", args.filter), ("matching", "ratio_threshold", args.ratio_threshold),
                                    ("geometry", "estimator", args.estimator), ("scoring", "mode", args.score),
                                    ("retrieval", "k", args.k)):
            if value is not None:
                if raw.get(section) is None:
                    raw[section] = {}
                if not isinstance(raw[section], dict):
                    raise ValueError(f"{section} settings must be a dictionary")
                raw[section][key] = value
        if args.min_inliers is not None:
            raw["rejection"] = dict(mode="min_inliers", min_inliers=args.min_inliers)
        config = resolve_run_config(raw)
        if args.select_threshold and not args.evaluate_dev:
            raise ValueError("--select-threshold requires --evaluate-dev")
        if args.evaluate_dev and args.db_limit is not None:
            raise ValueError("Development evaluation requires the full museum database")
        if args.evaluate_dev and not args.select_threshold and config["rejection"]["mode"] == "disabled":
            raise ValueError("Development evaluation requires --min-inliers or --select-threshold")
        if any(v is not None and v < 1 for v in (args.query_limit, args.db_limit)) or args.visualize_top < 0:
            raise ValueError("Debug limits must be positive; visualize-top must be nonnegative")
        paths, ids = list_images(args.query_dir)
        db_paths, _ = list_images(args.db_dir)
        if not paths or not db_paths:
            raise ValueError("Query and database directories require JPEG images")
        candidate_count = min(len(db_paths), args.db_limit) if args.db_limit else len(db_paths)
        if config["retrieval"]["k"] > candidate_count:
            raise ValueError("Top-K exceeds inspected candidate count; reduce --k or increase --db-limit")
        labels, frames = load_development(args.query_dir, paths, ids) if args.evaluate_dev else (None, None)
        development_evidence, development_labels, all_evidence, all_paintings, alignment = [], [], [], [], []
        segmentation = resolve_detection_config(json.loads(args.segmentation_config.read_text(encoding="utf-8")))
        args.output.mkdir(parents=True, exist_ok=False)
        before = perf_counter()
        database = load_or_compute_database_features(args.db_dir, config["feature_method"], args.cache_dir, **config["feature_params"])
        report = dict(config=config, segmentation=segmentation, opencv_version=cv2.__version__,
                      database_dir=str(args.db_dir.resolve()), query_dir=str(args.query_dir.resolve()),
                      database_size=len(database.ids), inspected_candidates=candidate_count,
                      scope="partial_database_diagnostic" if candidate_count < len(database.ids) else "full_database_retrieval",
                      ranking_semantics="Ranking retained independently of optional min-inliers rejection",
                      diagnostic_detail=args.detail,
                      query_limit=args.query_limit, cache=dict(status=database.cache_status, reason=database.cache_reason,
                                                              path=str(database.cache_path)),
                      timings=dict(database_feature_loading=perf_counter()-before), queries=[], warnings=[])
        print(f"{report['scope']}: {candidate_count}/{len(database.ids)} candidates; cache {database.cache_status}", flush=True)
        print(json.dumps(config, sort_keys=True), flush=True)
        for path, query_id in zip(paths[:args.query_limit], ids[:args.query_limit]):
            before = perf_counter()
            crops = prepare_painting_crops(load_image(path), query_id, segmentation, crop_inset=config["crop"]["inset"])
            query = dict(image=path.name, query_id=int(query_id), segmentation_seconds=perf_counter()-before, paintings=[])
            if not crops:
                report["warnings"].append(f"No paintings detected in {path.name}")
            query_evidence = []
            for crop in crops:
                before = perf_counter()
                features = crop.extract(config["feature_method"], **config["feature_params"])
                extraction_time = perf_counter()-before
                evidence, timings = retrieve_local(features.features, database, matching=config["matching"],
                                                    geometry=config["geometry"], scoring=config["scoring"], database_limit=args.db_limit)
                mode = config["scoring"]["mode"]
                before = perf_counter()
                ranked = rank_candidates(evidence, mode)
                top = ranked[:config["retrieval"]["k"]]
                timings["ranking"] = perf_counter()-before
                timings["total_retrieval"] += timings["ranking"]
                painting = dict(painting_index=crop.painting_index, detection_bbox=list(crop.detection_bbox),
                                crop_bbox=list(crop.crop_bbox), keypoints=len(features.features.keypoints),
                                query_feature_extraction_seconds=extraction_time, timings=timings,
                                top_ids=[c.database_id for c in top],
                                candidates=[candidate_report(c, i+1, mode, detail=(args.detail == "all" or
                                            (args.detail == "visualized" and i < args.visualize_top))) for i, c in enumerate(ranked)])
                decision = painting_evidence(ranked, config["retrieval"]["k"])
                query_evidence.append(decision)
                all_evidence.append(decision)
                all_paintings.append(painting)
                painting["prediction"] = predict(decision, config["rejection"])
                painting["rejected"] = painting["prediction"] == [-1]
                if not any(c.verification.success for c in ranked):
                    report["warnings"].append(f"{path.name}/{crop.painting_index}: no valid homography; ranking contains zero-score candidates")
                for i, c in enumerate(ranked[:args.visualize_top]):
                    name = f"{path.stem}_painting_{crop.painting_index}_rank_{i+1}_db_{c.database_id}.png"
                    draw_candidate(args.output / name, crop.image, load_image(args.db_dir / c.filename), c, mode, args.show_rejected)
                    painting["candidates"][i]["visualization"] = name
                query["paintings"].append(painting)
                print(f"{path.name}/{crop.painting_index}: {painting['top_ids']}; inliers {[c.verification.inlier_count for c in top]}", flush=True)
            if labels is not None:
                info = align_boxes([c.detection_bbox for c in crops], frames[path.name])
                info["query_id"] = int(query_id)
                alignment.append(info)
                for gt_index, detection in enumerate(info["matches"]):
                    if detection is not None:
                        development_evidence.append(query_evidence[detection])
                        development_labels.append(labels[query_id][gt_index])
            report["queries"].append(query)
        if labels is not None:
            selection = select_threshold(development_evidence, development_labels) if args.select_threshold else None
            threshold = selection["min_inliers"] if selection else config["rejection"]["min_inliers"]
            for painting, prediction in zip(all_paintings, apply_threshold(all_evidence, threshold)):
                painting["prediction"], painting["rejected"] = prediction, prediction == [-1]
            processed_labels = [label for q in report["queries"] for label in labels[q["query_id"]]]
            report["evaluation"] = dict(
                scope="development_subset" if len(report["queries"]) < len(paths) else "full_development_set",
                metric_scope="geometrically matched paintings only; missing/extra detections excluded and counted separately",
                min_inliers=threshold, selection=selection, alignment=alignment,
                missing=sum(a["missing"] for a in alignment), extra=sum(a["extra"] for a in alignment),
                detected_accepted=sum(not p["rejected"] for p in all_paintings),
                detected_rejected=sum(p["rejected"] for p in all_paintings),
                annotated_paintings=len(processed_labels), annotated_unknown=sum(x == -1 for x in processed_labels),
                annotated_known=sum(x != -1 for x in processed_labels),
                metrics=evaluate_unknown(development_labels, apply_threshold(development_evidence, threshold)),
                provenance="QSD1-W4 development results; not independent test performance")
            report["effective_rejection"] = dict(mode="min_inliers", min_inliers=threshold)
            evaluation = report["evaluation"]
            if evaluation["missing"] or evaluation["extra"]:
                report["warnings"].append("Unknown F1 excludes unmatched paintings; inspect missing/extra detection counts")
            print(f"Development unknown F1={evaluation['metrics']['f1']:.6f}; threshold={threshold}; "
                  f"missing={evaluation['missing']}, extra={evaluation['extra']}; {evaluation['scope']}", flush=True)
        report["timings"]["total_run"] = perf_counter()-start
        for warning in report["warnings"]:
            print(f"Warning: {warning}")
        with (args.output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Diagnostics: {args.output.resolve()}")
        return report
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
