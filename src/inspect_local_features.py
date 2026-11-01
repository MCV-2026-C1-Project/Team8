"""Inspect Week 4 keypoints/descriptors only: no matching, GT, rankings or scores."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from descriptors import list_images, load_image
from local_features import METHODS, create_detector, extract_local_features, resolve_local_config
from local_feature_cache import DEFAULT_CACHE_DIR, load_or_compute_database_features
from local_feature_queries import prepare_painting_crops
from painting_detection import resolve_detection_config
from retrieval import BBDD_DIR, ROOT


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def feature_summary(features):
    return dict(keypoints=len(features.keypoints), descriptor_shape=list(features.descriptors.shape),
                dtype=str(features.descriptors.dtype), descriptor_type=features.descriptor_type,
                norm="L2" if features.norm == cv2.NORM_L2 else "HAMMING",
                image_shape=list(features.image_shape), coordinate_frame="native input pixels")


def save_visualization(path, image, features):
    drawn = cv2.drawKeypoints(image, features.to_cv_keypoints(), None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ok, encoded = cv2.imencode(".png", drawn)
    if not ok:
        raise OSError(f"Could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--config", type=Path, help="JSON mapping method names to supported parameter overrides")
    parser.add_argument("--db-dir", type=Path, default=BBDD_DIR)
    parser.add_argument("--query-dir", type=Path, help="Optional JPEG query photos; annotations are never read")
    parser.add_argument("--segmentation-config", type=Path, default=ROOT / "configs/week3/task3_selected.json")
    parser.add_argument("--crop-inset", type=float, default=.03)
    parser.add_argument("--output", type=Path, required=True, help="New directory for summary.json and rich-keypoint PNGs")
    parser.add_argument("--query-limit", type=_positive, help="Inspect only the first N query images in filename order")
    parser.add_argument("--db-limit", type=_positive, help="Debug first N database images WITHOUT reading/writing caches")
    parser.add_argument("--cache-db", action="store_true", help="Compute/reuse complete database caches; incompatible with --db-limit")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--visualize-db", type=_positive, default=3, help="Number of representative DB images to draw (default 3)")
    args = parser.parse_args(argv)
    try:
        if args.cache_db and args.db_limit is not None:
            raise ValueError("--db-limit is debugging only; omit it when using --cache-db")
        if args.query_limit is not None and args.query_dir is None:
            raise ValueError("--query-limit requires --query-dir")
        if not np.isfinite(args.crop_inset) or not 0 <= args.crop_inset < .5:
            raise ValueError("--crop-inset must lie in [0, 0.5)")
        supplied = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if not isinstance(supplied, dict) or set(supplied) - set(METHODS):
            raise ValueError("Feature config must map sift, orb and/or akaze to parameter dictionaries")
        for method, params in supplied.items():
            resolve_local_config(method, params)
        methods = METHODS if args.method == "all" else (args.method,)
        configs = {m: resolve_local_config(m, supplied.get(m))["params"] for m in methods}
        for method, params in configs.items():
            create_detector(method, **params)  # Fail early on unavailable OpenCV capabilities.
        paths, ids = list_images(args.db_dir)
        if not paths:
            raise ValueError("Database contains no JPEG images")
        total_database = len(paths)
        paths, ids = paths[:args.db_limit], ids[:args.db_limit]
        query_paths, query_ids = list_images(args.query_dir) if args.query_dir else ([], [])
        if args.query_dir is not None and not query_paths:
            raise ValueError("Query directory contains no JPEG images")
        query_paths, query_ids = query_paths[:args.query_limit], query_ids[:args.query_limit]
        segmentation = (resolve_detection_config(json.loads(args.segmentation_config.read_text(encoding="utf-8")))
                        if args.query_dir else None)
        args.output.mkdir(parents=True, exist_ok=False)
        report = dict(opencv_version=cv2.__version__, configs=configs, db_dir=str(args.db_dir.resolve()),
                      query_dir=str(args.query_dir.resolve()) if args.query_dir else None,
                      segmentation=segmentation, crop_inset=args.crop_inset, query_filter={"name": "none"},
                      database_images_available=total_database, database_images_processed=len(paths),
                      query_images_processed=len(query_paths), painting_crops=0, methods={}, queries=[], warnings=[])
        for method, params in configs.items():
            print(f"{method}: {json.dumps(params, sort_keys=True)}")
            if args.cache_db:
                database = load_or_compute_database_features(args.db_dir, method, args.cache_dir, **params)
                features = database.features
                cache = dict(status=database.cache_status, reason=database.cache_reason, path=str(database.cache_path))
            else:
                features = [extract_local_features(load_image(p), method, **params) for p in paths]
                cache = dict(status="bypassed", reason="uncached inspection; no subset is saved", path=None)
            rows = []
            for index, (path, image_id, feature) in enumerate(zip(paths, ids, features)):
                row = dict(image=path.name, image_id=int(image_id), **feature_summary(feature))
                print(f"  DB {path.name}: {len(feature.keypoints)} keypoints, {feature.descriptors.shape}, {feature.descriptors.dtype}")
                if not len(feature.keypoints):
                    report["warnings"].append(f"{method}: no keypoints in DB {path.name}")
                if index < args.visualize_db:
                    target = Path(method) / "database" / f"{path.stem}.png"
                    save_visualization(args.output / target, load_image(path), feature)
                    row["visualization"] = target.as_posix()
                rows.append(row)
            report["methods"][method] = dict(database=rows, cache=cache,
                                               database_descriptors=sum(len(f.keypoints) for f in features),
                                               query_descriptors=0)
            print(f"  Cache: {cache['status']} ({cache['reason']}); {len(features)} database images")
            del features  # Do not hold all three complete databases in memory together.
            if args.cache_db:
                del database
        for path, query_id in zip(query_paths, query_ids):
            image = load_image(path)
            crops = prepare_painting_crops(image, query_id, segmentation, crop_inset=args.crop_inset)
            row = dict(image=path.name, query_id=int(query_id), image_shape=list(image.shape[:2]),
                       detected_paintings=len(crops), paintings=[])
            report["painting_crops"] += len(crops)
            if not crops:
                report["warnings"].append(f"No paintings detected in query {path.name}")
            for crop in crops:
                painting = dict(painting_index=crop.painting_index, detection_bbox=list(crop.detection_bbox),
                                crop_bbox=list(crop.crop_bbox), methods={})
                for method, params in configs.items():
                    feature = crop.extract(method, **params).features
                    target = Path(method) / "queries" / f"{path.stem}_painting_{crop.painting_index}.png"
                    save_visualization(args.output / target, crop.image, feature)
                    painting["methods"][method] = dict(**feature_summary(feature), visualization=target.as_posix())
                    report["methods"][method]["query_descriptors"] += len(feature.keypoints)
                    print(f"  {method} query {path.name}/{crop.painting_index}: {len(feature.keypoints)} keypoints, "
                          f"{feature.descriptors.shape}, {feature.descriptors.dtype}")
                    if not len(feature.keypoints):
                        report["warnings"].append(f"{method}: no keypoints in query {path.name}/{crop.painting_index}")
                row["paintings"].append(painting)
            report["queries"].append(row)
        for method, info in report["methods"].items():
            print(f"{method} totals: DB={info['database_descriptors']}, query={info['query_descriptors']} descriptors")
        print(f"Processed {len(paths)} DB images, {len(query_paths)} query images, {report['painting_crops']} painting crops")
        for warning in report["warnings"]:
            print(f"Warning: {warning}")
        with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Diagnostics: {args.output.resolve()} (summary.json and method/database|queries/*.png)")
        return report
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
