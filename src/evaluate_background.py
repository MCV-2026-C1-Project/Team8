import argparse
import csv
import inspect
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from remove_background import foreground_mask
from painting_detection import Painting, detect_paintings, resolve_detection_config


def evaluate_mask(predicted, ground_truth):
    """Evaluate binary masks, treating nonzero pixels as foreground."""
    pred = predicted > 0
    gt = ground_truth > 0

    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_image(path):
    # Supports paths containing non-ASCII characters.
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image



def resolve_evaluation_config(config):
    if not isinstance(config, dict) or set(config) - {"mode", "segmentation", "instances"}:
        raise ValueError("Configuration accepts mode, segmentation and instances")
    mode = config.get("mode", "instances")
    if mode == "instances":
        return dict(mode=mode, **resolve_detection_config({k: v for k, v in config.items() if k != "mode"}))
    if mode != "legacy" or config.get("instances"):
        raise ValueError("mode must be legacy or instances; legacy has no instance settings")
    params = {k: v.default for k, v in inspect.signature(foreground_mask).parameters.items() if k != "image"}
    params["color_sigma"] = 3.0  # Preserve the evaluator's existing CLI default.
    supplied = config.get("segmentation", {})
    if not isinstance(supplied, dict) or set(supplied) - set(params):
        raise ValueError("Unknown segmentation settings")
    params.update(supplied)
    # Validate raw model options through the shared resolver, without changing
    # legacy filling/cleanup settings or running inference during a dry run.
    raw = dict(params, fill_interior=False, min_row_foreground=0, min_col_foreground=0)
    resolve_detection_config(dict(segmentation=raw))
    for key in ("fill_interior", "fill_rows", "fill_columns"):
        if not isinstance(params[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("min_component_ratio", "min_row_foreground", "min_col_foreground"):
        if not isinstance(params[key], (int, float)) or not np.isfinite(params[key]) or not 0 <= params[key] <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    size = params["closing_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or (size > 1 and size % 2 == 0):
        raise ValueError("closing_size must be 0, 1 or a positive odd integer")
    return dict(mode=mode, segmentation=params)


def load_instance_annotations(path, image_paths):
    """Explicit opt-in only: course frames pickle or filename -> polygons JSON.

    Course frames are positional [angle, quadrilateral] lists. Reject ambiguous
    file order/subsets rather than silently associating annotations incorrectly.
    Binary union-mask connected components are never treated as GT instances.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Instance JSON must map image filenames to polygon lists")
    else:
        # Only load trusted local course annotation files (pickle is executable).
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        ids = [re.search(r"(\d+)$", p.stem) for p in image_paths]
        if (not isinstance(frames, (list, tuple)) or len(frames) != len(image_paths)
                or any(m is None for m in ids) or [int(m[1]) for m in ids] != list(range(len(frames)))):
            raise ValueError("Positional frames require all images in numeric 0..N-1 order; use filename-keyed JSON for subsets")
        document = {}
        for image, entries in zip(image_paths, frames):
            if not isinstance(entries, (list, tuple)) or any(not isinstance(e, (list, tuple)) or len(e) != 2 for e in entries):
                raise ValueError("Expected frames entries [angle, polygon]")
            document[image.name] = [entry[1] for entry in entries]
    for name, polygons in document.items():
        if not isinstance(polygons, list):
            raise ValueError(f"Expected a polygon list for {name}")
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                    or not np.isfinite(points).all()):
                raise ValueError(f"Invalid polygon for {name}")
    return document


def polygon_masks(polygons, shape):
    masks = []
    h, w = shape
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            raise ValueError("GT polygon is outside image bounds")
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        if not mask.any():
            raise ValueError("GT polygon is empty")
        masks.append(mask)
    return masks


def instance_diagnostics(predictions, truth, threshold=0.5, *, return_matches=False):
    """Exact one-to-one maximum-total-IoU matching, with unmatched GT scored 0.

    Bitmask dynamic programming is small for the one/two-painting task and does
    not assume any annotation ordering. Frame polygons are approximate masks.
    """
    states = {0: (0.0, [])}
    for predicted_index, pred in enumerate(predictions):
        overlaps = []
        for gt in truth:
            union = np.count_nonzero((pred > 0) | (gt > 0))
            overlaps.append(float(np.count_nonzero((pred > 0) & (gt > 0)) / union) if union else 0.0)
        updated = dict(states)
        for used, (score, values) in states.items():
            for index, iou in enumerate(overlaps):
                bit = 1 << index
                if not used & bit and score + iou > updated.get(used | bit, (-1, []))[0]:
                    updated[used | bit] = (score + iou, values + [(index, predicted_index, iou)])
        states = updated
    score, values = max(states.values(), key=lambda item: item[0])
    tp = int(sum(iou >= threshold for _, _, iou in values))
    result = dict(gt_count=len(truth), count_correct=int(len(predictions) == len(truth)),
                mean_instance_iou=score / len(truth) if truth else 0.0,
                instance_tp=tp, instance_precision=tp / len(predictions) if predictions else 0.0,
                instance_recall=tp / len(truth) if truth else 0.0)
    if return_matches:
        result.update(matches=[None] * len(truth), ious=[0.0] * len(truth))
        for gt_index, predicted_index, iou in values:
            result["ious"][gt_index] = iou
            if iou >= threshold:
                result["matches"][gt_index] = predicted_index
    return result


def _legacy_instances(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    paintings = []
    for label in range(1, count):
        x, y, w, h, _ = map(int, stats[label])
        paintings.append(Painting((x, y, x+w, y+h), (labels == label).astype(np.uint8) * 255, 0))
    paintings.sort(key=lambda p: (p.bbox[0], p.bbox[1]))
    for index, painting in enumerate(paintings):
        painting.order = index
    return paintings


def _write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def _write_json(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate_configuration(image_paths, gt_dir, output, config, annotations=None,
                           visualizations=False, match_iou=0.5, worst=5):
    """Evaluate one resolved configuration; inference never receives annotations."""
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    rows, predictions = [], {}
    for path in tqdm(image_paths):
        image = load_image(path)
        if config["mode"] == "instances":
            mask, paintings = detect_paintings(image, {k: v for k, v in config.items() if k != "mode"})
        else:
            mask = foreground_mask(image, **config["segmentation"])
            paintings = _legacy_instances(mask)
        row = dict(image=path.name, num_paintings=len(paintings))
        if gt_dir is not None:
            candidates = [gt_dir / (path.stem + ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif")]
            gt_path = next((p for p in candidates if p.is_file() and p.resolve() != path.resolve()), None)
            if gt_path is None:
                print(f"No union GT for {path.name}; saving prediction without pixel metrics")
            else:
                gt = cv2.imdecode(np.fromfile(str(gt_path), np.uint8), cv2.IMREAD_GRAYSCALE)
                if gt is None or gt.shape != mask.shape:
                    raise ValueError(f"Unreadable GT or image/mask size mismatch: {path.name}")
                row.update(evaluate_mask(mask, gt))
        if annotations is not None and path.name in annotations:
            truth = polygon_masks(annotations[path.name], mask.shape)
            row.update(instance_diagnostics([p.mask for p in paintings], truth, match_iou))
        _write_png(output / "masks" / f"{path.stem}.png", mask)
        predictions[path.name] = []
        for painting in paintings:
            filename = f"instance_masks/{path.stem}_{painting.order}.png"
            _write_png(output / filename, painting.mask)
            predictions[path.name].append(dict(order=painting.order, bbox=painting.bbox, mask=filename))
        if visualizations:
            canvas = image.copy()
            overlay = canvas.copy()
            overlay[mask > 0] = (40, 210, 40)
            canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0)
            for painting in paintings:
                x, y, right, bottom = painting.bbox
                cv2.rectangle(canvas, (x, y), (right-1, bottom-1), (0, 0, 255), 2)
                cv2.putText(canvas, str(painting.order + 1), (x, max(y, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _write_png(output / "visualizations" / f"{path.stem}.png", canvas)
        rows.append(row)
    fields = ["image", "precision", "recall", "f1", "tp", "fp", "fn", "num_paintings",
              "gt_count", "count_correct", "mean_instance_iou", "instance_tp", "instance_precision", "instance_recall"]
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pixel_rows = [r for r in rows if "f1" in r]
    instance_rows = [r for r in rows if "gt_count" in r]
    summary = dict(images=len(rows), pixel_evaluated=len(pixel_rows), instance_evaluated=len(instance_rows),
                   count_method="detected_instances" if config["mode"] == "instances" else "legacy_connected_components",
                   instance_gt="frame_polygons" if annotations is not None else None, match_iou=match_iou,
                   mean_paintings=float(np.mean([r["num_paintings"] for r in rows])),
                   worst_images=sorted(pixel_rows, key=lambda r: r["f1"])[:worst])
    for key in ("precision", "recall", "f1"):
        summary[key] = float(np.mean([r[key] for r in pixel_rows])) if pixel_rows else None
    for source, target in (("count_correct", "count_accuracy"), ("mean_instance_iou", "mean_instance_iou"),
                           ("instance_precision", "instance_precision"), ("instance_recall", "instance_recall")):
        summary[target] = float(np.mean([r[source] for r in instance_rows])) if instance_rows else None
    _write_json(output / "predictions.json", predictions)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate legacy wall masks or one/two painting instances")
    parser.add_argument("--images", type=Path, help="JPEG directory; required except for --dry-run")
    parser.add_argument("--gt", type=Path, help="Optional union-mask directory; omission generates masks only")
    parser.add_argument("--gt-frames", type=Path, help="Trusted frames.pkl or filename-keyed polygon JSON; evaluation only")
    parser.add_argument("--output", type=Path, default=Path("results/background_masks"))
    parser.add_argument("--config", type=Path, help="Single configuration or {experiments: [{id, config}]} suite")
    parser.add_argument("--instances", action="store_true", help="Use Task 3 detection (default without config is legacy)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visualizations", action="store_true")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--grid-search", action="store_true")
    for key in ("threshold", "border-ratio", "mad-multiplier", "l-weight", "color-sigma",
                "min-component-ratio", "min-row-foreground", "min-col-foreground",
                "dark-percentile", "dark-margin", "dark-mad-multiplier", "dark-mad-floor"):
        parser.add_argument("--" + key, type=float, default=argparse.SUPPRESS)
    for key in ("sample-step", "closing-size"):
        parser.add_argument("--" + key, type=int, default=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("quadratic", "color_distribution"), default=argparse.SUPPRESS)
    parser.add_argument("--dark-mode", choices=("legacy", "percentile", "robust", "off"), default=argparse.SUPPRESS)
    for flag in ("no-fill-interior", "enclosed-only", "no-fill-columns"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--refinement", choices=("none", "holes", "spans", "hull"))
    parser.add_argument("--group-gap", type=float)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.match_iou <= 1 or args.worst < 0:
            raise ValueError("match-iou must be in (0, 1] and worst must be nonnegative")
        document = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {"mode": "instances" if args.instances else "legacy"}
        suite = isinstance(document, dict) and "experiments" in document
        if suite and (set(document) != {"experiments"} or not isinstance(document["experiments"], list) or not document["experiments"]):
            raise ValueError("Suite requires only a nonempty experiments list")
        entries = document["experiments"] if suite else [dict(id="run", config=document)]
        if args.grid_search and (suite or args.instances):
            raise ValueError("Use a configuration suite for instance sweeps; --grid-search is a legacy option")
        jobs = []
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {"id", "config"}
                    or not isinstance(entry["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"])):
                raise ValueError("Each experiment requires a safe id and config")
            if not isinstance(entry["config"], dict):
                raise ValueError("Each config must be a dictionary")
            config = dict(entry["config"])
            if args.instances:
                config["mode"] = "instances"
            if not isinstance(config.get("segmentation", {}), dict):
                raise ValueError("segmentation must be a dictionary")
            seg = dict(config.get("segmentation", {}))
            for key in inspect.signature(foreground_mask).parameters:
                if hasattr(args, key):
                    seg[key] = getattr(args, key)
            if args.no_fill_interior:
                seg["fill_interior"] = False
            if args.enclosed_only:
                seg.update(fill_rows=False, fill_columns=False)
            if args.no_fill_columns:
                seg["fill_columns"] = False
            config["segmentation"] = seg
            if args.refinement is not None or args.group_gap is not None:
                inst = dict(config.get("instances", {}))
                inst.update({k: getattr(args, k) for k in ("refinement", "group_gap") if getattr(args, k) is not None})
                config["instances"] = inst
            config = resolve_evaluation_config(config)
            if args.grid_search:
                if config["mode"] != "legacy":
                    raise ValueError("--grid-search requires legacy mode")
                key = "color_sigma" if config["segmentation"]["method"] == "color_distribution" else "threshold"
                values = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6] if key == "color_sigma" else [10, 15, 20, 25]
                for value in values:
                    jobs.append((f"threshold_{value}", dict(config, segmentation=dict(config["segmentation"], **{key: value}))))
            else:
                jobs.append((entry["id"], config))
        if len({name for name, _ in jobs}) != len(jobs) or len(jobs) > 50:
            raise ValueError("Use unique experiment IDs and at most 50 configurations")
        if args.dry_run:
            print(json.dumps([dict(id=name, config=config) for name, config in jobs], indent=2, allow_nan=False))
            print(f"{len(jobs)} segmentation configurations; no images read")
            return
        if args.images is None:
            raise ValueError("--images is required unless --dry-run is used")
        paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        if not paths:
            raise ValueError(f"No JPEG images in {args.images}")
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError("Duplicate image stems would overwrite mask outputs")
        if args.gt is not None and not args.gt.is_dir():
            raise ValueError("--gt must be an existing directory")
        annotations = load_instance_annotations(args.gt_frames, paths) if args.gt_frames else None
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"{args.output} is not empty; choose a new output directory")
        args.output.mkdir(parents=True, exist_ok=True)
        comparison = []
        for name, config in jobs:
            target = args.output / name if suite or args.grid_search else args.output
            summary = evaluate_configuration(paths, args.gt, target, config, annotations,
                                               args.visualizations, args.match_iou, args.worst)
            key = "color_sigma" if config["segmentation"]["method"] == "color_distribution" else "threshold"
            comparison.append(dict(id=name, threshold=config["segmentation"][key],
                                   **{k: summary[k] for k in ("precision", "recall", "f1")}))
        if suite or args.grid_search:
            comparison.sort(key=lambda r: -(r["f1"] if r["f1"] is not None else -1))
            path = args.output / ("threshold_comparison.csv" if args.grid_search else "comparison.csv")
            fields = ["threshold", "precision", "recall", "f1"] if args.grid_search else list(comparison[0])
            with path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(comparison)
        return comparison
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
