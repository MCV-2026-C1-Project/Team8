"""One/two painting instances from the existing border-conditioned wall model.

No ground truth or retrieval descriptors are used here. Boxes are (x0, y0, x1,
y1), with exclusive upper bounds, suitable for image[y0:y1, x0:x1].
"""
from dataclasses import dataclass
import inspect
from numbers import Real

import cv2
import numpy as np

from remove_background import foreground_mask, fill_painting_interior, dark_foreground_cutoff


@dataclass
class Painting:
    bbox: tuple
    mask: np.ndarray  # Original-resolution uint8, 0 or 255.
    order: int        # Zero-based reading order.


INSTANCE_DEFAULTS = dict(
    max_paintings=2, min_fragment_area=0.0002, min_area=0.003,
    min_side=0.03, min_aspect=0.1, max_aspect=10.0, min_fill=0.1,
    group_gap=0.015, group_overlap=0.6, group_area_ratio=0.4,
    group_solid_fill=0.85, group_slenderness=0.25,
    refinement="holes", closing_ratio=0.0, vertical_overlap=0.5,
)


def resolve_detection_config(config=None):
    """Resolve a JSON-compatible {segmentation, instances} configuration.

    All areas are fractions of image area; side lengths and gap/closing sizes
    are fractions of the shorter image side. Aspect means width / height.
    Grouping requires aligned boxes, a small gap, and an unequal area ratio.
    Two solid, non-slender regions are protected as plausible separate paintings;
    group_gap=0 disables it. Defaults are conservative, not dataset-tuned.
    """
    config = {} if config is None else config
    if (not isinstance(config, dict) or set(config) - {"mode", "segmentation", "instances"}
            or config.get("mode", "instances") != "instances"):
        raise ValueError("Detection config accepts segmentation and instances dictionaries")
    seg = {key: p.default for key, p in inspect.signature(foreground_mask).parameters.items()
           if key != "image"}
    seg.update(color_sigma=3.0, fill_interior=False, min_row_foreground=0.0,
               min_col_foreground=0.0, dark_mode="robust", dark_margin=40.0)
    inst = dict(INSTANCE_DEFAULTS)
    for section, target in (("segmentation", seg), ("instances", inst)):
        supplied = config.get(section, {})
        if not isinstance(supplied, dict) or set(supplied) - set(target):
            raise ValueError(f"Unknown or invalid {section} settings")
        target.update(supplied)
    if seg["fill_interior"] or seg["min_row_foreground"] != 0 or seg["min_col_foreground"] != 0:
        raise ValueError("Instance detection requires raw evidence: fill_interior=false and row/column cleanup=0")
    if seg["method"] not in {"color_distribution", "quadratic"}:
        raise ValueError("Unknown segmentation method")
    if not isinstance(seg["sample_step"], int) or isinstance(seg["sample_step"], bool) or seg["sample_step"] < 1:
        raise ValueError("sample_step must be a positive integer")
    for key in ("border_ratio", "color_sigma", "threshold", "mad_multiplier", "l_weight"):
        value = seg[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < seg["border_ratio"] <= 0.5 or seg["color_sigma"] <= 0 or seg["l_weight"] > 1:
        raise ValueError("Invalid border_ratio, color_sigma or l_weight")
    dark_foreground_cutoff(np.array([128.]), seg["dark_mode"], seg["dark_percentile"],
                           seg["dark_margin"], seg["dark_mad_multiplier"], seg["dark_mad_floor"])
    if isinstance(inst["max_paintings"], bool) or not isinstance(inst["max_paintings"], int) or inst["max_paintings"] not in (1, 2):
        raise ValueError("max_paintings must be 1 or 2")
    if inst["refinement"] not in {"none", "holes", "spans", "hull"}:
        raise ValueError("refinement must be none, holes, spans or hull")
    for key, value in inst.items():
        if key in {"max_paintings", "refinement"}:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        if key not in {"min_aspect", "max_aspect"} and value > 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    if not 0 < inst["min_aspect"] <= inst["max_aspect"]:
        raise ValueError("Require 0 < min_aspect <= max_aspect")
    return dict(segmentation=seg, instances=inst)


def _bbox(mask):
    y, x = np.nonzero(mask)
    return (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)


def _join_score(a, b, options, length):
    ax, ay, ar, ab = a["bbox"]
    bx, by, br, bb = b["bbox"]
    def complete(group):
        x, y, right, bottom = group["bbox"]
        w, h = right - x, bottom - y
        return (group["area"] / (w * h) >= options["group_solid_fill"]
                and min(w, h) / max(w, h) > options["group_slenderness"])
    if complete(a) and complete(b):
        return None
    dx, dy = max(0, ax - br, bx - ar), max(0, ay - bb, by - ab)
    ox = max(0, min(ar, br) - max(ax, bx)) / min(ar - ax, br - bx)
    oy = max(0, min(ab, bb) - max(ay, by)) / min(ab - ay, bb - by)
    if min(a["area"], b["area"]) / max(a["area"], b["area"]) > options["group_area_ratio"]:
        return None
    if ((dx <= options["group_gap"] * length and oy >= options["group_overlap"]) or
            (dy <= options["group_gap"] * length and ox >= options["group_overlap"])):
        return np.hypot(dx, dy)
    return None


def instances_from_mask(evidence, config=None):
    """Group and refine binary foreground evidence independently of image loading."""
    options = resolve_detection_config(config)["instances"]
    if not isinstance(evidence, np.ndarray) or evidence.ndim != 2 or not evidence.size:
        raise ValueError("Expected a nonempty 2D foreground mask")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((evidence > 0).astype(np.uint8), 8)
    area, length = evidence.size, min(evidence.shape)
    groups = []
    for label in range(1, count):
        x, y, w, h, size = map(int, stats[label])
        if size >= options["min_fragment_area"] * area:
            groups.append(dict(labels=[label], area=size, bbox=(x, y, x+w, y+h)))
    # Merge the nearest admissible pair first, with deterministic label-order ties.
    while options["group_gap"] > 0:
        candidates = []
        for i, a in enumerate(groups):
            for j in range(i + 1, len(groups)):
                score = _join_score(a, groups[j], options, length)
                if score is not None:
                    candidates.append((score, i, j))
        if not candidates:
            break
        _, i, j = min(candidates)
        a, b = groups[i], groups.pop(j)
        ax, ay, ar, ab = a["bbox"]
        bx, by, br, bb = b["bbox"]
        a.update(labels=a["labels"] + b["labels"], area=a["area"] + b["area"],
                 bbox=(min(ax, bx), min(ay, by), max(ar, br), max(ab, bb)))
    candidates = []
    for group in groups:
        x, y, right, bottom = group["bbox"]
        w, h = right - x, bottom - y
        if (group["area"] >= options["min_area"] * area and min(w, h) >= options["min_side"] * length
                and options["min_aspect"] <= w / h <= options["max_aspect"]
                and group["area"] / (w * h) >= options["min_fill"]):
            candidates.append(group)
    candidates.sort(key=lambda g: (-g["area"], g["bbox"]))
    candidates = candidates[:options["max_paintings"]]
    paintings = []
    claimed = np.zeros(evidence.shape, np.uint8)
    selected = {label for group in candidates for label in group["labels"]}
    for group in candidates:
        seed = np.isin(labels, group["labels"]).astype(np.uint8) * 255
        mode = options["refinement"]
        closing = round(options["closing_ratio"] * length)
        closing = closing + (closing % 2 == 0) if closing > 1 else 0
        if mode in {"holes", "spans"}:
            refined = fill_painting_interior(seed, closing_size=closing,
                                             fill_rows=mode == "spans", fill_columns=mode == "spans",
                                             min_component_ratio=0)
        elif mode == "hull":
            points = cv2.findNonZero(seed)
            refined = seed.copy()
            cv2.fillConvexPoly(refined, cv2.convexHull(points), 255)
        else:
            refined = seed
        # Never steal another selected painting's evidence or overlap its mask.
        other = np.isin(labels, list(selected - set(group["labels"])))
        refined[other | (claimed > 0)] = 0
        if refined.any():
            paintings.append(Painting(_bbox(refined), refined, 0))
            claimed |= refined
    vertical = False
    if len(paintings) == 2:
        a, b = (p.bbox for p in paintings)
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) / min(a[2]-a[0], b[2]-b[0])
        vertical = overlap >= options["vertical_overlap"]
    paintings.sort(key=lambda p: (p.bbox[1], p.bbox[0]) if vertical else (p.bbox[0], p.bbox[1]))
    for index, painting in enumerate(paintings):
        painting.order = index
    return claimed, paintings


def detect_paintings(image, config=None):
    """Return (union mask, ordered Painting instances), at original resolution."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError("Expected a nonempty HxWx3 uint8 BGR image")
    config = resolve_detection_config(config)
    evidence = foreground_mask(image, **config["segmentation"])
    return instances_from_mask(evidence, config)
