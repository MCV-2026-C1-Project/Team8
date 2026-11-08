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
    def substantial(group):
        x, y, right, bottom = group["bbox"]
        w, h = right - x, bottom - y
        return (
            min(w, h) >= 0.15 * length
            and group["area"] >= 0.02 * length**2
            and group["area"] / (w * h) >= 0.2
        )

    # Keep substantial stacked regions separate even if their boxes overlap slightly.
    if oy <= 0.05 and ox >= options["group_overlap"]:
        if substantial(a) and substantial(b):
            return None

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
    count, labels, stats, _ = cv2.connectedComponentsWithStats((evidence > 0).astype(np.uint8), connectivity=8)
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



def _ordered_result(paintings, shape, options):
    """Build a disjoint union and preserve the existing reading-order convention."""
    union = np.zeros(shape, np.uint8)
    result = []
    for painting in paintings:
        mask = painting.mask.copy()
        mask[union > 0] = 0
        if mask.any():
            result.append(Painting(_bbox(mask), mask, 0))
            union |= mask
    vertical = False
    if len(result) == 2:
        a, b = (p.bbox for p in result)
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        vertical = overlap / min(a[2] - a[0], b[2] - b[0]) >= options['vertical_overlap']
    result.sort(key=lambda p: (p.bbox[1], p.bbox[0]) if vertical
                else (p.bbox[0], p.bbox[1]))
    for index, painting in enumerate(result):
        painting.order = index
    return union, result


def _split_retry(evidence, config):
    """Try opening only when the original evidence produces one instance."""
    union, paintings = instances_from_mask(evidence, config)
    if len(paintings) != 1 or config['instances']['max_paintings'] != 2:
        return union, paintings
    size = max(3, round(min(evidence.shape) * 0.015))
    size += size % 2 == 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(evidence, cv2.MORPH_OPEN, kernel)
    split_union, split = instances_from_mask(opened, config)
    if len(split) != 2:
        return union, paintings
    a, b = (p.bbox for p in split)
    overlap_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    separated = (overlap_x <= 0.05 * min(a[2] - a[0], b[2] - b[0])
                 or overlap_y <= 0.05 * min(a[3] - a[1], b[3] - b[1]))
    areas = [np.count_nonzero(p.mask) for p in split]
    retained = np.count_nonzero((split_union > 0) & (union > 0)) / np.count_nonzero(union)
    if separated and min(areas) >= 0.15 * sum(areas) and retained >= 0.65:
        return split_union, split
    return union, paintings


def _edge_candidates(image, options):
    """Multiscale frame proposals, verified against unclosed image edges.

    Work at a bounded resolution. Each side must have edge support, and nearly
    right-angled convex quadrilaterals suppress arbitrary artwork contours.
    All returned masks and bounding boxes use the original resolution.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    work = cv2.resize(image, (max(1, round(width * scale)),
                              max(1, round(height * scale))))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    short_side = min(gray.shape)
    if short_side < 16:
        return []
    proposals = []
    for sigma in (0.5, 1.0, 2.0, 4.0):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(gx, gy)
        peak = float(magnitude.max())
        if peak == 0:
            continue
        gradient = np.uint8(np.clip(magnitude * 255 / peak, 0, 255))
        threshold, _ = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = max(15.0, float(threshold) * peak / 255)
        edges = cv2.Canny(smooth, high * 0.4, high, L2gradient=True)
        nearby = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
        close_size = max(3, round(short_side * 0.008))
        close_size += close_size % 2 == 0
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 np.ones((close_size, close_size), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            points = quad[:, 0, :].astype(float)
            fraction = abs(cv2.contourArea(quad)) / gray.size
            if not max(0.01, options['min_area']) <= fraction <= 0.95:
                continue
            lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            if lengths.min() < max(0.04, options['min_side']) * short_side:
                continue
            aspect = max(lengths[0] + lengths[2], 1) / max(lengths[1] + lengths[3], 1)
            if not 0.1 <= aspect <= 10:
                continue
            first = np.roll(points, 1, axis=0) - points
            second = np.roll(points, -1, axis=0) - points
            cosines = np.abs(np.sum(first * second, axis=1) /
                             (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)))
            if cosines.max() > 0.55:
                continue
            support = []
            for start, end in zip(points, np.roll(points, -1, axis=0)):
                samples = np.linspace(start, end, max(2, round(np.linalg.norm(end - start))))
                samples = np.rint(samples).astype(int)
                support.append(float(nearby[samples[:, 1], samples[:, 0]].mean()))
            if min(support) < 0.65 or np.mean(support) < 0.8:
                continue
            native = points * [width / gray.shape[1], height / gray.shape[0]]
            native = np.rint(native).astype(np.int32)
            # Reject image-border rectangles created by cropping or acquisition.
            if (native[:, 0].min() < 2 or native[:, 1].min() < 2
                    or native[:, 0].max() >= width - 2 or native[:, 1].max() >= height - 2):
                continue
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, native, 255)
            score = float(np.mean(support))
            proposals.append((score, Painting(_bbox(mask), mask, 0)))
    # Prefer outer frames over nested inner frames and internal artwork panels.
    proposals.sort(key=lambda item: (-np.count_nonzero(item[1].mask), -item[0], item[1].bbox))
    unique = []
    for score, candidate in proposals:
        size = np.count_nonzero(candidate.mask)
        duplicate = False
        for _, existing in unique:
            intersection = np.count_nonzero((candidate.mask > 0) & (existing.mask > 0))
            if intersection / min(size, np.count_nonzero(existing.mask)) > 0.8:
                duplicate = True
                break
        if not duplicate:
            unique.append((score, candidate))
    return unique


def _recover_frames(image, evidence, paintings, options):
    """Associate edge frames with colour instances and recover missing frames.

    Strong frame proposals may enclose sparse evidence. Unlike completion,
    recovery therefore does not require a large colour-mask IoU.
    """
    proposals = _edge_candidates(image, options)
    chosen = []
    for score, frame in proposals:
        inside = frame.mask > 0
        size = int(inside.sum())
        colour_fraction = np.count_nonzero(inside & (evidence > 0)) / size
        associations = []
        for index, painting in enumerate(paintings):
            old = painting.mask > 0
            intersection = np.count_nonzero(old & inside)
            coverage = intersection / np.count_nonzero(old)
            frame_coverage = intersection / size
            # Completion of one region or subdivision of a merged colour crop.
            if coverage >= 0.75 or frame_coverage >= 0.6:
                associations.append(index)
        if colour_fraction < 0.02 or (not associations and (score < 0.9 or colour_fraction < 0.05)):
            continue
        if any(np.count_nonzero(inside & (p.mask > 0)) /
               min(size, np.count_nonzero(p.mask)) > 0.1 for p in chosen):
            continue
        chosen.append(frame)
        if len(chosen) == options['max_paintings']:
            break
    if not chosen:
        return paintings
    # A frame replacement must preserve the colour evidence it supersedes.
    # Several frames can jointly explain one merged detection.
    for painting in paintings:
        old = painting.mask > 0
        overlap = [np.count_nonzero(old & (frame.mask > 0)) for frame in chosen]
        coverage = sum(overlap) / np.count_nonzero(old)
        if coverage >= 0.9:
            continue
        # Discard internal panels that would replace only part of this artwork.
        chosen = [frame for frame, intersection in zip(chosen, overlap)
                  if intersection == 0]
        chosen.append(painting)

    return chosen[:options['max_paintings']]


def _merge_overlapping_fragments(paintings):
    """Merge two broad fragments with substantial bounding-box overlap.

    Thresholds are development heuristics. Genuinely overlapping artworks can
    also satisfy them, so validate painting counts on the complete query set.
    """
    if len(paintings) != 2:
        return paintings

    first, second = paintings
    ax, ay, ar, ab = first.bbox
    bx, by, br, bb = second.bbox
    width_a, height_a = ar - ax, ab - ay
    width_b, height_b = br - bx, bb - by
    overlap_x = max(0, min(ar, br) - max(ax, bx))
    overlap_y = max(0, min(ab, bb) - max(ay, by))
    horizontal_overlap = overlap_x / min(width_a, width_b)
    vertical_overlap = overlap_y / min(height_a, height_b)
    width_similarity = min(width_a, width_b) / max(width_a, width_b)

    if (horizontal_overlap >= 0.9 and vertical_overlap >= 0.3
            and width_similarity >= 0.85):
        merged = cv2.bitwise_or(first.mask, second.mask)
        return [Painting(_bbox(merged), merged, 0)]
    return paintings


def detect_paintings(image, config=None):
    """Return (union mask, ordered Painting instances), at original resolution.

    Uses the existing wall model, a conditional bridge-opening retry, and
    multiscale edge recovery. Existing segmentation/instances JSON files work.
    Thresholds for new proposal checks are initial development defaults.
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError('Expected a nonempty HxWx3 uint8 BGR image')
    config = resolve_detection_config(config)
    evidence = foreground_mask(image, **config['segmentation'])
    _, paintings = _split_retry(evidence, config)
    paintings = _merge_overlapping_fragments(paintings)
    paintings = _recover_frames(image, evidence, paintings, config['instances'])
    return _ordered_result(paintings, evidence.shape, config['instances'])
