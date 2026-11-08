"""QSD1-W4 annotation parsing and geometry-only alignment, isolated from retrieval."""
import pickle
from numbers import Integral
from pathlib import Path
import cv2
import numpy as np
from evaluate_background import load_instance_annotations


def parse_labels(raw):
    """Observed course format: photo -> painting IDs, with -1 for unknown."""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Expected nonempty photo rows of painting IDs")
    result = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Each photo must list its painting IDs")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < -1 for x in row):
            raise ValueError("Painting labels must be integers >= -1")
        result.append([int(x) for x in row])
    return result


def load_development(folder, paths, ids):
    # Deliberately limited to the named development set. Never infer GT for other datasets.
    folder = Path(folder)
    if folder.resolve().name.lower() != "qsd1_w4":
        raise ValueError("Development evaluation only supports qsd1_w4; blind-test GT is never loaded")
    with (folder / "gt_corresps.pkl").open("rb") as stream:
        labels = parse_labels(pickle.load(stream))
    if ids != list(range(len(labels))):
        raise ValueError("Development annotations require the complete numeric query inventory")
    frames = load_instance_annotations(folder / "frames.pkl", paths)
    for path, row in zip(paths, labels):
        if len(frames.get(path.name, [])) != len(row):
            raise ValueError(f"Frame/label counts disagree for {path.name}")
    return labels, frames


def align_boxes(boxes, polygons, threshold=.5):
    """Unique rectangle/polygon IoU >= threshold. Ambiguous eligible edges fail.

    Returns GT->detection indices. Missing and extra detections are reported;
    unknown F1 is conditional on matched paintings, never silently end-to-end F1.
    """
    if isinstance(threshold, bool) or not 0 < threshold <= 1:
        raise ValueError("Alignment IoU must be in (0, 1]")
    ious = np.zeros((len(polygons), len(boxes)))
    for i, polygon in enumerate(polygons):
        poly = np.asarray(polygon, np.float32)
        if (poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all()
                or not cv2.isContourConvex(poly) or cv2.contourArea(poly) <= 0):
            raise ValueError("Expected finite nondegenerate convex frame polygons")
        for j, box in enumerate(boxes):
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError("Invalid detection box")
            x, y, r, b = box
            if r <= x or b <= y:
                raise ValueError("Empty detection box")
            rect = np.array([[x,y],[r,y],[r,b],[x,b]], np.float32)
            intersection = cv2.intersectConvexConvex(poly, rect)[0]
            ious[i,j] = intersection/(cv2.contourArea(poly)+(r-x)*(b-y)-intersection)
    eligible = ious >= threshold
    if (eligible.sum(axis=0) > 1).any() or (eligible.sum(axis=1) > 1).any():
        raise ValueError("Ambiguous geometric alignment; multiple eligible frame/detection associations")
    matches = [int(np.flatnonzero(row)[0]) if row.any() else None for row in eligible]
    used = {i for i in matches if i is not None}
    return dict(matches=matches, missing=sum(i is None for i in matches), extra=len(boxes)-len(used),
                missing_gt_indices=[i for i,m in enumerate(matches) if m is None],
                extra_detection_indices=[i for i in range(len(boxes)) if i not in used],
                iou_threshold=threshold, ious=ious.tolist(), policy="unique geometry-only IoU; F1 on matched paintings only")
