"""Homographies map query -> database; estimation/error units are normalized pixels."""
from dataclasses import dataclass

import cv2
import numpy as np

from local_matching_config import resolve_geometry
from local_matching import Correspondences


@dataclass
class Verification:
    matches: Correspondences
    config: dict
    success: bool
    homography: np.ndarray | None          # Query normalized -> database normalized.
    native_homography: np.ndarray | None   # Query native -> database native.
    query_transform: np.ndarray
    database_transform: np.ndarray
    inlier_mask: np.ndarray     # Aligned to ALL filtered correspondences.
    reprojection_errors: list   # Forward error for each filtered row; None if undefined.
    failure_reason: str | None

    @property
    def inlier_count(self):
        return int(self.inlier_mask.sum()) if self.success else 0

    @property
    def inlier_ratio(self):
        return self.inlier_count / self.matches.count if self.matches.count else 0.

    @property
    def mean_error(self):
        values = [e for e, keep in zip(self.reprojection_errors, self.inlier_mask) if keep]
        return float(np.mean(values)) if values else None

    @property
    def median_error(self):
        values = [e for e, keep in zip(self.reprojection_errors, self.inlier_mask) if keep]
        return float(np.median(values)) if values else None


def normalization(shape, reference):
    if len(shape) != 2 or any(not np.isfinite(v) or v <= 0 for v in shape):
        raise ValueError("Geometry needs positive (height, width) image shapes")
    return np.diag([reference / max(shape), reference / max(shape), 1.])


def _spans_plane(points):
    if len(points) < 4:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # Numerical degeneracy check, not a learned acceptance/rejection threshold.
    return bool(singular[0] > 1e-9 and singular[1] / singular[0] > 1e-3)


def verify_geometry(matches, config=None):
    config = resolve_geometry(config)
    tq = normalization(matches.query_shape, config["reference_long_side"])
    td = normalization(matches.database_shape, config["reference_long_side"])
    n = matches.count
    q, d = np.asarray(matches.query_points, float), np.asarray(matches.database_points, float)
    result = Verification(matches, config, False, None, None, tq, td, np.zeros(n, bool), [None]*n, None)

    def fail(reason):
        result.failure_reason = reason
        return result

    if q.shape != (n, 2) or d.shape != (n, 2):
        raise ValueError("Correspondence point arrays must align with filtered indices")
    if n < 4:
        return fail("insufficient_correspondences")
    if not np.isfinite(q).all() or not np.isfinite(d).all():
        return fail("nonfinite_coordinates")
    # Multiple orientations/duplicate coordinates must not inflate geometric evidence.
    unique, seen_q, seen_d = [], set(), set()
    for i, (a, b) in enumerate(zip(q, d)):
        a, b = tuple(a), tuple(b)
        if a not in seen_q and b not in seen_d:
            unique.append(i)
            seen_q.add(a)
            seen_d.add(b)
    selected = np.asarray(unique, dtype=int)
    q, d = q * tq[0, 0], d * td[0, 0]
    if not _spans_plane(q[selected]) or not _spans_plane(d[selected]):
        return fail("degenerate_correspondences")
    flag = cv2.RANSAC if config["estimator"] == "ransac" else cv2.USAC_MAGSAC
    try:
        h, mask = cv2.findHomography(q[selected], d[selected], method=flag,
                                    ransacReprojThreshold=config["reprojection_threshold"],
                                    maxIters=config["max_iterations"], confidence=config["confidence"])
    except cv2.error as error:
        return fail(f"opencv_estimation_failed: {error.code}")
    if h is None or h.shape != (3, 3) or not np.isfinite(h).all() or not np.any(h):
        return fail("invalid_homography")
    h = h / np.max(np.abs(h))
    if np.linalg.cond(h) > 1e12:
        return fail("singular_or_ill_conditioned_homography")
    if mask is None or mask.shape not in ((len(selected),), (len(selected), 1)) or not np.isin(mask, [0, 1]).all():
        return fail("invalid_inlier_mask")
    projected = np.column_stack((q, np.ones(n))) @ h.T
    defined = np.abs(projected[:, 2]) > 1e-12
    errors = np.full(n, np.inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors[defined] = np.linalg.norm(projected[defined, :2] / projected[defined, 2, None] - d[defined], axis=1)
    result.reprojection_errors = [float(e) if np.isfinite(e) else None for e in errors]
    keep = np.zeros(n, bool)
    keep[selected] = mask.reshape(-1).astype(bool)
    keep &= np.isfinite(errors) & (errors <= config["reprojection_threshold"])
    if not _spans_plane(q[keep]) or not _spans_plane(d[keep]):
        return fail("no_valid_geometric_consensus")
    native = np.linalg.inv(td) @ h @ tq
    if not np.isfinite(native).all():
        return fail("invalid_native_homography")
    result.success, result.homography, result.native_homography = True, h, native
    result.inlier_mask = keep
    return result
