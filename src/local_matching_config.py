"""Validated single-run settings, independent of the inspection CLI."""
import json
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from local_features import resolve_local_config

ROOT = Path(__file__).resolve().parents[1]


def section(value, defaults, label):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError(f"Invalid {label} settings; supported keys: {list(defaults)}")
    return dict(defaults, **value)


def number(value, label, minimum=0, maximum=None, integer=False, inclusive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral if integer else Real)
            or not np.isfinite(value) or (value < minimum if inclusive else value <= minimum)
            or (maximum is not None and value >= maximum)):
        raise ValueError(f"Invalid {label}: {value!r}")
    return int(value) if integer else float(value)


def resolve_matching(config=None):
    result = section(config, dict(backend="bf", filter="ratio", ratio_threshold=.75), "matching")
    if result["backend"] != "bf" or result["filter"] not in ("ratio", "ratio_mutual"):
        raise ValueError("Matching requires bf and ratio or ratio_mutual")
    result["ratio_threshold"] = number(result["ratio_threshold"], "ratio_threshold", maximum=1)
    return result


def resolve_geometry(config=None):
    result = section(config, dict(model="homography", estimator="ransac", reprojection_threshold=3.,
                                  reference_long_side=1000., max_iterations=2000, confidence=.995), "geometry")
    if result["model"] != "homography" or result["estimator"] not in ("ransac", "usac_magsac"):
        raise ValueError("Geometry requires homography with ransac or usac_magsac")
    if result["estimator"] == "usac_magsac" and not hasattr(cv2, "USAC_MAGSAC"):
        raise ValueError(f"OpenCV {cv2.__version__} does not support USAC_MAGSAC; no estimator fallback is used")
    for key in ("reprojection_threshold", "reference_long_side"):
        result[key] = number(result[key], key)
    result["max_iterations"] = number(result["max_iterations"], "max_iterations", integer=True, maximum=2**31)
    result["confidence"] = number(result["confidence"], "confidence", maximum=1)
    return result


def resolve_scoring(config=None):
    result = section(config, dict(mode="inlier_count", coverage_grid=[4, 4]), "scoring")
    if result["mode"] not in ("inlier_count", "coverage_weighted"):
        raise ValueError("Scoring mode must be inlier_count or coverage_weighted")
    grid = result["coverage_grid"]
    if not isinstance(grid, (list, tuple)) or len(grid) != 2:
        raise ValueError("coverage_grid must be [rows, columns]")
    result["coverage_grid"] = [number(v, "coverage grid", integer=True, maximum=2**31) for v in grid]
    return result


def resolve_run_config(config=None, feature_defaults=None):
    result = section(config, dict(feature_method="sift", feature_params={}, matching=None, geometry=None,
                                  scoring=None, retrieval=None, crop=None, rejection=None), "run")
    if feature_defaults is None:
        feature_defaults = json.loads((ROOT / "configs/week4/local_features.json").read_text(encoding="utf-8"))
    method = result["feature_method"]
    resolve_local_config(method)  # Validate before dictionary access.
    if not isinstance(result["feature_params"], dict):
        raise ValueError("feature_params must be a dictionary")
    params = dict(feature_defaults.get(method, {}), **result["feature_params"])
    result["feature_params"] = resolve_local_config(method, params)["params"]
    result["rejection"] = resolve_rejection(result["rejection"])
    result["matching"] = resolve_matching(result["matching"])
    result["geometry"] = resolve_geometry(result["geometry"])
    result["scoring"] = resolve_scoring(result["scoring"])
    result["retrieval"] = section(result["retrieval"], dict(k=5), "retrieval")
    result["retrieval"]["k"] = number(result["retrieval"]["k"], "k", integer=True)
    result["crop"] = section(result["crop"], dict(inset=.03), "crop")
    result["crop"]["inset"] = number(result["crop"]["inset"], "crop inset", maximum=.5, inclusive=True)
    return result


def resolve_rejection(config=None):
    result = section(config, dict(mode="disabled", min_inliers=None), "rejection")
    if result["mode"] not in ("disabled", "min_inliers"):
        raise ValueError("rejection mode must be disabled or min_inliers")
    if result["min_inliers"] is not None or result["mode"] == "min_inliers":
        result["min_inliers"] = number(result["min_inliers"], "min_inliers", integer=True)
    return result
