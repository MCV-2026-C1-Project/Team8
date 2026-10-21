"""Shared effective configuration, dimensions and identities for every interface."""
import hashlib
import json
import math
from numbers import Integral, Real

from spatial import resolve_spatial

JOINT_METHODS = {"hsv_hs2d", "lab_ab2d", "hsv_3d", "lab_3d"}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def resolve_params(method, supplied):
    # Import lazily so legacy descriptors remain the public extraction API.
    from descriptors import DEFAULTS
    if method not in DEFAULTS:
        raise ValueError(f"Unknown descriptor {method!r}; choose from {list(DEFAULTS)}")
    supplied = {k: v for k, v in supplied.items() if v is not None}
    unknown = set(supplied) - set(DEFAULTS[method]) - {"spatial", "normalization"}
    if unknown:
        raise ValueError(f"Unknown parameters for {method}: {sorted(unknown)}")
    params = dict(DEFAULTS[method])
    params.update(supplied)
    spatial = resolve_spatial(params.pop("spatial", None))
    normalization = params.pop("normalization", "none")
    if normalization not in ("none", "photometric"):
        raise ValueError("normalization must be 'none' or 'photometric'")
    if normalization == "photometric":
        if method not in {"lab", "ycbcr", "lab_ab2d", "lab_3d"}:
            raise ValueError("Photometric normalization supports only lab, ycbcr, lab_ab2d and lab_3d descriptors")
        params["normalization"] = normalization
    from descriptors import TEXTURE_METHODS
    if method in TEXTURE_METHODS:
        params["resize"] = positive_int(params["resize"], "resize")
        size = params["resize"]
        if method in {"lbp", "multiscale_lbp"}:
            radii = [params["radius"]] if method == "lbp" else params["radii"]
            points = [params["points"]] if method == "lbp" else params["points"]
            if (not isinstance(radii, (list, tuple)) or not isinstance(points, (list, tuple))
                    or not radii or len(radii) != len(points)):
                raise ValueError("radii and points require matching nonempty lists")
            for radius, count in zip(radii, points):
                if (isinstance(radius, bool) or not isinstance(radius, Real) or
                        not math.isfinite(radius) or radius <= 0 or 2*math.ceil(radius) >= size):
                    raise ValueError("LBP radius must be positive and leave valid pixels")
                positive_int(count, "points")
            if method == "multiscale_lbp":
                params["radii"], params["points"] = list(radii), list(points)
        elif method == "dct":
            params["n_coeffs"] = positive_int(params["n_coeffs"], "n_coeffs")
            if size % 2 or params["n_coeffs"] >= size**2:
                raise ValueError("DCT resize must be even and n_coeffs <= resize**2 - 1")
        else:
            params["levels"] = positive_int(params["levels"], "levels")
            if params["levels"] > int(math.log2(size)) or size % (2**params["levels"]):
                raise ValueError("Wavelet resize must be divisible by 2**levels")
        if spatial["mode"] != "global":
            params["spatial"] = spatial
        return params
    # Omit the disabled setting so all existing tags and cache metadata survive.
    count = 2 if method in {"hsv_hs2d", "lab_ab2d"} else 3
    bins = params["bins"]
    if not isinstance(bins, (tuple, list)) or len(bins) != count:
        raise ValueError(f"{method} bins must contain {count} positive integers")
    params["bins"] = [positive_int(b, "bins") for b in bins]
    for key in ("v_bins", "l_bins"):
        if key in params:
            params[key] = positive_int(params[key], key)
    for key in ("hue_smoothing", "hue_valid_weight", "filter_hue", "achromatic_bin",
                "include_v", "include_l", "chroma_overflow"):
        if key in params and not isinstance(params[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("s_min", "v_min", "chroma_weight"):
        if key in params:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
            lo, hi = (0, 1) if key == "chroma_weight" else (-1, 255)
            if not lo <= value <= hi:
                raise ValueError(f"{key} must lie in [{lo}, {hi}]")
            params[key] = float(value)
    for key in ("y_range", "chroma_range"):
        if key in params:
            value = params[key]
            if (not isinstance(value, (tuple, list)) or len(value) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in value)
                    or value[0] >= value[1]):
                raise ValueError(f"{key} must contain two finite increasing numbers")
            params[key] = [float(x) for x in value]
    # Global is the legacy representation: keep its numerical behaviour/tags.
    if spatial["mode"] != "global":
        params["spatial"] = spatial
    return params


def resolve_descriptor_config(config):
    """Accept a Python dict (or an exported document); never require JSON I/O.

    Shape: {descriptor: name, params: {...}, spatial: {...} (optional)}.
    Canonical form keeps explicit spatial settings inside params, including global.
    """
    if not isinstance(config, dict):
        raise ValueError("Descriptor configuration must be a dictionary")
    if "config" in config:
        config = config["config"]
    if not isinstance(config, dict) or set(config) - {"descriptor", "params", "spatial"}:
        raise ValueError("Descriptor configuration accepts descriptor, params and spatial")
    if "descriptor" not in config or not isinstance(config.get("params", {}), dict):
        raise ValueError("A descriptor name and a params dictionary are required")
    params = dict(config.get("params", {}))
    if "spatial" in config:
        if "spatial" in params:
            raise ValueError("Specify spatial only once")
        params["spatial"] = config["spatial"]
    params = resolve_params(config["descriptor"], params)
    params.setdefault("spatial", resolve_spatial())
    return dict(descriptor=config["descriptor"], params=params)


def descriptor_dim(method, **params):
    p = resolve_params(method, params)
    if method == "lbp":
        size = p["points"] + 2
    elif method == "multiscale_lbp":
        size = sum(count + 2 for count in p["points"])
    elif method == "dct":
        size = p["n_coeffs"]
    elif method == "wavelet":
        size = 6 * p["levels"]
    elif method in {"hsv_3d", "lab_3d"}:
        size = math.prod(p["bins"])
    elif method == "hsv_hs2d":
        size = math.prod(p["bins"]) + int(p["achromatic_bin"]) + (p["v_bins"] if p["include_v"] else 0)
    elif method == "lab_ab2d":
        size = math.prod(p["bins"]) + (p["l_bins"] if p["include_l"] else 0)
    else:
        size = sum(p["bins"]) + (4 if method == "ycbcr" and p["chroma_overflow"] else 0)
    return size * sum(n * n for n in p.get("spatial", {"grids": [1]})["grids"])


def hashed_tag(method, params):
    digest = hashlib.sha256(canonical_json(dict(version=1, descriptor=method, params=params)).encode()).hexdigest()[:24]
    return f"{method}_{'-'.join(map(str, params.get('bins', [])))}_{digest}"


def measure_channels(method, params, measure):
    """Validate compatibility before any extraction; preserve legacy W1 formulas."""
    from descriptors import channel_sizes
    from distances import MEASURES
    params = resolve_params(method, params)
    if measure not in MEASURES:
        raise ValueError(f"Unknown measure {measure!r}")
    from descriptors import TEXTURE_METHODS
    if method in {"dct", "wavelet"} and measure not in {"l1", "euclidean"}:
        raise ValueError(f"{method} supports l1 or euclidean distances")
    if method in TEXTURE_METHODS and measure == "wasserstein":
        raise ValueError("Wasserstein is unsupported for texture descriptors")
    if measure == "wasserstein":
        if method in JOINT_METHODS or "spatial" in params:
            raise ValueError("Wasserstein supports only global independent 1D histograms, not joint or spatial descriptors")
        return channel_sizes(method, **params)
    return None
