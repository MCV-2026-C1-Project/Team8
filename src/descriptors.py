"""1D per-channel colour histogram descriptors (Week 1, Task 1).

All descriptors are concatenations of independent 1D histograms, one per
channel, each normalised to sum to 1. No joint (2D/3D) or spatial histograms
and no preprocessing (e.g. equalisation) are applied.
"""
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

YCBCR_DEFAULTS = dict(bins=(8, 32, 32), y_range=(0, 256), chroma_range=(0, 256))
HSV_DEFAULTS = dict(bins=(32, 16, 8), s_min=40, v_min=40)
RGB_DEFAULTS = dict(bins=(16, 16, 16))
LAB_DEFAULTS = dict(bins=(16, 16, 16))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def parse_id(path):
    """'bbdd_00007.jpg' -> 7, '00007.jpg' -> 7."""
    m = re.search(r"(\d+)$", Path(path).stem)
    if m is None:
        raise ValueError(f"Cannot parse an integer ID from {path}")
    return int(m.group(1))


def list_images(folder):
    """Return (paths, ids) of the .jpg files in `folder`, sorted by filename."""
    paths = sorted(Path(folder).glob("*.jpg"), key=lambda p: p.name)
    ids = [parse_id(p) for p in paths]
    return paths, ids


def load_image(path):
    """Load an image as BGR uint8 (safe with non-ASCII Windows paths)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load image: {path}"
    return img


# ----------------------------------------------------------------------------
# Histograms
# ----------------------------------------------------------------------------
def _norm(h):
    """Normalise to sum 1; an all-zero histogram stays all zeros (no NaNs)."""
    h = h.ravel().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def _hist(channel, bins, rng, mask=None):
    return cv2.calcHist([channel], [0], mask, [int(bins)], [float(rng[0]), float(rng[1])])


def ycbcr_hist(img_bgr, bins=YCBCR_DEFAULTS["bins"], y_range=YCBCR_DEFAULTS["y_range"],
               chroma_range=YCBCR_DEFAULTS["chroma_range"]):
    """Concatenated 1D histograms of Y, Cr, Cb (OpenCV order: Y, Cr, Cb)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return np.concatenate([
        _norm(_hist(y, bins[0], y_range)),
        _norm(_hist(cr, bins[1], chroma_range)),
        _norm(_hist(cb, bins[2], chroma_range)),
    ])


def hsv_hist(img_bgr, bins=HSV_DEFAULTS["bins"], s_min=HSV_DEFAULTS["s_min"],
             v_min=HSV_DEFAULTS["v_min"]):
    """Concatenated 1D histograms of H, S, V.

    H (range 0-180 in OpenCV) is computed only over pixels with S > s_min and
    V > v_min, since hue is unreliable for dark / grey pixels. S and V use all
    pixels.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s > s_min) & (v > v_min)).astype(np.uint8) * 255
    return np.concatenate([
        _norm(_hist(h, bins[0], (0, 180), mask)),
        _norm(_hist(s, bins[1], (0, 256))),
        _norm(_hist(v, bins[2], (0, 256))),
    ])


def _three_channel_hist(image, bins):
    """Concatenate normalised 1D histograms of three uint8 channels."""
    if len(bins) != 3 or any(int(b) != b or b <= 0 for b in bins):
        raise ValueError("bins must contain three positive integers")
    return np.concatenate([
        _norm(_hist(channel, size, (0, 256)))
        for channel, size in zip(cv2.split(image), bins)
    ])


def rgb_hist(img_bgr, bins=RGB_DEFAULTS["bins"]):
    """Independent 1D histograms in true R, G, B order, each summing to 1."""
    return _three_channel_hist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bins)


def lab_hist(img_bgr, bins=LAB_DEFAULTS["bins"]):
    """L, a, b histograms using OpenCV uint8 Lab encoding (all in [0, 256))."""
    return _three_channel_hist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab), bins)


METHODS = {"ycbcr": ycbcr_hist, "hsv": hsv_hist, "rgb": rgb_hist, "lab": lab_hist}
DEFAULTS = {"ycbcr": YCBCR_DEFAULTS, "hsv": HSV_DEFAULTS,
            "rgb": RGB_DEFAULTS, "lab": LAB_DEFAULTS}


def full_params(method, **params):
    """Default parameters of `method` updated with the given ones."""
    p = dict(DEFAULTS[method])
    p.update({k: v for k, v in params.items() if v is not None})
    return p


def method_tag(method, **params):
    """Filename tag encoding method and parameters, e.g. 'ycbcr_8-32-32'.

    Non-default ranges/thresholds are appended, e.g. 'ycbcr_8-32-32_c64-192'
    or 'hsv_32-16-8_s30v50'.
    """
    p = full_params(method, **params)
    tag = f"{method}_" + "-".join(str(int(b)) for b in p["bins"])
    if method == "ycbcr":
        if tuple(p["y_range"]) != YCBCR_DEFAULTS["y_range"]:
            tag += "_y{}-{}".format(*p["y_range"])
        if tuple(p["chroma_range"]) != YCBCR_DEFAULTS["chroma_range"]:
            tag += "_c{}-{}".format(*p["chroma_range"])
    elif method == "hsv":
        if (p["s_min"], p["v_min"]) != (HSV_DEFAULTS["s_min"], HSV_DEFAULTS["v_min"]):
            tag += f"_s{p['s_min']}v{p['v_min']}"
    return tag


def compute_descriptor(img_bgr, method, **params):
    return METHODS[method](img_bgr, **full_params(method, **params))


def compute_folder(folder, method, **params):
    """Compute descriptors for all .jpg images in `folder`.

    Returns (ids: int64 [N], desc: float32 [N x D]); row i corresponds to ids[i].
    """
    paths, ids = list_images(folder)
    assert paths, f"No .jpg images found in {folder}"
    desc = [compute_descriptor(load_image(p), method, **params)
            for p in tqdm(paths, desc=f"{Path(folder).name}/{method}")]
    return np.asarray(ids, dtype=np.int64), np.stack(desc).astype(np.float32)


def channel_sizes(method, **params):
    """Number of bins of each channel's slice in the descriptor."""
    return [int(b) for b in full_params(method, **params)["bins"]]
