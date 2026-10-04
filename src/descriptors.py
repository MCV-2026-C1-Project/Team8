"""1D per-channel colour histogram descriptors.

All descriptors are concatenations of independent 1D histograms, one per
channel, each normalised to sum to 1 by default. No joint (2D/3D)
or spatial histograms and no image preprocessing (e.g. equalisation) are applied.
"""
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

YCBCR_DEFAULTS = dict(bins=(8, 32, 32), y_range=(0, 256), chroma_range=(0, 256),
                      chroma_overflow=False)
HSV_DEFAULTS = dict(bins=(16, 16, 16), s_min=10, v_min=40,
                    hue_valid_weight=False, hue_smoothing=False)
HSV_BASELINE_DEFAULTS = dict(bins=(180, 256, 256))
RGB_DEFAULTS = dict(bins=(8, 8, 8))
LAB_DEFAULTS = dict(bins=(8, 64, 64))


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


def _chroma_hist(channel, bins, rng, overflow):
    """Histogram over [lo, hi); with overflow, pixels below/above get their own end bins.

    calcHist drops out-of-range pixels, so without overflow the histogram only
    describes the kept pixels. The overflow bins keep how many were dropped
    (and on which side), so the in-range bins sum to the kept fraction.
    """
    h = _hist(channel, bins, rng).ravel()
    if not overflow:
        return _norm(h)
    below = np.count_nonzero(channel < rng[0])
    above = np.count_nonzero(channel >= rng[1])
    return _norm(np.concatenate([[below], h, [above]]))


def ycbcr_hist(img_bgr, bins=YCBCR_DEFAULTS["bins"], y_range=YCBCR_DEFAULTS["y_range"],
               chroma_range=YCBCR_DEFAULTS["chroma_range"],
               chroma_overflow=YCBCR_DEFAULTS["chroma_overflow"]):
    """Concatenated 1D histograms of Y, Cr, Cb (OpenCV order: Y, Cr, Cb).

    With chroma_overflow, Cr and Cb each get bins + 2 values (below, range, above).
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return np.concatenate([
        _norm(_hist(y, bins[0], y_range)),
        _chroma_hist(cr, bins[1], chroma_range, chroma_overflow),
        _chroma_hist(cb, bins[2], chroma_range, chroma_overflow),
    ])


def hsv_hist(img_bgr, bins=HSV_DEFAULTS["bins"], s_min=HSV_DEFAULTS["s_min"],
             v_min=HSV_DEFAULTS["v_min"], hue_valid_weight=HSV_DEFAULTS["hue_valid_weight"],
             hue_smoothing=HSV_DEFAULTS["hue_smoothing"]):
    """Concatenated 1D histograms of H, S, V.

    H (range 0-180 in OpenCV) is computed only over pixels with S > s_min and
    V > v_min, since hue is unreliable for dark / grey pixels. S and V use all
    pixels. Optional circular smoothing uses [1/4, 1/2, 1/4] on normalised H;
    optional valid-Hue weighting then scales H by the fraction of valid pixels.
    Both options leave S/V unchanged and preserve the baseline when disabled.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s > s_min) & (v > v_min)).astype(np.uint8) * 255
    hist_h = _norm(_hist(h, bins[0], (0, 180), mask))
    if hue_smoothing:
        hist_h = 0.25 * np.roll(hist_h, 1) + 0.5 * hist_h + 0.25 * np.roll(hist_h, -1)
    if hue_valid_weight:
        hist_h *= np.count_nonzero(mask) / mask.size
    return np.concatenate([
        hist_h,
        _norm(_hist(s, bins[1], (0, 256))),
        _norm(_hist(v, bins[2], (0, 256))),
    ])


def hsv_baseline_hist(img_bgr, bins=HSV_BASELINE_DEFAULTS["bins"]):
    """Baseline HSV: one bin per value, every pixel counted in every channel.

    OpenCV 8-bit HSV has H in 0-179 and S, V in 0-255, so 180-256-256 bins
    means no quantisation. No S/V mask on Hue.
    """
    h, s, v = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    return np.concatenate([
        _norm(_hist(h, bins[0], (0, 180))),
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


METHODS = {"ycbcr": ycbcr_hist, "hsv": hsv_hist, "hsv_baseline": hsv_baseline_hist,
           "rgb": rgb_hist, "lab": lab_hist}
DEFAULTS = {"ycbcr": YCBCR_DEFAULTS, "hsv": HSV_DEFAULTS, "hsv_baseline": HSV_BASELINE_DEFAULTS,
            "rgb": RGB_DEFAULTS, "lab": LAB_DEFAULTS}


def full_params(method, **params):
    """Default parameters of `method` updated with the given ones."""
    p = dict(DEFAULTS[method])
    p.update({k: v for k, v in params.items() if v is not None})
    return p


def method_tag(method, **params):
    """Canonical tag encoding every effective parameter, independent of defaults."""
    p = full_params(method, **params)
    def number(value):
        return str(int(value)) if value == int(value) else str(float(value))

    tag = f"{method}_" + "-".join(str(int(b)) for b in p["bins"])
    if method == "ycbcr":
        tag += "_y" + "-".join(number(v) for v in p["y_range"])
        tag += "_c" + "-".join(number(v) for v in p["chroma_range"])
        if p["chroma_overflow"]:
            tag += "_o"
    elif method == "hsv":
        tag += f"_s{number(p['s_min'])}v{number(p['v_min'])}"
        tag += f"_hw{int(p['hue_valid_weight'])}"
        tag += f"_hs{int(p['hue_smoothing'])}"
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
    p = full_params(method, **params)
    sizes = [int(b) for b in p["bins"]]
    if method == "ycbcr" and p["chroma_overflow"]:
        sizes[1] += 2
        sizes[2] += 2
    return sizes
