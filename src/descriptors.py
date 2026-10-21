"""Legacy 1D and joint colour descriptors with optional spatial concatenation."""
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from descriptor_config import JOINT_METHODS, descriptor_dim, hashed_tag, resolve_params
from joint_histograms import (hsv_hs2d, lab_ab2d, hsv_3d, lab_3d,
                              lab_ab2d_from_color, lab_3d_from_color)
from photometric import color_image
from spatial import block_slices, crop_to_foreground, resolve_spatial, spatial_descriptor, trim_background_blocks

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
    """Discover JPEG files (case-insensitive), sorted by filename; reject duplicate IDs.

    Filename ordering preserves the existing lowercase .jpg query/GT order.
    PNG files, including segmentation masks, are never included.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist or is not a directory: {folder.resolve()}")
    paths = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
                   key=lambda p: p.name)
    ids = [parse_id(p) for p in paths]
    seen = {}
    for path, image_id in zip(paths, ids):
        if image_id in seen:
            raise ValueError(f"Duplicate image ID {image_id} in {folder}: {seen[image_id]} and {path.name}")
        seen[image_id] = path.name
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


def _chroma_hist(channel, bins, rng, overflow, mask=None):
    """Histogram over [lo, hi); with overflow, pixels below/above get their own end bins.

    calcHist drops out-of-range pixels, so without overflow the histogram only
    describes the kept pixels. The overflow bins keep how many were dropped
    (and on which side), so the in-range bins sum to the kept fraction.
    """
    h = _hist(channel, bins, rng, mask).ravel()
    if not overflow:
        return _norm(h)
    foreground = channel if mask is None else channel[mask > 0]
    below = np.count_nonzero(foreground < rng[0])
    above = np.count_nonzero(foreground >= rng[1])
    return _norm(np.concatenate([[below], h, [above]]))


def ycbcr_hist(img_bgr, bins=YCBCR_DEFAULTS["bins"], y_range=YCBCR_DEFAULTS["y_range"],
               chroma_range=YCBCR_DEFAULTS["chroma_range"],
               chroma_overflow=YCBCR_DEFAULTS["chroma_overflow"], normalization="none", mask=None):
    """Concatenated 1D histograms of Y, Cr, Cb (OpenCV order: Y, Cr, Cb).

    With chroma_overflow, Cr and Cb each get bins + 2 values (below, range, above).
    """
    return ycbcr_hist_from_color(color_image(img_bgr, "ycbcr", normalization, mask), bins,
                                 y_range, chroma_range, chroma_overflow, mask)


def ycbcr_hist_from_color(ycrcb, bins=YCBCR_DEFAULTS["bins"],
                          y_range=YCBCR_DEFAULTS["y_range"],
                          chroma_range=YCBCR_DEFAULTS["chroma_range"],
                          chroma_overflow=YCBCR_DEFAULTS["chroma_overflow"], mask=None):
    """Independent histograms from an already prepared Y, Cr, Cb image."""
    y, cr, cb = cv2.split(ycrcb)
    return np.concatenate([
        _norm(_hist(y, bins[0], y_range, mask)),
        _chroma_hist(cr, bins[1], chroma_range, chroma_overflow, mask),
        _chroma_hist(cb, bins[2], chroma_range, chroma_overflow, mask),
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


def _three_channel_hist(image, bins, mask=None):
    """Concatenate normalised 1D histograms in [0, 256), optionally masked."""
    if len(bins) != 3 or any(int(b) != b or b <= 0 for b in bins):
        raise ValueError("bins must contain three positive integers")
    return np.concatenate([
        _norm(_hist(channel, size, (0, 256), mask))
        for channel, size in zip(cv2.split(image), bins)
    ])


def rgb_hist(img_bgr, bins=RGB_DEFAULTS["bins"], mask=None):
    """Independent 1D histograms in true R, G, B order, each summing to 1."""
    return _three_channel_hist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bins, mask)


def lab_hist(img_bgr, bins=LAB_DEFAULTS["bins"], mask=None, normalization="none"):
    """L, a, b histograms in [0, 256), optionally photometrically normalised."""
    return _three_channel_hist(color_image(img_bgr, "lab", normalization, mask), bins, mask)



# ----------------------------------------------------------------------------
# Texture descriptors (BGR input, grayscale intensity structure)
# ----------------------------------------------------------------------------
def _texture_gray(image, resize):
    """Square canonical resolution; direct functions follow the repo BGR convention."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)


def lbp_descriptor(img_bgr, radius=1, points=8, resize=256):
    """Rotation-invariant uniform circular LBP (P+2 bins), L1 normalised.

    Bilinear circular samples compare >= the centre. Uniform patterns have at
    most two circular bit transitions; their bin is the number of set bits.
    Other patterns share bin P+1. Exclude a radius-wide image boundary.
    """
    gray = _texture_gray(img_bgr, resize).astype(np.float32)
    margin = int(np.ceil(radius))
    if 2 * margin >= resize:
        raise ValueError("LBP radius leaves no valid pixels")
    yy, xx = np.mgrid[margin:resize-margin, margin:resize-margin].astype(np.float32)
    centre = gray[margin:resize-margin, margin:resize-margin]
    bits = []
    for p in range(points):
        angle = 2 * np.pi * p / points
        sample = cv2.remap(gray, (xx + radius * np.cos(angle)).astype(np.float32),
                           (yy - radius * np.sin(angle)).astype(np.float32), cv2.INTER_LINEAR)
        bits.append(sample >= centre)
    bits = np.stack(bits)
    transitions = np.sum(bits != np.roll(bits, 1, axis=0), axis=0)
    codes = np.where(transitions <= 2, bits.sum(axis=0), points + 1)
    return _norm(np.bincount(codes.ravel(), minlength=points + 2))


def multiscale_lbp_descriptor(img_bgr, radii=(1, 2, 3), points=(8, 16, 24), resize=256):
    """Concatenate equally weighted LBP histograms at paired radii/point counts."""
    if len(radii) != len(points) or not len(radii):
        raise ValueError("radii and points must have matching nonzero lengths")
    return np.concatenate([lbp_descriptor(img_bgr, radius, count, resize)
                           for radius, count in zip(radii, points)]) / len(radii)


def dct_descriptor(img_bgr, n_coeffs=64, resize=256):
    """Signed low-frequency DCT AC coefficients in zigzag order, L2 normalised.

    Remove mean brightness and standardise contrast before the orthonormal DCT.
    DC is omitted. Use L1/Euclidean distance, not histogram distances.
    Combine with the existing spatial grid for block DCT.
    """
    gray = _texture_gray(img_bgr, resize).astype(np.float32)
    gray -= gray.mean()
    gray /= max(float(gray.std()), 1e-6)
    coeff = cv2.dct(gray)
    values = []
    for diagonal in range(1, 2 * resize - 1):
        pairs = [(y, diagonal-y) for y in range(max(0, diagonal-resize+1),
                                               min(resize-1, diagonal)+1)]
        if diagonal % 2 == 0:
            pairs.reverse()
        for y, x in pairs:
            values.append(coeff[y, x])
            if len(values) == n_coeffs:
                vector = np.asarray(values, dtype=np.float32)
                return vector / max(float(np.linalg.norm(vector)), 1e-6)
    raise ValueError("n_coeffs exceeds available AC coefficients")


def wavelet_descriptor(img_bgr, levels=3, resize=256):
    """Haar detail statistics: mean absolute value and std of LH/HL/HH per level.

    Standardise brightness/contrast first. Approximation band is decomposed
    recursively; no approximation/DC statistics are included. L2 normalised.
    This uses an orthonormal separable Haar transform, requiring no dependency.
    """
    gray = _texture_gray(img_bgr, resize).astype(np.float32)
    gray -= gray.mean()
    gray /= max(float(gray.std()), 1e-6)
    features = []
    for _ in range(levels):
        low = (gray[:, 0::2] + gray[:, 1::2]) / np.sqrt(2)
        high = (gray[:, 0::2] - gray[:, 1::2]) / np.sqrt(2)
        approximation = (low[0::2] + low[1::2]) / np.sqrt(2)
        details = [(low[0::2] - low[1::2]) / np.sqrt(2),
                   (high[0::2] + high[1::2]) / np.sqrt(2),
                   (high[0::2] - high[1::2]) / np.sqrt(2)]
        for band in details:
            features.extend([np.abs(band).mean(), band.std()])
        gray = approximation
    vector = np.asarray(features, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


TEXTURE_METHODS = {"lbp", "multiscale_lbp", "dct", "wavelet"}

METHODS = {"ycbcr": ycbcr_hist, "hsv": hsv_hist, "hsv_baseline": hsv_baseline_hist,
           "rgb": rgb_hist, "lab": lab_hist, "hsv_hs2d": hsv_hs2d,
           "lab_ab2d": lab_ab2d, "hsv_3d": hsv_3d, "lab_3d": lab_3d}
METHODS.update(lbp=lbp_descriptor, multiscale_lbp=multiscale_lbp_descriptor,
               dct=dct_descriptor, wavelet=wavelet_descriptor)
# Methods whose histograms can be restricted to foreground pixels.
MASKABLE_METHODS = {"rgb", "lab", "ycbcr", "lab_ab2d", "lab_3d"}
DEFAULTS = {"ycbcr": YCBCR_DEFAULTS, "hsv": HSV_DEFAULTS, "hsv_baseline": HSV_BASELINE_DEFAULTS,
            "rgb": RGB_DEFAULTS, "lab": LAB_DEFAULTS,
            "hsv_hs2d": dict(bins=(16, 16), s_min=10, v_min=40, filter_hue=True,
                             achromatic_bin=True, hue_smoothing=False, include_v=True,
                             v_bins=8, chroma_weight=0.7),
            "lab_ab2d": dict(bins=(16, 16), include_l=True, l_bins=8, chroma_weight=0.7),
            "hsv_3d": dict(bins=(8, 8, 4)), "lab_3d": dict(bins=(4, 8, 8))}

DEFAULTS.update(lbp=dict(radius=1, points=8, resize=256),
                multiscale_lbp=dict(radii=(1, 2, 3), points=(8, 16, 24), resize=256),
                dct=dict(n_coeffs=64, resize=256), wavelet=dict(levels=3, resize=256))


def full_params(method, **params):
    """Default parameters of `method` updated with the given ones."""
    return resolve_params(method, params)


def method_tag(method, **params):
    """Canonical tag encoding every effective parameter, independent of defaults."""
    p = full_params(method, **params)
    if method in JOINT_METHODS or method in TEXTURE_METHODS or "spatial" in p:
        return hashed_tag(method, p)
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
    if p.get("normalization") == "photometric":
        tag += "_norm-photometric"
    return tag


def compute_descriptor(img_bgr, method, mask=None, trim="block", **params):
    """Descriptor of an image; with a mask, only foreground (nonzero) pixels are counted.

    For spatial layouts with a mask, the background around the painting is removed
    first: trim="block" deletes block rows/columns that are almost only background
    (trim_background_blocks), trim="bbox" crops to the foreground's bounding box
    (crop_to_foreground), and trim=None keeps the whole image.
    Photometric preparation uses the complete foreground before any trimming.
    """
    if (not isinstance(img_bgr, np.ndarray) or img_bgr.dtype != np.uint8 or
            img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or not img_bgr.size):
        raise ValueError("Expected a nonempty H x W x 3 uint8 BGR image")
    params = full_params(method, **params)
    spatial = params.pop("spatial", None)
    if mask is not None:
        if method not in MASKABLE_METHODS:
            raise ValueError(f"Foreground masks are supported only for {sorted(MASKABLE_METHODS)}, not {method}")
        if not isinstance(mask, np.ndarray) or mask.shape != img_bgr.shape[:2]:
            raise ValueError("Mask and image sizes differ")
        mask = (mask > 0).astype(np.uint8) * 255
        if not np.any(mask):
            raise ValueError("Foreground mask is empty; cannot compute a painting descriptor")
    extract_color = METHODS[method]
    if params.get("normalization") == "photometric":
        params.pop("normalization")
        # Prepare before trimming/partitioning, using the complete foreground.
        img_bgr = color_image(img_bgr, "ycbcr" if method == "ycbcr" else "lab", "photometric", mask)
        extract_color = {"lab": _three_channel_hist, "ycbcr": ycbcr_hist_from_color,
                         "lab_ab2d": lab_ab2d_from_color, "lab_3d": lab_3d_from_color}[method]
    if method in TEXTURE_METHODS and spatial is not None:
        # Coefficients are signed and/or L2 normalised, so do not use the
        # histogram mass normalisation in spatial_descriptor.
        spatial = resolve_spatial(spatial)
        return np.concatenate([
            extract_color(img_bgr[ys, xs], **params) * (weight / grid**2)
            for grid, weight in zip(spatial["grids"], spatial["level_weights"])
            for ys, xs in block_slices(img_bgr.shape, grid)]).astype(np.float32)
    if mask is None:
        if spatial is None:
            return extract_color(img_bgr, **params)
        return spatial_descriptor(img_bgr, lambda block: extract_color(block, **params), spatial)
    if spatial is None:
        return extract_color(img_bgr, mask=mask, **params)
    spatial = resolve_spatial(spatial)
    if trim == "block":
        img_bgr, mask = trim_background_blocks(img_bgr, mask, max(spatial["grids"]))
    elif trim == "bbox":
        img_bgr, mask = crop_to_foreground(img_bgr, mask)
    elif trim is not None:
        raise ValueError("trim must be 'block', 'bbox' or None")
    return spatial_descriptor(img_bgr, lambda block, block_mask: extract_color(block, mask=block_mask, **params),
                              spatial, mask)


def compute_folder(folder, method, *, preprocess=None, **params):
    """Compute descriptors for all JPEG images (.jpg/.jpeg, any case) in `folder`.

    preprocess optionally filters each loaded BGR image in memory before extraction.
    Returns (ids: int64 [N], desc: float32 [N x D]); row i corresponds to ids[i].
    """
    paths, ids = list_images(folder)
    if not paths:
        raise ValueError(f"No JPEG images (.jpg/.jpeg) found in {folder}")
    def extract(path):
        image = load_image(path)
        if preprocess is not None:
            image = preprocess(image)
        return compute_descriptor(image, method, **params)

    desc = [extract(p)
            for p in tqdm(paths, desc=f"{Path(folder).name}/{method}")]
    return np.asarray(ids, dtype=np.int64), np.stack(desc).astype(np.float32)


def channel_sizes(method, **params):
    """Number of bins of each channel's slice in the descriptor."""
    p = full_params(method, **params)
    if method in JOINT_METHODS or method in TEXTURE_METHODS or "spatial" in p:
        raise ValueError("channel_sizes applies only to global independent 1D histograms")
    sizes = [int(b) for b in p["bins"]]
    if method == "ycbcr" and p["chroma_overflow"]:
        sizes[1] += 2
        sizes[2] += 2
    return sizes
