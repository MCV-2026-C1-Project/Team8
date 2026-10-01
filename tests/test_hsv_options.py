"""HSV options preserve the baseline and affect only its Hue histogram."""
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descriptors import compute_descriptor, method_tag
from run_experiments import iter_configurations


def legacy_hsv(image, bins=(32, 16, 8), s_min=40, v_min=40):
    """Reference computation from before the optional Hue transformations."""
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = ((s > s_min) & (v > v_min)).astype(np.uint8) * 255
    histograms = []
    for channel, size, upper, selected in zip((h, s, v), bins, (180, 256, 256),
                                             (mask, None, None)):
        hist = cv2.calcHist([channel], [0], selected, [size], [0., float(upper)])
        hist = hist.ravel().astype(np.float32)
        mass = hist.sum()
        histograms.append(hist / mass if mass > 0 else hist)
    return np.concatenate(histograms)


@pytest.mark.parametrize("params", [{}, dict(bins=(7, 5, 3), s_min=70, v_min=100)])
def test_hsv_default_matches_legacy_exactly(params):
    rng = np.random.default_rng(2026)
    for image in (rng.integers(0, 256, (17, 23, 3), dtype=np.uint8),
                  np.zeros((3, 4, 3), dtype=np.uint8)):
        expected = legacy_hsv(image, **params)
        np.testing.assert_array_equal(compute_descriptor(image, "hsv", **params), expected)
        np.testing.assert_array_equal(compute_descriptor(
            image, "hsv", **params, hue_valid_weight=False, hue_smoothing=False), expected)


@pytest.mark.parametrize("valid_pixels", [0, 1, 4])
@pytest.mark.parametrize("smoothing", [False, True])
def test_hsv_weight_uses_valid_fraction_and_preserves_sv(valid_pixels, smoothing):
    image = np.full((1, 4, 3), 80, dtype=np.uint8)  # Grey pixels are invalid for H.
    image[0, :valid_pixels] = [0, 0, 255]
    baseline = compute_descriptor(image, "hsv", hue_smoothing=smoothing)
    weighted = compute_descriptor(image, "hsv", hue_smoothing=smoothing, hue_valid_weight=True)
    np.testing.assert_array_equal(weighted[:32], baseline[:32] * (valid_pixels / 4))
    np.testing.assert_allclose(weighted[:32].sum(), valid_pixels / 4)
    np.testing.assert_array_equal(weighted[32:], baseline[32:])
    assert weighted.dtype == np.float32
    assert np.isfinite(weighted).all()


@pytest.mark.parametrize("hue,occupied", [(0, 0), (179, 7)])
def test_hsv_smoothing_wraps_both_boundaries(hue, occupied):
    image = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    baseline = compute_descriptor(image, "hsv", bins=(8, 4, 4))
    smoothed = compute_descriptor(image, "hsv", bins=(8, 4, 4), hue_smoothing=True)
    assert baseline[occupied] == 1
    expected = np.zeros(8, dtype=np.float32)
    expected[occupied] = 0.5
    expected[(occupied - 1) % 8] = 0.25
    expected[(occupied + 1) % 8] = 0.25
    np.testing.assert_array_equal(smoothed[:8], expected)
    np.testing.assert_array_equal(smoothed[8:], baseline[8:])


@pytest.mark.parametrize("weighting", [False, True])
@pytest.mark.parametrize("h_bins", [1, 2, 7, 32])
def test_hsv_smoothing_preserves_mass_and_sv(h_bins, weighting):
    image = np.random.default_rng(7).integers(0, 256, (15, 17, 3), dtype=np.uint8)
    params = dict(bins=(h_bins, 8, 4), s_min=150, v_min=150, hue_valid_weight=weighting)
    baseline = compute_descriptor(image, "hsv", **params)
    smoothed = compute_descriptor(image, "hsv", **params, hue_smoothing=True)
    np.testing.assert_allclose(smoothed[:h_bins].sum(), baseline[:h_bins].sum(), rtol=1e-6)
    np.testing.assert_array_equal(smoothed[h_bins:], baseline[h_bins:])


def test_hsv_variant_tags_and_grid_are_distinct():
    variants = list(product([False, True], repeat=2))
    tags = [method_tag("hsv", hue_valid_weight=w, hue_smoothing=s) for w, s in variants]
    assert tags[0] == method_tag("hsv") == "hsv_32-16-8"
    assert len(set(tags)) == 4
    for w, s in variants:
        assert method_tag("hsv", bins=(7, 5, 3), s_min=20, v_min=50,
                          hue_valid_weight=w, hue_smoothing=s).startswith("hsv_7-5-3_s20v50")
    grid = [dict(descriptor="hsv",
                 grid={"bins": [(32, 16, 8)], "s_min": [40], "v_min": [40],
                       "hue_valid_weight": [False, True], "hue_smoothing": [False, True]},
                 measures=["hellinger"])]
    configs = iter_configurations(grid)
    assert len(configs) == 4  # Cache tags must not merge distinct Hue options.
    assert [method_tag(method, **params) for method, params, _ in configs] == tags
