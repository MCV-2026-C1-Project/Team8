"""Small numerical and integration checks, independent of course datasets."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descriptors import compute_descriptor, method_tag
from distances import MEASURES, rank, wasserstein
from retrieval import retrieve
import retrieval
import run_experiments as experiments


@pytest.mark.parametrize("method", ["rgb", "lab"])
@pytest.mark.parametrize("bins", [(16, 16, 16), (4, 8, 12)])
def test_new_descriptor_channels(method, bins):
    # Non-symmetric BGR colour detects accidental BGR ordering in RGB.
    image = np.full((5, 7, 3), [10, 80, 240], dtype=np.uint8)
    descriptor = compute_descriptor(image, method, bins=bins)
    assert descriptor.shape == (sum(bins),)
    assert descriptor.dtype == np.float32
    colour = ([240, 80, 10] if method == "rgb" else
              cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[0, 0])
    start = 0
    for value, size in zip(colour, bins):
        expected = np.zeros(size)
        expected[int(value) * size // 256] = 1
        np.testing.assert_array_equal(descriptor[start:start + size], expected)
        start += size
    assert method_tag(method, bins=bins) == method + "_" + "-".join(map(str, bins))


def test_existing_tags():
    assert method_tag("hsv") == "hsv_32-16-8"
    assert method_tag("ycbcr") == "ycbcr_8-32-32"
    assert method_tag("hsv", s_min=30, v_min=50) == "hsv_32-16-8_s30v50"
    assert method_tag("ycbcr", chroma_range=(64, 192)) == "ycbcr_8-32-32_c64-192"


@pytest.mark.parametrize("name,expected", [
    ("euclidean", [np.sqrt(2), 0, np.sqrt(2)]),
    ("l1", [2, 0, 2]),
    ("chi2", [2 / (1 + 1e-10), 0, 2 / (1 + 1e-10)]),
    ("intersection", [0, 1, 0]),
    ("hellinger", [0, 1, 0]),
])
def test_existing_measures_and_stable_ranking(name, expected):
    q = np.array([1., 0., 0.])
    db = np.array([[0., 1., 0.], q, [0., 0., 1.]])
    np.testing.assert_allclose(MEASURES[name][0](q, db), expected)
    np.testing.assert_array_equal(rank(q, db, name), [1, 0, 2])
    np.testing.assert_array_equal(rank(q, db, name, [1, 2]), [1, 0, 2])
    assert retrieve([q], db, np.array([20, 80, 99]), name, 2) == [[80, 20]]


def test_wasserstein_properties_and_channel_boundaries():
    q = np.array([1., 0., 0., 0., 1., 0.])
    nearby = np.array([0., 1., 0., 0., 1., 0.])
    far = np.array([0., 0., 0., 1., 1., 0.])
    sizes = [4, 2]
    scores = wasserstein(q, np.array([q, nearby, far]), sizes)
    np.testing.assert_allclose(scores, [0, 0.125, 0.375])
    assert np.all(scores >= 0)
    np.testing.assert_allclose(wasserstein(nearby, q, sizes), [scores[1]])
    np.testing.assert_allclose(wasserstein(q * 3, nearby * 2, sizes), [scores[1]])
    np.testing.assert_array_equal(rank(q, [far, nearby, q], "wasserstein", sizes), [2, 1, 0])
    assert retrieve([q], np.array([far, nearby, q]), np.array([17, 32, 80]),
                    "wasserstein", 2, sizes) == [[80, 32]]


def test_wasserstein_known_transport_and_independent_channels():
    # Moving half the mass by one bin of width 1/4 costs 1/8.
    np.testing.assert_allclose(wasserstein([0.5, 0.5, 0, 0], [0, 1, 0, 0], [4]), [0.125])
    # Opposite shifts in adjacent channels must not cancel across the boundary.
    np.testing.assert_allclose(wasserstein([1, 0, 0, 1], [0, 1, 1, 0], [2, 2]), [0.5])
    # Single-bin distributions have no transport distance.
    np.testing.assert_allclose(wasserstein([1, 1], [2, 3], [1, 1]), [0])


@pytest.mark.parametrize("sizes", [None, [4], [0, 2], [1.5, 0.5]])
def test_wasserstein_requires_valid_boundaries(sizes):
    with pytest.raises(ValueError):
        wasserstein([1, 0], [0, 1], sizes)


@pytest.mark.parametrize("q", [[0, 0], [-1, 2], [np.nan, 1]])
def test_wasserstein_rejects_undefined_distributions(q):
    with pytest.raises(ValueError):
        wasserstein(q, [0, 1], [2])


@pytest.mark.parametrize("q,db,expected", [
    ([1, 0, 1, 0], [1, 0, 0, 1], 0.25),  # Both channels available.
    ([0, 0, 1, 0], [1, 0, 0, 1], 0.5),   # Empty query channel.
    ([1, 0, 1, 0], [0, 0, 0, 1], 0.5),   # Empty database channel.
    ([0, 0, 1, 0], [0, 0, 0, 1], 0.5),   # Both empty in one channel.
])
def test_wasserstein_averages_only_available_channels(q, db, expected):
    with np.errstate(all="raise"):
        np.testing.assert_allclose(wasserstein(q, db, [2, 2]), [expected])
        np.testing.assert_allclose(wasserstein(db, q, [2, 2]), [expected])


def test_wasserstein_valid_channel_count_is_per_pair():
    q = [1, 0, 1, 0]
    db = [[1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0]]
    np.testing.assert_allclose(wasserstein(q, db, [2, 2]), [0.25, 0.5, 0.5])


@pytest.mark.parametrize("q,db", [
    ([0, 0, 0, 0], [[0, 0, 0, 0]]),
    ([1, 0, 0, 0], [[1, 0, 0, 0], [0, 0, 1, 0]]),
])
def test_wasserstein_no_shared_valid_channel_raises(q, db):
    with pytest.raises(ValueError, match="no valid channels"):
        wasserstein(q, db, [2, 2])


def test_wasserstein_with_grey_hsv_image():
    grey = compute_descriptor(np.full((8, 8, 3), 80, dtype=np.uint8), "hsv")
    red = compute_descriptor(np.full((8, 8, 3), [0, 0, 255], dtype=np.uint8), "hsv")
    assert not grey[:32].any()
    scores = wasserstein(grey, [grey, red], [32, 16, 8])
    assert scores[0] == 0 and scores[1] > 0
    np.testing.assert_allclose(scores[1:], wasserstein(grey[32:], red[32:], [16, 8]))


def test_default_grid_uses_common_measures():
    configs = experiments.iter_configurations(experiments.EXPERIMENTS)
    assert [method for method, _, _ in configs] == ["hsv", "ycbcr", "rgb", "lab"]
    assert all(measures == list(MEASURES) for _, _, measures in configs)


def test_grid_product_and_duplicate_merging():
    grid = [dict(descriptor="hsv", grid={"bins": [(8, 8, 8)], "s_min": [20, 40]},
                 measures=["l1", "l1"]),
            dict(descriptor="hsv", grid={"bins": [(8, 8, 8)], "s_min": [20]},
                 measures=["hellinger"])]
    configs = experiments.iter_configurations(grid)
    assert len(configs) == 2
    assert configs[0][1]["s_min"] == 20
    assert configs[0][2] == ["l1", "hellinger"]
    assert configs[1][2] == ["l1"]


def test_runner_reuses_cache_and_evaluates(tmp_path, monkeypatch):
    # Exercise real histogram extraction, cache, retrieval and metric functions.
    data = tmp_path / "data"
    for folder in (data / "BBDD", data / "qsd1_w1"):
        folder.mkdir(parents=True)
        for i, colour in enumerate(([0, 0, 255], [0, 255, 0], [255, 0, 0])):
            assert cv2.imwrite(str(folder / f"{i:05d}.jpg"),
                               np.full((8, 8, 3), colour, dtype=np.uint8))
    import pickle
    with (data / "qsd1_w1" / "gt_corresps.pkl").open("wb") as f:
        pickle.dump([[0], [1], [2]], f)
    monkeypatch.setattr(experiments, "DATA_DIR", data)
    monkeypatch.setattr(experiments, "BBDD_DIR", data / "BBDD")
    original = retrieval.compute_folder
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "compute_folder", counted)
    grid = [dict(descriptor="rgb", grid={"bins": [(4, 4, 4)]},
                 measures=["l1", "wasserstein"])]
    rows = experiments.run_experiments(grid, desc_dir=tmp_path / "cache")
    assert len(calls) == 2  # BBDD + QSD1, not once per measure.
    assert len(rows) == 2
    assert all(row["map1"] == row["map5"] == 1 for row in rows)
    assert experiments.run_experiments(grid, desc_dir=tmp_path / "cache") == rows
    assert len(calls) == 2  # Second run reads the saved descriptors.
