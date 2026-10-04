"""Large HSV sweep on QSD1: bins x Hue S/V mask x valid-Hue weighting x measures.

Each image is converted to HSV once. Per-value counts are kept (S, V once;
H once per (s_min, v_min) mask) and every bin configuration is obtained by
summing them with OpenCV's own value->bin mapping, so descriptors are
identical to descriptors.hsv_hist (checked on a sample of configurations).

s_min / v_min = -1 means "no threshold" (mask is S > s_min, V > v_min).
With both at -1 Hue uses all pixels and weighting has no effect, so only
the unweighted variant is run.

Run: python src/sweep_hsv.py --output config_results/hsv_sweep.csv
"""
import argparse
import csv
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from descriptors import compute_descriptor, list_images, load_image
from distances import MEASURES
from metrics import mapk
from retrieval import BBDD_DIR, DATA_DIR, ROOT, load_gt, retrieve

H_BINS = [8, 16, 32, 64, 180]
SV_BINS = [8, 16, 32, 64, 256]
THRESHOLDS = [-1, 10, 20, 40, 60]
FIELDS = ["descriptor", "bins", "parameters", "measure", "map1", "map5"]


def bin_lut(bins, rng):
    """OpenCV's bin index for every uint8 value (-1 if out of range)."""
    lut = np.full(256, -1)
    for v in range(256):
        h = cv2.calcHist([np.array([[v]], np.uint8)], [0], None, [bins], [float(rng[0]), float(rng[1])])
        if h.sum():
            lut[v] = int(np.argmax(h))
    return lut


def rebin(counts, bins, rng):
    """(N, 256) per-value counts -> (N, bins) histograms normalised to sum 1."""
    lut = bin_lut(bins, rng)
    onehot = np.zeros((256, bins))
    onehot[lut >= 0, lut[lut >= 0]] = 1
    h = counts @ onehot
    s = h.sum(axis=1, keepdims=True)
    return np.divide(h, s, out=np.zeros_like(h), where=s > 0)


def value_counts(folder):
    """Per-value S, V counts and masked H counts / valid fractions per threshold pair."""
    paths, ids = list_images(folder)
    n = len(paths)
    S, V = np.zeros((n, 256)), np.zeros((n, 256))
    H = {t: np.zeros((n, 256)) for t in product(THRESHOLDS, THRESHOLDS)}
    frac = {t: np.zeros(n) for t in H}
    for i, p in enumerate(tqdm(paths, desc=Path(folder).name)):
        h, s, v = cv2.split(cv2.cvtColor(load_image(p), cv2.COLOR_BGR2HSV))
        S[i] = np.bincount(s.ravel(), minlength=256)
        V[i] = np.bincount(v.ravel(), minlength=256)
        for (s_min, v_min) in H:
            mask = (s > s_min) & (v > v_min)
            H[s_min, v_min][i] = np.bincount(h[mask], minlength=256)
            frac[s_min, v_min][i] = mask.mean()
    return np.asarray(ids), S, V, H, frac, paths


def descriptors(data, bins, s_min, v_min, weight):
    _, S, V, H, frac, _ = data
    hh = rebin(H[s_min, v_min], bins[0], (0, 180))
    if weight:
        hh = hh * frac[s_min, v_min][:, None]
    return np.hstack([hh, rebin(S, bins[1], (0, 256)), rebin(V, bins[2], (0, 256))]).astype(np.float32)


def configurations():
    for bins in product(H_BINS, SV_BINS, SV_BINS):
        for s_min, v_min in product(THRESHOLDS, THRESHOLDS):
            for weight in ([False] if (s_min, v_min) == (-1, -1) else [False, True]):
                yield bins, s_min, v_min, weight


def check(data, rng, n_configs=15):
    """Compare against descriptors.hsv_hist for random configs and images."""
    configs = list(configurations())
    for k in rng.choice(len(configs), n_configs, replace=False):
        bins, s_min, v_min, weight = configs[k]
        desc = descriptors(data, bins, s_min, v_min, weight)
        for i in rng.choice(len(desc), 3, replace=False):
            ref = compute_descriptor(load_image(data[5][i]), "hsv", bins=bins, s_min=s_min,
                                     v_min=v_min, hue_valid_weight=weight, hue_smoothing=False)
            assert np.allclose(desc[i], ref, atol=1e-6), f"Mismatch for {configs[k]}, image {i}"
    print(f"Check passed: {n_configs} configs match descriptors.hsv_hist")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "config_results" / "hsv_sweep.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.overwrite:
        ap.error(f"{args.output} already exists; use --overwrite to replace it")

    query_dir = DATA_DIR / "qsd1_w1"
    gt = load_gt(query_dir)
    db, q = value_counts(BBDD_DIR), value_counts(query_dir)
    if q[0].tolist() != list(range(len(gt))):
        raise ValueError("QSD1 query IDs must follow ground-truth order (0, 1, ...)")
    check(db, np.random.default_rng(0))

    configs = list(configurations())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for bins, s_min, v_min, weight in tqdm(configs, desc="sweep"):
            db_desc = descriptors(db, bins, s_min, v_min, weight)
            q_desc = descriptors(q, bins, s_min, v_min, weight)
            params = json.dumps(dict(hue_smoothing=False, hue_valid_weight=weight,
                                     s_min=s_min, v_min=v_min), sort_keys=True)
            for measure in MEASURES:
                pred = retrieve(q_desc, db_desc, db[0], measure, 10, channel_sizes=list(bins))
                writer.writerow(dict(descriptor="hsv", bins="-".join(map(str, bins)), parameters=params,
                                     measure=measure, map1=float(mapk(gt, pred, 1)),
                                     map5=float(mapk(gt, pred, 5))))
    print(f"Saved {len(configs) * len(MEASURES)} rows to {args.output}")


if __name__ == "__main__":
    main()
