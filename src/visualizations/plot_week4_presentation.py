"""Figures for the final presentation, produced with the real Week 4 modules.

Panels are label-free: the slide deck carries the captions. Pipeline examples come
from QSD2-W3; unknown and failure examples from QSD1-W4. Needs data/ and the SIFT cache.
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from descriptors import load_image
from local_feature_cache import load_or_compute_database_features
from local_feature_queries import prepare_painting_crops
from local_features import extract_local_features
from local_geometry import verify_geometry
from local_matching import filter_matches, tentative_matches
from local_matching_config import resolve_run_config
from local_retrieval import rank_candidates, verify_candidate
from painting_detection import detect_paintings, resolve_detection_config

OUT = Path(__file__).resolve().parent / "presentation_w4"
QUERIES, DEV, DB = ROOT / "data/qsd2_w3", ROOT / "data/qsd1_w4", ROOT / "data/BBDD"
SEGMENTATION = json.loads((ROOT / "configs/week3/task3_selected.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "configs/week4/local_matching_baseline.json").read_text(encoding="utf-8"))
with (QUERIES / "gt_corresps.pkl").open("rb") as stream:
    LABELS = pickle.load(stream)
GREEN, ORANGE, INDIGO, BRASS = (117, 138, 14), (30, 97, 200), (199, 88, 90), (74, 178, 226)
BRIGHT_GREEN, BRIGHT_ORANGE = (120, 235, 60), (0, 150, 255)
# Final system selected by the 144-configuration QSD1-W4 grid.
FINAL = dict(BASE, feature_method="sift", matching=dict(BASE["matching"], ratio_threshold=.65),
             geometry=dict(BASE["geometry"], estimator="usac_magsac", reprojection_threshold=2.5))
METHOD_COLOURS = dict(sift=GREEN, orb=ORANGE, akaze=INDIGO)
EXAMPLES = dict(detection=8, keypoints=15, matching=3, known=4, unknown=26)


def save(name, image, long_side=1100):
    scale = min(1, long_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{name}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def fit(image, height):
    return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height), interpolation=cv2.INTER_AREA)


def query_crop(query_id, index=0, folder=QUERIES):
    image = load_image(folder / f"{query_id:05d}.jpg")
    return prepare_painting_crops(image, query_id, SEGMENTATION, crop_inset=.03)[index]


def detection_panels(meta):
    query_id = EXAMPLES["detection"]
    image = load_image(QUERIES / f"{query_id:05d}.jpg")
    mask, paintings = detect_paintings(image, resolve_detection_config(SEGMENTATION))
    boxed = image.copy()
    for painting in paintings:
        x, y, r, b = painting.bbox
        cv2.rectangle(boxed, (x, y), (r, b), BRIGHT_GREEN, max(3, image.shape[1] // 250), cv2.LINE_AA)
    save("detection_original", image)
    save("detection_mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    save("detection_boxes", boxed)
    crops = prepare_painting_crops(image, query_id, SEGMENTATION, crop_inset=.03)
    for index, crop in enumerate(crops):
        save(f"detection_crop_{index}", crop.image)
    meta["detection"] = dict(query=query_id, paintings=len(crops), labels=LABELS[query_id])


def keypoint_panels(meta):
    crop = query_crop(EXAMPLES["keypoints"])
    meta["keypoints"] = dict(query=EXAMPLES["keypoints"], shape=list(crop.image.shape[:2]))
    scale = 900 / max(crop.image.shape[:2])
    base = cv2.resize(crop.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    config = json.loads((ROOT / "configs/week4/local_features.json").read_text(encoding="utf-8"))
    for method, colour in METHOD_COLOURS.items():
        features = extract_local_features(crop.image, method, **config[method])
        canvas = (base * .55 + 60).astype(np.uint8)
        for x, y in features.points * scale:
            cv2.circle(canvas, (round(x), round(y)), 5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (round(x), round(y)), 5, colour, 2, cv2.LINE_AA)
        save(f"keypoints_{method}", canvas)
        meta["keypoints"][method] = dict(count=len(features.keypoints),
                                         bytes=int(features.descriptors.nbytes // max(len(features.keypoints), 1)))


def side_by_side(query, database, height=800, gap=24):
    left, right = fit(query, height), fit(database, height)
    canvas = np.full((height, left.shape[1] + gap + right.shape[1], 3), 255, np.uint8)
    canvas[:, :left.shape[1]], canvas[:, left.shape[1] + gap:] = left, right
    return canvas, height / query.shape[0], height / database.shape[0], left.shape[1] + gap


def draw_lines(canvas, q, d, sq, sd, offset, colour, width=2):
    overlay = canvas.copy()
    for a, b in zip(q, d):
        start = (round(a[0] * sq), round(a[1] * sq))
        end = (round(b[0] * sd + offset), round(b[1] * sd))
        cv2.line(overlay, start, end, colour, width, cv2.LINE_AA)
        cv2.circle(overlay, start, 4, colour, -1, cv2.LINE_AA)
        cv2.circle(overlay, end, 4, colour, -1, cv2.LINE_AA)
    canvas[:] = cv2.addWeighted(overlay, .85, canvas, .15, 0)


def verified_panel(name, crop, database_image, verification, sample=None):
    matches = verification.matches
    canvas, sq, sd, offset = side_by_side(crop, database_image)
    keep = verification.inlier_mask
    if sample and keep.sum() > sample:
        chosen = np.random.default_rng(0).choice(np.flatnonzero(keep), sample, replace=False)
        keep = np.zeros_like(keep)
        keep[chosen] = True
    draw_lines(canvas, matches.query_points[keep], matches.database_points[keep], sq, sd, offset, BRIGHT_GREEN)
    save(name, canvas, 1500)


def all_candidates(features, database, config):
    return [verify_candidate(i, n, tentative_matches(features, f, compute_reverse=False), config["matching"],
                             config["geometry"], config["scoring"])
            for i, n, f in zip(database.ids, database.filenames, database.features)]


def matching_panels(meta, database, config):
    query_id = EXAMPLES["matching"]
    truth = LABELS[query_id][0]
    crop = query_crop(query_id)
    features = crop.extract("sift", **config["feature_params"]).features
    candidates = all_candidates(features, database, config)
    ranked = rank_candidates(candidates, "inlier_count")
    best = next(c for c in candidates if c.database_id == truth)
    image = load_image(DB / best.filename)
    tentative = tentative_matches(features, database.features[database.ids.index(truth)], compute_reverse=True)
    nearest = tentative.neighbors[:, 0]
    shown = np.random.default_rng(0).choice(np.flatnonzero(nearest >= 0), 200, replace=False)
    canvas, sq, sd, offset = side_by_side(crop.image, image)
    draw_lines(canvas, tentative.query_points[shown], tentative.database_points[nearest[shown]], sq, sd, offset, BRIGHT_ORANGE)
    save("matching_before", canvas, 1500)
    filtered = filter_matches(tentative, config["matching"])
    canvas, sq, sd, offset = side_by_side(crop.image, image)
    draw_lines(canvas, filtered.query_points, filtered.database_points, sq, sd, offset, BRIGHT_ORANGE)
    save("matching_after", canvas, 1500)
    verification = verify_geometry(filtered, config["geometry"])
    verified_panel("geometry_inliers", crop.image, image, verification, sample=120)
    mutual = filter_matches(tentative, dict(config["matching"], filter="ratio_mutual"))
    ransac = verify_geometry(filtered, dict(config["geometry"], estimator="ransac"))
    meta["matching"] = dict(query=query_id, truth=truth, tentative=tentative.candidate_count, ratio=filtered.count,
                            ratio_mutual=mutual.count, inliers=verification.inlier_count,
                            ransac_inliers=ransac.inlier_count, shown_before=len(shown),
                            top5=[dict(id=c.database_id, inliers=c.verification.inlier_count,
                                       coverage=c.scores["coverage_weighted"]) for c in ranked[:5]])


def unknown_panels(meta, database, config):
    """Real QSD1-W4 decisions: one known painting and one true unknown."""
    with (DEV / "gt_corresps.pkl").open("rb") as stream:
        labels = pickle.load(stream)
    results = {}
    for name, query_id in (("known", EXAMPLES["known"]), ("unknown", EXAMPLES["unknown"])):
        crop = query_crop(query_id, folder=DEV)
        features = crop.extract("sift", **config["feature_params"]).features
        best = rank_candidates(all_candidates(features, database, config), "inlier_count")[0]
        verified_panel(f"unknown_{name}", crop.image, load_image(DB / best.filename), best.verification, sample=120)
        results[name] = dict(query=query_id, truth=labels[query_id][0], top=best.database_id,
                             inliers=best.verification.inlier_count)
    meta["unknown"] = results


def failure_panels(meta, database, config):
    """Boxes saved from the detector before (3e8f9bc) and after (a0ea011) the Week 4 changes."""
    cases = json.loads((OUT / "segmentation_before_after.json").read_text(encoding="utf-8"))
    for case in cases:
        image = load_image(ROOT / "data" / case["ds"] / f"{case['name']}.jpg")
        for key, colour in (("bbox_old", ORANGE), ("bbox_new", GREEN)):
            canvas = image.copy()
            for x, y, r, b in case[key]:
                cv2.rectangle(canvas, (x, y), (r, b), (255, 255, 255), max(9, image.shape[1] // 90), cv2.LINE_AA)
                cv2.rectangle(canvas, (x, y), (r, b), colour, max(5, image.shape[1] // 160), cv2.LINE_AA)
            save(f"failure_{case['ds']}_{case['name']}_{key[5:]}", fit(canvas, 700))
    for query_id, truth in ((8, 128), (16, 11)):
        crop = query_crop(query_id, folder=DEV)
        features = crop.extract("sift", **config["feature_params"]).features
        candidate = all_candidates(features, database, config)[database.ids.index(truth)]
        verified_panel(f"failure_retrieval_{truth}", crop.image, load_image(DB / candidate.filename), candidate.verification)
        meta[f"failure_{truth}"] = dict(query=query_id, inliers=candidate.verification.inlier_count)


def hero_panel():
    crop = query_crop(9)
    features = extract_local_features(crop.image, "sift", nfeatures=450)
    scale = 1300 / crop.image.shape[0]
    canvas = cv2.resize(crop.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for (x, y), keypoint in zip(features.points * scale, features.keypoints):
        radius = int(np.clip(keypoint["size"] * scale * .8, 6, 42))
        cv2.circle(canvas, (round(float(x)), round(float(y))), radius, BRASS, 2, cv2.LINE_AA)
    save("hero_keypoints", canvas)


def main():
    OUT.mkdir(exist_ok=True)
    config = resolve_run_config(FINAL)
    database = load_or_compute_database_features(DB, "sift", **config["feature_params"])
    meta = {}
    hero_panel()
    detection_panels(meta)
    keypoint_panels(meta)
    matching_panels(meta, database, config)
    unknown_panels(meta, database, config)
    failure_panels(meta, database, config)
    (OUT / "figure_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
