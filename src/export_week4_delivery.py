"""Week 4 Task 4: QST1-W4 predictions from the system selected on QSD1-W4.

Run from the repository root with OpenCV < 5 (requirements.txt). Writes
results/week4/QST1/method1/result.pkl: image -> painting -> K ranked IDs, or [-1].
Also writes method1_split_fixed/, where two detections in one image that are both
accepted with the same top ID count as one painting, keeping the better supported.
"""
import json
import pickle
from pathlib import Path

from descriptors import list_images, load_image
from local_feature_cache import load_or_compute_database_features
from local_feature_queries import prepare_painting_crops
from local_matching import tentative_matches
from local_matching_config import resolve_run_config
from local_retrieval import rank_candidates, verify_candidate

ROOT = Path(__file__).resolve().parents[1]
QUERIES, DB, OUT = ROOT / "data/qst1_w4", ROOT / "data/BBDD", ROOT / "results/week4/QST1"
BASE = json.loads((ROOT / "configs/week4/local_matching_baseline.json").read_text(encoding="utf-8"))
SEGMENTATION = json.loads((ROOT / "configs/week3/task3_selected.json").read_text(encoding="utf-8"))
FINAL = dict(BASE, feature_method="sift", matching=dict(BASE["matching"], ratio_threshold=.65),
             geometry=dict(BASE["geometry"], estimator="usac_magsac", reprojection_threshold=2.5))
MIN_INLIERS, K = 5, 10


def predict_image(path, query_id, config, database):
    paintings = []
    for crop in prepare_painting_crops(load_image(path), query_id, SEGMENTATION, crop_inset=config["crop"]["inset"]):
        features = crop.extract("sift", **config["feature_params"]).features
        ranked = rank_candidates([verify_candidate(i, n, tentative_matches(features, f, compute_reverse=False),
                                                   config["matching"], config["geometry"], config["scoring"])
                                  for i, n, f in zip(database.ids, database.filenames, database.features)], "inlier_count")
        best = ranked[0].verification
        unknown = not best.success or best.inlier_count < MIN_INLIERS
        paintings.append(dict(ids=[-1] if unknown else [int(c.database_id) for c in ranked[:K]], inliers=int(best.inlier_count)))
    return paintings


def merge_splits(paintings):
    if len(paintings) == 2 and paintings[0]["ids"][0] == paintings[1]["ids"][0] != -1:
        return [max(paintings, key=lambda p: p["inliers"])]
    return paintings


def check(result, images, database_ids):
    allowed = set(database_ids)
    assert isinstance(result, list) and len(result) == images
    for row in result:
        assert isinstance(row, list) and 1 <= len(row) <= 2
        for ids in row:
            assert all(type(x) is int for x in ids)
            assert ids == [-1] or (len(ids) == K and len(set(ids)) == K and set(ids) <= allowed)


def save(folder, result, images, database_ids):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "result.pkl"
    with path.open("xb") as stream:  # Never overwrite a delivered file.
        pickle.dump(result, stream)
    with path.open("rb") as stream:
        loaded = pickle.load(stream)
    assert loaded == result
    check(loaded, images, database_ids)
    flat = [ids for row in loaded for ids in row]
    print(f"{path}: {len(loaded)} images, {len(flat)} paintings, {sum(ids == [-1] for ids in flat)} unknown")


def main():
    config = resolve_run_config(FINAL)
    database = load_or_compute_database_features(DB, "sift", **config["feature_params"])
    paths, ids = list_images(QUERIES)
    assert ids == list(range(len(paths))), "Blind images must be the complete sorted inventory"
    rows = [predict_image(path, query_id, config, database) for path, query_id in zip(paths, ids)]
    for query_id, row in zip(ids, rows):
        print(query_id, [(p["ids"][0], p["inliers"]) for p in row], flush=True)
    save(OUT / "method1", [[p["ids"] for p in row] for row in rows], len(paths), database.ids)
    save(OUT / "method1_split_fixed", [[p["ids"] for p in merge_splits(row)] for row in rows], len(paths), database.ids)


if __name__ == "__main__":
    main()
