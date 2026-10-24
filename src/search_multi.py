"""Week 3: choose the multi-painting retrieval configuration on QSD2-W3.

Paintings are detected once (Task 3 configuration) and matched to the GT frames
geometrically. Every candidate (descriptor x measure x crop inset x noise policy)
is then scored on the detected crops with painting-level mAP, exactly as
run_retrieval.py --multi-painting does, plus on QSD1-W3 whole images as a sanity
check. Texture/colour late fusion of the best candidates is scored last.

Run: python src/search_multi.py --config configs/week3/qsd2_search.json
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from descriptor_config import canonical_json, measure_channels
from descriptors import TEXTURE_METHODS, compute_descriptor, list_images, load_image, method_tag
from evaluate_background import instance_diagnostics, load_instance_annotations, polygon_masks
from experiment_config import expand_experiments
from filter_noise import query_preprocessor, resolve_filter_config
from metrics import evaluate_painting_retrieval, mapk, normalize_gt, normalize_painting_gt
from painting_detection import detect_paintings, resolve_detection_config
from retrieval import (BBDD_DIR, DESC_DIR, ROOT, distance_matrix, fuse_distances, inset_box,
                       load_or_compute, rank_distances)

K = 10
FIELDS = ["candidate", "descriptor", "measure", "inset", "query_filter", "qsd2_map1", "qsd2_map5",
          "qsd1_map1", "qsd1_map5", "fusion_weight", "config_json", "fusion_json"]


def detect_set(folder, segmentation, frames):
    """Original images, detected boxes and frame-based GT matches, computed once."""
    paths, _ = list_images(folder)
    annotations = load_instance_annotations(frames, paths)
    images, boxes, matches = [], [], []
    for path in tqdm(paths, desc=f"{folder.name}/detection"):
        image = load_image(path)
        mask, paintings = detect_paintings(image, segmentation)
        truth = polygon_masks(annotations[path.name], mask.shape)
        info = instance_diagnostics([p.mask for p in paintings], truth, 0.5, return_matches=True)
        images.append(image)
        boxes.append([p.bbox for p in paintings])
        matches.append(info["matches"])
    return images, boxes, matches


def crop_variants(images, boxes, insets, filters):
    """(inset, filter label) -> flat list of filtered crops, in photo/painting order."""
    variants = {}
    for inset in insets:
        for policy in filters:
            preprocess = query_preprocessor(policy)
            crops = []
            for image, photo_boxes in zip(images, boxes):
                for bbox in photo_boxes:
                    x, y, right, bottom = inset_box(bbox, inset)
                    crop = image[y:bottom, x:right]
                    crops.append(crop if preprocess is None else preprocess(crop))
            variants[(inset, canonical_json(policy))] = crops
    return variants


def rankings(distance, db_ids, degenerate):
    return rank_distances(distance, db_ids, K, degenerate)


def regroup(flat, boxes):
    nested, offset = [], 0
    for photo_boxes in boxes:
        nested.append(flat[offset:offset + len(photo_boxes)])
        offset += len(photo_boxes)
    return nested


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "week3" / "qsd2_search.json")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "week3" / "qsd2_search.csv")
    ap.add_argument("--export-best", type=Path, default=ROOT / "results" / "week3" / "qsd2_best.json")
    ap.add_argument("--cache-dir", type=Path, default=DESC_DIR)
    args = ap.parse_args(argv)
    suite = json.loads(args.config.read_text(encoding="utf-8"))
    query_dir, sanity_dir = ROOT / suite["query_dir"], ROOT / suite["sanity_dir"]
    segmentation = resolve_detection_config(json.loads((ROOT / suite["segmentation"]).read_text(encoding="utf-8")))
    filters = [resolve_filter_config(f) for f in suite["query_filters"]]
    insets = [float(v) for v in suite["insets"]]
    jobs = expand_experiments(dict(experiments=suite["experiments"]))

    with (query_dir / "gt_corresps.pkl").open("rb") as stream:
        gt = normalize_painting_gt(pickle.load(stream))
    with (sanity_dir / "gt_corresps.pkl").open("rb") as stream:
        sanity_gt = normalize_gt(pickle.load(stream))
    images, boxes, matches = detect_set(query_dir, segmentation, ROOT / suite["gt_frames"])
    print(f"Detected {sum(map(len, boxes))} paintings in {len(images)} photos (GT: {sum(map(len, gt))})")
    variants = crop_variants(images, boxes, insets, filters)
    sanity_paths, _ = list_images(sanity_dir)
    sanity_images = [load_image(p) for p in sanity_paths]
    sanity_variants = crop_variants(sanity_images, [[(0, 0, i.shape[1], i.shape[0])] for i in sanity_images],
                                    insets, filters)

    rows, kept = [], {}
    for index, job in enumerate(jobs, 1):
        config = job["config"]
        method, params = config["descriptor"], config["params"]
        tag = method_tag(method, **params)
        print(f"[{index}/{len(jobs)}] {tag}")
        db_ids, db_desc = load_or_compute(None, BBDD_DIR, method, params, args.cache_dir)
        degenerate = ~np.any(db_desc != 0, axis=1)
        for (inset, policy), crops in variants.items():
            q_desc = np.stack([compute_descriptor(c, method, **params) for c in crops]).astype(np.float32)
            s_desc = np.stack([compute_descriptor(c, method, **params)
                               for c in sanity_variants[(inset, policy)]]).astype(np.float32)
            for measure in job["measures"]:
                channels = measure_channels(method, params, measure)
                distance = distance_matrix(q_desc, db_desc, measure, channels)
                sanity_distance = distance_matrix(s_desc, db_desc, measure, channels)
                result = evaluate_painting_retrieval(
                    gt, regroup(rankings(distance, db_ids, degenerate), boxes), matches=matches, ks=(1, 5))
                sanity = rankings(sanity_distance, db_ids, degenerate)
                key = (tag, measure, inset, policy)
                kept[key] = dict(distance=distance, sanity=sanity_distance, degenerate=degenerate,
                                 method=method, config=config)
                rows.append(dict(candidate=f"{tag}/{measure}", descriptor=method, measure=measure, inset=inset,
                                 query_filter=policy, qsd2_map1=result["map1"], qsd2_map5=result["map5"],
                                 qsd1_map1=float(mapk(sanity_gt, sanity, 1)),
                                 qsd1_map5=float(mapk(sanity_gt, sanity, 5)),
                                 fusion_weight="", config_json=canonical_json(config), fusion_json=""))
        best = max((r for r in rows if r["config_json"] == canonical_json(config)), key=lambda r: r["qsd2_map1"])
        print(f"  best: QSD2 {best['qsd2_map1']:.4f}/{best['qsd2_map5']:.4f} "
              f"({best['measure']}, inset {best['inset']}, {best['query_filter']})")

    # Late fusion: best texture candidates x best colour candidates, same crops.
    fusion = suite.get("fusion", {})
    rank_key = lambda r: (r["qsd2_map1"], r["qsd2_map5"], r["qsd1_map1"])
    single = sorted(rows, key=rank_key, reverse=True)
    texture = [r for r in single if r["descriptor"] in TEXTURE_METHODS][:fusion.get("top_texture", 3)]
    colour = [r for r in single if r["descriptor"] not in TEXTURE_METHODS][:fusion.get("top_colour", 1)]
    db_ids = list_images(BBDD_DIR)[1]
    db_ids = np.asarray(db_ids, dtype=np.int64)
    for t in texture:
        for c in colour:
            for inset, policy in variants:
                tk = (t["candidate"].rsplit("/", 1)[0], t["measure"], inset, policy)
                ck = (c["candidate"].rsplit("/", 1)[0], c["measure"], inset, policy)
                if tk not in kept or ck not in kept:
                    continue
                degenerate = kept[tk]["degenerate"] | kept[ck]["degenerate"]
                for weight in fusion.get("weights", [0.5]):
                    fused = fuse_distances([(kept[tk]["distance"], weight), (kept[ck]["distance"], 1 - weight)])
                    sanity = fuse_distances([(kept[tk]["sanity"], weight), (kept[ck]["sanity"], 1 - weight)])
                    result = evaluate_painting_retrieval(
                        gt, regroup(rankings(fused, db_ids, degenerate), boxes), matches=matches, ks=(1, 5))
                    sanity_rank = rankings(sanity, db_ids, degenerate)
                    rows.append(dict(
                        candidate=f"fusion[{t['candidate']} + {c['candidate']}]", descriptor="fusion",
                        measure=f"{t['measure']}+{c['measure']}", inset=inset, query_filter=policy,
                        qsd2_map1=result["map1"], qsd2_map5=result["map5"],
                        qsd1_map1=float(mapk(sanity_gt, sanity_rank, 1)),
                        qsd1_map5=float(mapk(sanity_gt, sanity_rank, 5)), fusion_weight=weight,
                        config_json=t["config_json"],
                        fusion_json=canonical_json(dict(config=json.loads(c["config_json"]), measure=c["measure"]))))

    rows.sort(key=rank_key, reverse=True)
    write_csv(args.output, rows)
    print(f"Saved {len(rows)} candidates to {args.output}. Top 10 (QSD2 mAP@1/@5 | QSD1 mAP@1):")
    for r in rows[:10]:
        print(f"  {r['qsd2_map1']:.4f}/{r['qsd2_map5']:.4f} | {r['qsd1_map1']:.4f}  {r['candidate']} "
              f"inset={r['inset']} filter={r['query_filter']} w={r['fusion_weight']}")
    best = rows[0]
    document = dict(config=json.loads(best["config_json"]), measure=best["measure"].split("+")[0],
                    select_by="qsd2_map1",
                    development_metrics=dict(qsd2_map1=best["qsd2_map1"], qsd2_map5=best["qsd2_map5"],
                                             qsd1_map1=best["qsd1_map1"], qsd1_map5=best["qsd1_map5"]),
                    provenance=dict(source=str(args.config), query_filter=json.loads(best["query_filter"]),
                                    crop_inset=best["inset"],
                                    fusion=None if not best["fusion_json"] else
                                    dict(json.loads(best["fusion_json"]), weight=best["fusion_weight"])))
    args.export_best.parent.mkdir(parents=True, exist_ok=True)
    args.export_best.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Exported the best candidate to {args.export_best}")


if __name__ == "__main__":
    main()
