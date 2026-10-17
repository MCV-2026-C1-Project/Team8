"""Top-K retrieval and descriptor loading helpers (Week 1, Task 3)."""
import json
import hashlib
import pickle
import re
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np
from tqdm import tqdm

from descriptors import compute_descriptor, compute_folder, full_params, method_tag, list_images, load_image
from descriptor_config import descriptor_dim, canonical_json, measure_channels, resolve_descriptor_config, positive_int
from distances import MEASURES, rank
from metrics import normalize_gt
from remove_background import foreground_mask

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptors"
DATA_DIR = ROOT / "data"
BBDD_DIR = DATA_DIR / "BBDD"


def retrieve(query_desc, db_desc, db_ids, measure_name, k, channel_sizes=None, *, allow_truncate=False):
    """For each query, the IDs of the k most similar database images.

    query_desc: (Q, D) query descriptors, rows in sorted query-file order.
    db_desc:    (N, D) database descriptors.
    db_ids:     (N,)   database image IDs; db_ids[i] is the image of row i.
    channel_sizes: bins per channel, required only for Wasserstein.
    K must be a positive integer no larger than N. Explicit allow_truncate=True
    permits K > N and returns all N IDs; it never permits an empty database.
    Returns a list of Q lists of k plain Python ints (BBDD IDs), most similar first.
    """
    k = positive_int(k, "k")
    if not isinstance(allow_truncate, bool):
        raise ValueError("allow_truncate must be a boolean")
    query_desc, db_desc, db_ids = np.asarray(query_desc), np.asarray(db_desc), np.asarray(db_ids)
    if query_desc.ndim != 2 or db_desc.ndim != 2:
        raise ValueError("Query and database descriptors must be 2D arrays (images, dimensions)")
    if db_desc.shape[0] == 0:
        raise ValueError("The database must contain at least one descriptor")
    if query_desc.shape[1] != db_desc.shape[1] or db_desc.shape[1] == 0:
        raise ValueError("Query and database descriptor dimensions must agree and be positive")
    if db_ids.ndim != 1 or len(db_ids) != len(db_desc):
        raise ValueError("Database IDs must be a 1D array matching the number of descriptor rows")
    if not np.issubdtype(db_ids.dtype, np.integer):
        raise ValueError("Database IDs must be integers")
    if k > len(db_desc) and not allow_truncate:
        raise ValueError(f"k={k} exceeds database size {len(db_desc)}; reduce k or explicitly enable allow_truncate")
    if measure_name not in MEASURES:
        raise ValueError(f"Unknown measure {measure_name!r}")
    k = min(k, len(db_desc))
    results = []
    for q in query_desc:
        rows = rank(q, db_desc, measure_name, channel_sizes)[:k]  # row indices, best first
        results.append([int(i) for i in db_ids[rows]])   # rows -> image IDs, np.int64 -> int
    return results


def _cache_metadata(method, params):
    """Deterministic identity shared by cache writers and validation."""
    params = full_params(method, **params)
    return dict(method=method, tag=method_tag(method, **params),
                params_json=json.dumps(params, sort_keys=True, separators=(",", ":"),
                                       default=lambda value: value.tolist(), allow_nan=False))


def source_identity(folder):
    """Prevent cache reuse across different folders or modified image collections."""
    paths, _ = list_images(folder)
    if not paths:
        raise ValueError(f"No JPEG images (.jpg/.jpeg) found in {folder}")
    manifest = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    payload = canonical_json([str(Path(folder).resolve()), manifest])
    return hashlib.sha256(payload.encode()).hexdigest()


def save_descriptor_cache(path, ids, desc, method, params, folder=None):
    """Save arrays and effective descriptor identity, including for precomputation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, desc=desc, source=source_identity(folder) if folder else "",
             **_cache_metadata(method, params))


def load_or_compute(name, folder, method, params, desc_dir=DESC_DIR):
    """Reuse only canonical caches with matching metadata and valid array shapes.

    Missing, legacy, mismatched or unreadable caches are recomputed from images.
    Old filenames are never renamed or used as a fallback.
    """
    params = full_params(method, **params)
    expected = _cache_metadata(method, params)
    source = source_identity(folder)
    path = Path(desc_dir) / f"{name}_{expected['tag']}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as d:
                if (d["method"].item() != expected["method"] or d["tag"].item() != expected["tag"] or
                        json.loads(d["params_json"].item()) != json.loads(expected["params_json"])):
                    raise ValueError("descriptor identity does not match the requested configuration")
                if d["source"].item() != source:
                    raise ValueError("source image folder or file metadata changed")
                ids, desc = d["ids"], d["desc"]
                if (ids.ndim != 1 or desc.ndim != 2 or
                        desc.shape != (len(ids), descriptor_dim(method, **params)) or
                        not np.issubdtype(ids.dtype, np.integer) or
                        ids.tolist() != list_images(folder)[1] or
                        np.any(desc < 0) or
                        not np.isfinite(desc).all()):
                    raise ValueError("invalid descriptor arrays")
                return ids, desc
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
            print(f"Ignoring invalid cache {path.name}: {error}. Recomputing from {folder}")
    else:
        print(f"{path.name} not found, computing it from {folder}")
    ids, desc = compute_folder(folder, method, **params)
    save_descriptor_cache(path, ids, desc, method, params, folder)
    return ids, desc


def load_gt(query_folder, gt_path=None):
    """Ground truth from gt_corresps.pkl: gt[i] = list of BBDD IDs for the i-th query."""
    with open(gt_path or Path(query_folder) / "gt_corresps.pkl", "rb") as f:
        return normalize_gt(pickle.load(f))


def query_folder(query_set):
    """Resolve a dataset ID or actual folder and fail clearly when it is missing.

    Bare course aliases (qsd1/qst1) default to Week 1. Explicit week suffixes
    are retained; qsd1_w2 never silently redirects to qsd1_w1.
    """
    p = Path(query_set)
    if not p.is_dir() and re.fullmatch(r"qs[dt]\d+(?:_w\d+)?", str(query_set)):
        p = DATA_DIR / (str(query_set) if "_w" in str(query_set) else f"{query_set}_w1")
    if not p.is_dir():
        hint = (" QSD1-W2 uses the QSD1-W1 cropped images; use qsd1 or data/qsd1_w1 if that is your available folder."
                if str(query_set) == "qsd1_w2" else "")
        raise FileNotFoundError(f"Query directory does not exist or is not a directory: {p.resolve()}.{hint}")
    return p


def load_configured_data(config, db_dir=BBDD_DIR, query_dir=DATA_DIR / "qsd1_w1", desc_dir=DESC_DIR):
    """Shared extraction/cache path for experiments and final retrieval."""
    config = resolve_descriptor_config(config)
    method, params = config["descriptor"], config["params"]
    db_ids, db_desc = load_or_compute("bbdd", db_dir, method, params, desc_dir)
    q_ids, q_desc = load_or_compute(Path(query_dir).name.split("_")[0].lower(),
                                   query_dir, method, params, desc_dir)
    return db_ids, db_desc, q_ids, q_desc


def query_masks(folder, mask_source="pred"):
    """Foreground masks (255 = painting) of the query images, in list_images order.

    pred: computed with foreground_mask; gt: read from <stem>.png next to each image.
    """
    if mask_source not in ("pred", "gt"):
        raise ValueError("mask_source must be pred or gt")
    paths, ids = list_images(folder)
    if not paths:
        raise ValueError(f"No JPEG images (.jpg/.jpeg) found in {folder}")
    images, masks = [], []
    for path in tqdm(paths, desc=f"{Path(folder).name}/masks ({mask_source})"):
        image = load_image(path)
        if mask_source == "pred":
            mask = foreground_mask(image)
        else:
            mask = cv2.imread(str(path.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Ground-truth mask not found: {path.with_suffix('.png')}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask and image sizes differ: {path}")
        if not np.any(mask):
            raise ValueError(f"Foreground mask is empty: {path}; cannot compute a painting descriptor")
        images.append(image)
        masks.append(mask)
    return paths, np.asarray(ids, dtype=np.int64), images, masks


def compute_masked_queries(config, images, masks, trim="block"):
    """Foreground-only query descriptors. Not cached: the cache tag ignores masks."""
    config = resolve_descriptor_config(config)
    desc = [compute_descriptor(image, config["descriptor"], mask=mask, trim=trim, **config["params"])
            for image, mask in zip(images, masks)]
    return np.stack(desc).astype(np.float32)


def retrieve_config(config, measure, db_dir=BBDD_DIR, query_dir=DATA_DIR / "qsd1_w1",
                    desc_dir=DESC_DIR, k=10, *, allow_truncate=False, masks=None, trim="block"):
    """masks=(query ids, images, masks) computes foreground-only query descriptors."""
    config = resolve_descriptor_config(config)
    channels = measure_channels(config["descriptor"], config["params"], measure)
    k = positive_int(k, "k")
    if masks is None:
        db_ids, db_desc, q_ids, q_desc = load_configured_data(config, db_dir, query_dir, desc_dir)
    else:
        db_ids, db_desc = load_or_compute("bbdd", db_dir, config["descriptor"], config["params"], desc_dir)
        q_ids, images, query_masks_ = masks
        q_desc = compute_masked_queries(config, images, query_masks_, trim)
    return q_ids, retrieve(q_desc, db_desc, db_ids, measure, k, channels, allow_truncate=allow_truncate)


def validate_gt_order(q_ids, gt):
    if len(q_ids) != len(gt) or q_ids.tolist() != list(range(len(gt))):
        raise ValueError("Query IDs must follow ground-truth order (0, 1, ...) with one GT entry per image")


def save_results(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump([[int(value) for value in row] for row in results], stream)
