"""Top-K retrieval and descriptor loading helpers (Week 1, Task 3)."""
import json
import pickle
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from descriptors import channel_sizes, compute_folder, full_params, method_tag
from distances import rank
from metrics import normalize_gt

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptors"
DATA_DIR = ROOT / "data"
BBDD_DIR = DATA_DIR / "BBDD"


def retrieve(query_desc, db_desc, db_ids, measure_name, k, channel_sizes=None):
    """For each query, the IDs of the k most similar database images.

    query_desc: (Q, D) query descriptors, rows in sorted query-file order.
    db_desc:    (N, D) database descriptors.
    db_ids:     (N,)   database image IDs; db_ids[i] is the image of row i.
    channel_sizes: bins per channel, required only for Wasserstein.
    Returns a list of Q lists of k plain Python ints (BBDD IDs), most similar first.
    """
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


def save_descriptor_cache(path, ids, desc, method, params):
    """Save arrays and effective descriptor identity, including for precomputation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, desc=desc, **_cache_metadata(method, params))


def load_or_compute(name, folder, method, params, desc_dir=DESC_DIR):
    """Reuse only canonical caches with matching metadata and valid array shapes.

    Missing, legacy, mismatched or unreadable caches are recomputed from images.
    Old filenames are never renamed or used as a fallback.
    """
    params = full_params(method, **params)
    expected = _cache_metadata(method, params)
    path = Path(desc_dir) / f"{name}_{expected['tag']}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as d:
                if (d["method"].item() != expected["method"] or d["tag"].item() != expected["tag"] or
                        json.loads(d["params_json"].item()) != json.loads(expected["params_json"])):
                    raise ValueError("descriptor identity does not match the requested configuration")
                ids, desc = d["ids"], d["desc"]
                if (ids.ndim != 1 or desc.ndim != 2 or
                        desc.shape != (len(ids), sum(channel_sizes(method, **params))) or
                        not np.issubdtype(ids.dtype, np.integer) or
                        not np.isfinite(desc).all()):
                    raise ValueError("invalid descriptor arrays")
                return ids, desc
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
            print(f"Ignoring invalid cache {path.name}: {error}. Recomputing from {folder}")
    else:
        print(f"{path.name} not found, computing it from {folder}")
    ids, desc = compute_folder(folder, method, **params)
    save_descriptor_cache(path, ids, desc, method, params)
    return ids, desc


def load_gt(query_folder):
    """Ground truth from gt_corresps.pkl: gt[i] = list of BBDD IDs for the i-th query."""
    with open(Path(query_folder) / "gt_corresps.pkl", "rb") as f:
        return normalize_gt(pickle.load(f))


def query_folder(query_set):
    """'qsd1' -> data/qsd1_w1; an existing folder path is used as is."""
    p = Path(query_set)
    return p if p.is_dir() else DATA_DIR / f"{query_set}_w1"
