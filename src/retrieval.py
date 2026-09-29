"""Top-K retrieval and descriptor loading helpers (Week 1, Task 3)."""
import pickle
from pathlib import Path

import numpy as np

from descriptors import compute_folder, method_tag
from distances import rank
from metrics import normalize_gt

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptors"
DATA_DIR = ROOT / "data"
BBDD_DIR = DATA_DIR / "BBDD"


def retrieve(query_desc, db_desc, db_ids, measure_name, k):
    """For each query, the IDs of the k most similar database images.

    query_desc: (Q, D) query descriptors, rows in sorted query-file order.
    db_desc:    (N, D) database descriptors.
    db_ids:     (N,)   database image IDs; db_ids[i] is the image of row i.
    Returns a list of Q lists of k plain Python ints (BBDD IDs), most similar first.
    """
    results = []
    for q in query_desc:
        rows = rank(q, db_desc, measure_name)[:k]        # row indices, best first
        results.append([int(i) for i in db_ids[rows]])   # rows -> image IDs, np.int64 -> int
    return results


def load_or_compute(name, folder, method, params, desc_dir=DESC_DIR):
    """Load descriptors <name>_<tag>.npz, computing and saving them if missing.

    The tag encodes the method and parameters (see descriptors.method_tag), so
    BBDD and the query set are always described with identical settings.
    Returns (ids, desc).
    """
    path = Path(desc_dir) / f"{name}_{method_tag(method, **params)}.npz"
    if path.exists():
        d = np.load(path)
        return d["ids"], d["desc"]
    print(f"{path.name} not found, computing it from {folder}")
    ids, desc = compute_folder(folder, method, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, desc=desc)
    return ids, desc


def load_gt(query_folder):
    """Ground truth from gt_corresps.pkl: gt[i] = list of BBDD IDs for the i-th query."""
    with open(Path(query_folder) / "gt_corresps.pkl", "rb") as f:
        return normalize_gt(pickle.load(f))


def query_folder(query_set):
    """'qsd1' -> data/qsd1_w1; an existing folder path is used as is."""
    p = Path(query_set)
    return p if p.is_dir() else DATA_DIR / f"{query_set}_w1"
