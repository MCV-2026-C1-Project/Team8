"""Distance / similarity measures between histogram descriptors (Week 1, Task 2).

Every measure compares ONE query descriptor q (shape D) against the WHOLE
database db (shape N x D) and returns N scores, one per database image.
NumPy broadcasting does the work: q (D,) is broadcast against db (N, D), and
summing over axis=1 collapses the D bins, so there is no Python loop over images.

Distances  (euclidean, l1, chi2):          lower  = more similar.
Similarities (intersection, hellinger):    higher = more similar.
"""
import numpy as np

EPS = 1e-10  # avoids 0/0 in chi-squared when a bin is empty in both histograms


def _check(q, db):
    """Convert to float64 arrays and validate shapes (q: D, db: N x D)."""
    q = np.asarray(q, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    if db.ndim == 1:                     # allow comparing two single descriptors
        db = db[None, :]
    if q.ndim != 1 or db.ndim != 2:
        raise ValueError(f"Expected q of shape (D,) and db of shape (N, D), got {q.shape} and {db.shape}")
    if q.shape[0] != db.shape[1]:
        raise ValueError(
            f"Descriptor length mismatch: query has {q.shape[0]} values, database has {db.shape[1]}. "
            "Are you comparing descriptors from different methods (e.g. YCbCr vs HSV)?")
    return q, db


# ----------------------------------------------------------------------------
# Distances (lower = more similar)
# ----------------------------------------------------------------------------
def euclidean(q, db):
    """sqrt( sum_i (q_i - h_i)^2 )"""
    q, db = _check(q, db)
    return np.sqrt(np.sum((db - q) ** 2, axis=1))


def l1(q, db):
    """sum_i |q_i - h_i|"""
    q, db = _check(q, db)
    return np.sum(np.abs(db - q), axis=1)


def chi2(q, db):
    """sum_i (q_i - h_i)^2 / (q_i + h_i + eps)

    Bins that are empty in both histograms give 0 / eps = 0, never NaN.
    """
    q, db = _check(q, db)
    return np.sum((db - q) ** 2 / (db + q + EPS), axis=1)


# ----------------------------------------------------------------------------
# Similarities (higher = more similar)
# ----------------------------------------------------------------------------
def intersection(q, db):
    """sum_i min(q_i, h_i)"""
    q, db = _check(q, db)
    return np.sum(np.minimum(db, q), axis=1)


def hellinger(q, db):
    """Hellinger kernel: sum_i sqrt(q_i * h_i)"""
    q, db = _check(q, db)
    return np.sum(np.sqrt(db * q), axis=1)


# name -> (function, higher_is_better)
MEASURES = {
    "euclidean": (euclidean, False),
    "l1": (l1, False),
    "chi2": (chi2, False),
    "intersection": (intersection, True),
    "hellinger": (hellinger, True),
}


def rank(q, db, measure_name):
    """Indices (rows) of db sorted from most to least similar to q.

    Similarities are negated so a single ascending sort works for both kinds.
    A stable sort keeps ties in database order, so results are reproducible.
    Note: these are ROW indices; map them to image IDs with ids[rank(...)].
    """
    if measure_name not in MEASURES:
        raise ValueError(f"Unknown measure '{measure_name}'. Options: {list(MEASURES)}")
    func, higher_is_better = MEASURES[measure_name]
    scores = func(q, db)
    return np.argsort(-scores if higher_is_better else scores, kind="stable")
