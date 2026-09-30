"""Distance / similarity measures between histogram descriptors (Week 1, Task 2).

Every measure compares ONE query descriptor q (shape D) against the WHOLE
database db (shape N x D) and returns N scores, one per database image.
NumPy broadcasting does the work: q (D,) is broadcast against db (N, D), and
summing over axis=1 collapses the D bins, so there is no Python loop over images.

Distances  (euclidean, l1, chi2, wasserstein): lower = more similar.
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


def wasserstein(q, db, channel_sizes):
    """Mean 1D Wasserstein distance across independent histogram channels.

    Each channel uses equally spaced bin centres on [0, 1], spacing 1/bins.
    Thus channels with different units/bin counts have comparable distances.
    Positive channel masses are normalised locally for the CDF calculation;
    stored descriptors and other measures are unchanged. For each query/database
    pair, channels empty in either histogram are excluded from the average.
    A pair with no valid channels raises an error.
    """
    q, db = _check(q, db)
    if channel_sizes is None:
        raise ValueError("Wasserstein requires channel_sizes (bins per channel)")
    sizes = np.asarray(channel_sizes)
    if (sizes.ndim != 1 or sizes.size == 0 or
            not np.issubdtype(sizes.dtype, np.integer) or
            np.any(sizes <= 0) or sizes.sum() != q.size):
        raise ValueError("channel_sizes must be positive integers summing to descriptor length")
    if (not np.isfinite(q).all() or not np.isfinite(db).all() or
            np.any(q < 0) or np.any(db < 0)):
        raise ValueError("Wasserstein requires finite, non-negative histogram values")

    distances = np.zeros(db.shape[0], dtype=np.float64)
    valid_counts = np.zeros(db.shape[0], dtype=np.int64)
    start = 0
    for size in sizes:
        qc = q[start:start + size]
        dc = db[:, start:start + size]
        qm, dm = qc.sum(), dc.sum(axis=1, keepdims=True)
        valid = (qm > 0) & (dm[:, 0] > 0)
        if np.any(valid):
            # Integral of |CDF_q - CDF_db| between adjacent bin centres.
            delta = np.cumsum(dc[valid] / dm[valid] - qc / qm, axis=1)
            distances[valid] += np.abs(delta[:, :-1]).sum(axis=1) / size
            valid_counts[valid] += 1
        start += size
    if np.any(valid_counts == 0):
        rows = np.flatnonzero(valid_counts == 0).tolist()
        raise ValueError(f"Wasserstein has no valid channels for database rows {rows}; "
                         "at least one channel must have positive mass in both histograms")
    return distances / valid_counts


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
    "wasserstein": (wasserstein, False),
}


def rank(q, db, measure_name, channel_sizes=None):
    """Indices (rows) of db sorted from most to least similar to q.

    Similarities are negated so a single ascending sort works for both kinds.
    A stable sort keeps ties in database order, so results are reproducible.
    Note: these are ROW indices; map them to image IDs with ids[rank(...)].
    """
    if measure_name not in MEASURES:
        raise ValueError(f"Unknown measure '{measure_name}'. Options: {list(MEASURES)}")
    func, higher_is_better = MEASURES[measure_name]
    scores = (func(q, db, channel_sizes) if measure_name == "wasserstein"
              else func(q, db))
    return np.argsort(-scores if higher_is_better else scores, kind="stable")
