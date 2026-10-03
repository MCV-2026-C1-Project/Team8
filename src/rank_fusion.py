"""Optional weighted rank-position fusion; no raw distance scores are combined."""
import numpy as np

from distances import rank


def normalise_weights(weights, count):
    """Require one finite non-negative weight per measure and positive total."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != count or count == 0:
        raise ValueError("Provide one weight per measure/ranking, with at least one measure")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Weights must be finite, non-negative, and not all zero")
    # Scaling first also avoids overflow when summing large finite weights.
    weights = weights / weights.max()
    return weights / weights.sum()


def fuse_rankings(rankings, weights):
    """Fuse complete permutations of DB row indices, returning best-first rows.

    Input shape is (measures, DB images). Convert orders to 1-based positions,
    average positions with normalised weights, and stably sort. Exact ties
    retain database row order, just as in the existing ranking implementation.
    """
    rankings = np.asarray(rankings)
    if rankings.ndim != 2 or rankings.shape[1] == 0:
        raise ValueError("Expected complete rankings with shape (measures, DB images)")
    weights = normalise_weights(weights, rankings.shape[0])
    n = rankings.shape[1]
    if (not np.issubdtype(rankings.dtype, np.integer) or
            not np.all(np.sort(rankings, axis=1) == np.arange(n))):
        raise ValueError("Each ranking must be a permutation of all DB row indices")
    positions = np.empty(rankings.shape, dtype=np.float64)
    np.put_along_axis(positions, rankings, np.arange(1, n + 1)[None, :], axis=1)
    scores = np.sum(weights[:, None] * positions, axis=0)
    return np.argsort(scores, kind="stable")


def rank_fusion(q, db, measures, weights, channel_sizes=None):
    """Rank one query with existing measures, then fuse their full rankings."""
    weights = normalise_weights(weights, len(measures))
    rankings = [rank(q, db, measure, channel_sizes) for measure in measures]
    return fuse_rankings(rankings, weights)


def best_relevant_ranks(ranked_ids, ground_truth):
    """Best 1-based relevant rank per query; absent relevance gets DB size + 1."""
    if len(ranked_ids) != len(ground_truth):
        raise ValueError("Ground truth and rankings must have the same number of queries")
    return np.asarray([
        next((i for i, image_id in enumerate(order, 1) if image_id in relevant), len(order) + 1)
        for order, relevant in zip(ranked_ids, ground_truth)
    ])
