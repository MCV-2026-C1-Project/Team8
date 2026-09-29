"""Mean Average Precision at K (mAP@K) (Week 1, Task 3).

apk / mapk reproduce the logic of the reference implementation by Ben Hamner:
https://github.com/benhamner/Metrics  (Python/ml_metrics/average_precision.py)
"""
import numpy as np


def apk(actual, predicted, k=10):
    """Average precision at k between one list of relevant IDs and one ranked list.

    actual:    list of relevant (ground-truth) IDs, order does not matter.
    predicted: list of retrieved IDs, most similar first.

    Walks down the top-k predictions; every time a relevant ID appears (for the
    first time), precision at that position (hits so far / position) is added.
    The sum is divided by min(len(actual), k).
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # relevant and not a duplicate
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """Mean of apk over all queries.

    actual:    list of lists of relevant IDs, one per query.
    predicted: list of lists of retrieved IDs, one per query (same order).
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def normalize_gt(gt):
    """Make sure every ground-truth entry is a list of plain ints (as mapk expects).

    QSD1's gt_corresps.pkl is already [[120], [170], ...]; a bare int entry
    (e.g. [120, 170, ...]) is wrapped into a one-element list.
    """
    out = []
    for entry in gt:
        if isinstance(entry, (list, tuple, np.ndarray)):
            out.append([int(x) for x in entry])
        else:
            out.append([int(entry)])
    return out
