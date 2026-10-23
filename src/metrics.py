"""Mean Average Precision at K (mAP@K).

apk / mapk reproduce the logic of the reference implementation by Ben Hamner:
https://github.com/benhamner/Metrics  (Python/ml_metrics/average_precision.py)
"""
import numpy as np
from numbers import Integral


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
    if len(actual) != len(predicted):
        raise ValueError(f"Ground truth has {len(actual)} queries but predictions have {len(predicted)}")
    if len(actual) == 0:
        raise ValueError("mAP requires at least one query; evaluation collections cannot be empty")
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


def _sequence(value):
    return isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim > 0)


def _integer_ids(values):
    if not _sequence(values) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError("Image IDs must be nonnegative integers")
    return [int(v) for v in values]


def normalize_painting_gt(gt):
    """Canonical photo -> painting -> relevant IDs, separate from legacy GT.

    Course W3 rows [id_a, id_b] mean TWO paintings, not two relevant IDs for
    one painting. Explicit rows [[id_a], [id_b]] are also accepted. Multiple
    relevant IDs for one painting require that explicit additional level.
    Annotation painting order is retained verbatim.
    """
    if not _sequence(gt) or not len(gt):
        raise ValueError("Painting GT requires a nonempty list of photo entries")
    result = []
    for row in gt:
        if not _sequence(row):
            raise ValueError("Each photo GT must list its paintings")
        if all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in row):
            result.append([[v] for v in _integer_ids(row)])
        elif all(_sequence(v) for v in row):
            paintings = [_integer_ids(v) for v in row]
            if any(not v for v in paintings):
                raise ValueError("Each GT painting needs at least one corresponding ID")
            result.append(paintings)
        else:
            raise ValueError("Ambiguous GT row: use IDs per painting or explicit relevant-ID lists")
    return result


def validate_predictions(results, *, nested=False, k=None, query_count=None, painting_counts=None, db_ids=None):
    """Validate structural correspondence and convert IDs to plain Python ints."""
    if not _sequence(results) or (query_count is not None and len(results) != query_count):
        raise ValueError("Prediction query count does not match query images")
    if k is not None and (isinstance(k, bool) or not isinstance(k, Integral) or k < 1):
        raise ValueError("k must be a positive integer")
    if painting_counts is not None and len(painting_counts) != len(results):
        raise ValueError("Painting counts must match the number of queries")
    allowed = None if db_ids is None else set(_integer_ids(db_ids))
    canonical = []
    for index, row in enumerate(results):
        if not _sequence(row):
            raise ValueError("Each query prediction must be a list")
        rankings = row if nested else [row]
        if painting_counts is not None and len(rankings) != painting_counts[index]:
            raise ValueError("Ranking count does not match detected painting count")
        converted = []
        for ranking in rankings:
            ids = _integer_ids(ranking)
            if k is not None and len(ids) != k:
                raise ValueError(f"Each painting ranking must contain exactly {k} IDs")
            if len(ids) != len(set(ids)) or (allowed is not None and not set(ids) <= allowed):
                raise ValueError("Ranking contains duplicate or unknown database IDs")
            converted.append(ids)
        canonical.append(converted if nested else converted[0])
    return canonical


def evaluate_painting_retrieval(gt, predictions, *, matches=None, assume_ordered=False, ks=(1, 5)):
    """Painting-weighted AP; missed GT paintings contribute zero.

    matches[photo][GT painting] is a predicted index or None, obtained from
    geometry, NEVER retrieval labels/scores. Without matches, multi-painting
    order must be explicitly confirmed and counts must agree. Extras do not
    contribute AP; detection-aware AP additionally penalizes extra detections.
    """
    actual = normalize_painting_gt(gt)
    predictions = validate_predictions(predictions, nested=True, query_count=len(actual))
    if matches is not None and len(matches) != len(actual):
        raise ValueError("Geometric matches must have one entry per query")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or k < 1 for k in ks):
        raise ValueError("AP cutoffs must be positive integers")
    rows, sums = [], {k: 0.0 for k in ks}
    total = sum(map(len, actual))
    if not total:
        raise ValueError("Painting mAP requires at least one GT painting")
    for image_index, (truth, predicted) in enumerate(zip(actual, predictions)):
        if matches is not None:
            assignment = matches[image_index]
        elif not predicted:
            assignment = [None] * len(truth)
        elif len(truth) == len(predicted) and (assume_ordered or len(truth) <= 1):
            assignment = list(range(len(truth)))
        else:
            raise ValueError("Ambiguous painting alignment: provide frame-based matches; do not zip unequal detections")
        if not _sequence(assignment) or len(assignment) != len(truth):
            raise ValueError("Matches must contain one predicted index or None per GT painting")
        used = [v for v in assignment if v is not None]
        if (any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < len(predicted) for v in used)
                or len(set(used)) != len(used)):
            raise ValueError("Geometric matches must be valid one-to-one prediction indices")
        aps = {k: [float(apk(relevant, [] if match is None else predicted[match], k))
                   for relevant, match in zip(truth, assignment)] for k in ks}
        for k in ks:
            sums[k] += sum(aps[k])
        rows.append(dict(query_index=image_index, detected=len(predicted), expected=len(truth),
                         missing=len(truth)-len(used), extra=len(predicted)-len(used),
                         count_correct=len(predicted) == len(truth),
                         matches=[None if v is None else int(v) for v in assignment],
                         ap={str(k): values for k, values in aps.items()},
                         rankings=predicted, ground_truth=truth,
                         all_top1_correct=(len(used) == len(truth) == len(predicted) and
                                           all(predicted[m] and predicted[m][0] in relevant
                                               for relevant, m in zip(truth, assignment)))))
    extra = sum(r["extra"] for r in rows)
    return dict(paintings=total, detected=sum(map(len, predictions)),
                missing=sum(r["missing"] for r in rows), extra=extra,
                count_accuracy=float(np.mean([r["count_correct"] for r in rows])),
                all_top1_accuracy=float(np.mean([r["all_top1_correct"] for r in rows])),
                **{f"map{k}": sums[k] / total for k in ks},
                detection_aware_map={str(k): sums[k] / (total + extra) for k in ks}, per_image=rows)
