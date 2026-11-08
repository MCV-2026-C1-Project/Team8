"""Unknown-positive evaluation on reusable, ranked painting evidence; no retrieval work."""
from dataclasses import dataclass
from local_matching_config import resolve_rejection, number


@dataclass(frozen=True)
class PaintingEvidence:
    top_ids: tuple
    verified: bool
    inlier_count: int


def painting_evidence(ranked, k):
    k = number(k, "k", integer=True)
    if not ranked:
        return PaintingEvidence((), False, 0)
    best = ranked[0].verification
    return PaintingEvidence(tuple(c.database_id for c in ranked[:k]), bool(best.success), best.inlier_count)


def predict(evidence, config=None):
    config = resolve_rejection(config)
    if config["mode"] == "disabled":
        return list(evidence.top_ids)
    if not evidence.top_ids or not evidence.verified or evidence.inlier_count < config["min_inliers"]:
        return [-1]
    return list(evidence.top_ids)


def apply_threshold(evidence, threshold):
    config = resolve_rejection(dict(mode="min_inliers", min_inliers=threshold))
    return [predict(item, config) for item in evidence]


def evaluate_unknown(labels, predictions):
    """Unknown is positive. Undefined precision/recall/F1 are zero, explicitly flagged.

    Labels and predictions must already be geometrically aligned. This metric
    does not measure whether an accepted museum ID is correct.
    """
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Evaluation requires nonempty aligned labels and predictions")
    tp = fp = fn = tn = 0
    for label, prediction in zip(labels, predictions):
        if isinstance(label, bool) or not isinstance(label, int) or label < -1:
            raise ValueError("Labels must be museum IDs or -1")
        if (not prediction or any(isinstance(x, bool) or not isinstance(x, int) or x < -1 for x in prediction)
                or (-1 in prediction and prediction != [-1]) or len(set(prediction)) != len(prediction)):
            raise ValueError("Predictions must be distinct museum IDs or exactly [-1]")
        actual, rejected = label == -1, prediction == [-1]
        tp += int(actual and rejected)
        fp += int(not actual and rejected)
        fn += int(actual and not rejected)
        tn += int(not actual and not rejected)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision_defined=bool(tp+fp), recall_defined=bool(tp+fn), f1_defined=bool(2*tp+fp+fn),
                paintings=len(labels), known=fp+tn, unknown=tp+fn, accepted=fn+tn, rejected=tp+fp,
                positive_class="unknown", zero_division=0)


def select_threshold(evidence, labels):
    """Test each decision boundary; F1 ties choose the smallest positive threshold."""
    if len(evidence) != len(labels) or not labels:
        raise ValueError("Threshold selection requires aligned development evidence")
    thresholds = sorted({1, *(e.inlier_count+1 for e in evidence if e.verified and e.top_ids)})
    rows = [dict(min_inliers=t, **evaluate_unknown(labels, apply_threshold(evidence, t))) for t in thresholds]
    best = max(rows, key=lambda r: (r["f1"], -r["min_inliers"]))
    return dict(min_inliers=best["min_inliers"], metrics=best, thresholds=rows,
                tie_break="smallest threshold", provenance="development-selected; not independent test performance")
