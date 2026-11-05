"""Painting-weighted Week 4 metrics; missed GT paintings always contribute zero AP."""
from numbers import Integral
from metrics import apk
from local_development import parse_labels
from local_rejection import evaluate_unknown


def validate_nested(predictions, counts, *, allow_unknown=True):
    if not isinstance(predictions,list) or len(predictions) != len(counts):
        raise ValueError("Predictions must retain every photo")
    for row,count in zip(predictions,counts):
        if not isinstance(row,list) or len(row) != count:
            raise ValueError("Prediction painting counts disagree")
        for ids in row:
            if (not isinstance(ids,list) or not ids or any(isinstance(x,bool) or not isinstance(x,Integral) or x < -1 for x in ids)
                    or len(set(ids)) != len(ids) or (-1 in ids and (not allow_unknown or ids != [-1]))):
                raise ValueError("Each painting needs distinct museum IDs or exactly [-1]")


def evaluate_local(labels, rankings, predictions, alignments, k_values=(1,5)):
    labels = parse_labels(labels)
    if len(rankings) != len(labels) or len(alignments) != len(labels):
        raise ValueError("Labels, predictions and alignment require identical photo counts")
    counts = [len(row) for row in rankings]
    validate_nested(rankings, counts, allow_unknown=False)
    validate_nested(predictions, counts)
    if any(isinstance(k,bool) or not isinstance(k,Integral) or k < 1 for k in k_values):
        raise ValueError("Evaluation K must be positive integers")
    total = sum(map(len,labels)); known = sum(x != -1 for row in labels for x in row)
    missing = extra = 0
    sums = {name:{str(k):0. for k in k_values} for name in ("known_ranking_map","known_after_rejection_map","overall_map")}
    aligned_labels, aligned_predictions = [], []
    for truth, raw, predicted, alignment in zip(labels,rankings,predictions,alignments):
        matches = alignment["matches"]
        used = [m for m in matches if m is not None]
        if (len(matches) != len(truth) or len(set(used)) != len(used)
                or any(isinstance(m,bool) or not isinstance(m,Integral) or not 0 <= m < len(raw) for m in used)):
            raise ValueError("Alignment must be one-to-one GT->detection indices")
        missing += len(truth)-len(used); extra += len(raw)-len(used)
        for label,m in zip(truth,matches):
            if m is None:
                continue
            aligned_labels.append(label); aligned_predictions.append(predicted[m])
            for k in k_values:
                sums["overall_map"][str(k)] += apk([label],predicted[m],k)
                if label != -1:
                    sums["known_ranking_map"][str(k)] += apk([label],raw[m],k)
                    sums["known_after_rejection_map"][str(k)] += apk([label],predicted[m],k)
    result = {name:{k:value/(known if name.startswith("known") else total) if (known if name.startswith("known") else total) else None
                    for k,value in values.items()} for name,values in sums.items()}
    result.update(detection_aware_map={k:value/(total+extra) for k,value in sums["overall_map"].items()},
                  paintings=total, known=known, unknown=total-known, missing=missing, extra=extra,
                  unknown_detection=evaluate_unknown(aligned_labels,aligned_predictions) if aligned_labels else None,
                  unknown_metric_scope="aligned paintings only", retrieval_scope="all annotated paintings; missing AP=0",
                  provenance="QSD1-W4 development; not independent generalization performance")
    return result


def selection_key(result):
    m = result["metrics"]
    f1 = m["unknown_detection"]["f1"] if m["unknown_detection"] else 0.
    return (-m["overall_map"]["1"], -f1, -m["overall_map"]["5"], result["runtime"], result["id"])
