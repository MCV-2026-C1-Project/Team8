"""Exhaustive local retrieval; evidence can be re-filtered, re-verified and re-ranked."""
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from local_matching import TentativeMatches, tentative_matches, filter_matches
from local_geometry import Verification, verify_geometry
from local_matching_config import number, resolve_matching, resolve_geometry, resolve_scoring


def grid_coverage(points, shape, grid):
    """Fraction of occupied cells; grid is [rows, columns], coordinates are native."""
    if not len(points):
        return 0.
    rows, columns = grid
    h, w = shape
    cells = np.floor(np.asarray(points) / [w, h] * [columns, rows]).astype(np.int64)
    cells = np.clip(cells, [0, 0], [columns-1, rows-1])
    return len(np.unique(cells, axis=0)) / (rows * columns)


def score_verification(verification, config=None):
    config = resolve_scoring(config)
    count = verification.inlier_count
    query_coverage = database_coverage = 0.
    if count:
        mask, matches = verification.inlier_mask, verification.matches
        query_coverage = grid_coverage(matches.query_points[mask], matches.query_shape, config["coverage_grid"])
        database_coverage = grid_coverage(matches.database_points[mask], matches.database_shape, config["coverage_grid"])
    return dict(inlier_count=int(count), query_coverage=float(query_coverage), database_coverage=float(database_coverage),
                coverage_weighted=float(count * np.sqrt(query_coverage * database_coverage)),
                coverage_grid=config["coverage_grid"])


@dataclass
class CandidateEvidence:
    database_id: int
    filename: str
    tentative: TentativeMatches
    verification: Verification
    scores: dict
    timings: dict


def rank_candidates(candidates, mode="inlier_count", k=None):
    resolve_scoring(dict(mode=mode))
    if k is not None:
        k = number(k, "k", integer=True)
        if k > len(candidates):
            raise ValueError(f"k={k} exceeds the {len(candidates)} inspected database candidates")
    # No rejection: zero-score candidates remain, with an explicit ID tie-break.
    return sorted(candidates, key=lambda c: (-c.scores[mode] if c.verification.success else 0., c.database_id))[:k]


def verify_candidate(database_id, filename, tentative, matching=None, geometry=None, scoring=None):
    start = perf_counter()
    matches = filter_matches(tentative, matching)
    filtered = perf_counter()
    verification = verify_geometry(matches, geometry)
    verified = perf_counter()
    scores = score_verification(verification, scoring)
    return CandidateEvidence(int(database_id), filename, tentative, verification, scores,
                             dict(filtering=filtered-start, geometry=verified-filtered,
                                  scoring=perf_counter()-verified))


def retrieve_local(query, database, *, matching=None, geometry=None, scoring=None, database_limit=None):
    """One query feature set against every cached painting (or an explicit debug prefix).

    Returns all evidence plus timings, never a thresholded identification. Call
    rank_candidates repeatedly for different K/modes; call score_verification for
    different grids, verify_candidate for new filtering/geometry, reusing tentative.
    """
    # Omitted settings preserve the original eager evidence API; explicit ratio runs are lazy.
    prepare_reverse = matching is None
    matching, geometry, scoring = resolve_matching(matching), resolve_geometry(geometry), resolve_scoring(scoring)
    if (len(database.ids) != len(database.features) or len(database.ids) != len(database.filenames)
            or len(set(database.ids)) != len(database.ids) or not database.ids):
        raise ValueError("Database features require nonempty, unique IDs aligned with filenames/features")
    if database_limit is not None:
        database_limit = number(database_limit, "database_limit", integer=True)
    start = perf_counter()
    candidates = []
    for image_id, name, features in zip(database.ids[:database_limit], database.filenames[:database_limit],
                                        database.features[:database_limit]):
        before = perf_counter()
        tentative = tentative_matches(query, features, compute_reverse=prepare_reverse or matching["filter"] == "ratio_mutual")
        elapsed = perf_counter()-before
        candidate = verify_candidate(image_id, name, tentative, matching, geometry, scoring)
        candidate.timings["matching"] = elapsed
        candidates.append(candidate)
    timings = {key: float(sum(c.timings[key] for c in candidates))
               for key in ("matching", "filtering", "geometry", "scoring")}
    timings["total_retrieval"] = perf_counter()-start
    return candidates, timings


def candidate_report(candidate, rank, mode, *, detail=True):
    v, t = candidate.verification, candidate.tentative
    m = v.matches
    summary = dict(database_id=candidate.database_id, filename=candidate.filename, rank=rank,
                method=t.method, norm="L2" if t.norm == cv2.NORM_L2 else "HAMMING", matching=m.config, geometry=v.config,
                tentative_count=t.candidate_count, filtered_count=m.count, verified=v.success,
                inlier_count=v.inlier_count, inlier_ratio=v.inlier_ratio, scores=candidate.scores,
                selected_score=float(candidate.scores[mode]) if v.success else 0.,
                failure_reason=v.failure_reason, mean_inlier_error=v.mean_error, median_inlier_error=v.median_error,
                timings=candidate.timings)
    if not detail:
        return summary
    summary.update(
                query_indices=m.query_indices.tolist(), database_indices=m.database_indices.tolist(),
                distances=m.distances.tolist(), query_points=m.query_points.tolist(), database_points=m.database_points.tolist(),
                query_shape=list(m.query_shape), database_shape=list(m.database_shape),
                inlier_mask=v.inlier_mask.tolist(), reprojection_errors=v.reprojection_errors,
                homography=v.homography.tolist() if v.success else None,
                native_homography=v.native_homography.tolist() if v.success else None,
                query_transform=v.query_transform.tolist(), database_transform=v.database_transform.tolist(),
                homography_direction="query_to_database", error_units="normalized_database_pixels",
                timings=candidate.timings)

    return summary
