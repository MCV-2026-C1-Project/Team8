"""BF neighbour search and reusable correspondence filtering; query -> database."""
from dataclasses import dataclass

import cv2
import numpy as np

from local_features import LocalFeatures
from local_matching_config import resolve_matching


@dataclass
class TentativeMatches:
    method: str
    norm: int
    query_points: np.ndarray
    database_points: np.ndarray
    query_shape: tuple
    database_shape: tuple
    neighbors: np.ndarray       # [query descriptor, neighbor rank], -1 if unavailable
    distances: np.ndarray       # Same shape; infinity only for absent neighbors (internal).
    reverse_nearest: np.ndarray # [database descriptor] -> query descriptor, or -1.

    descriptor_sources: tuple | None = None  # Shared references for lazy reverse matching; never copies.

    def ensure_reverse(self):
        if self.reverse_nearest is None:
            query, database = self.descriptor_sources
            self.reverse_nearest = np.full(len(database), -1, np.int64)
            if len(query) and len(database):
                for match in cv2.BFMatcher(self.norm, crossCheck=False).match(database, query):
                    self.reverse_nearest[match.queryIdx] = match.trainIdx
            self.descriptor_sources = None
        return self.reverse_nearest

    @property
    def candidate_count(self):
        """Query descriptors with at least one forward neighbor, before ratio filtering."""
        return int(np.count_nonzero(self.neighbors[:, 0] >= 0))


@dataclass
class Correspondences:
    method: str
    config: dict
    query_indices: np.ndarray
    database_indices: np.ndarray
    distances: np.ndarray
    query_points: np.ndarray
    database_points: np.ndarray
    query_shape: tuple
    database_shape: tuple
    candidate_count: int

    @property
    def count(self):
        return len(self.query_indices)


def tentative_matches(query, database, *, compute_reverse=True):
    for features in (query, database):
        if not isinstance(features, LocalFeatures):
            raise ValueError("Matching requires LocalFeatures objects")
        # Revalidate mutable arrays without coercing their dtype or copying descriptors.
        LocalFeatures(features.method, features.params, features.image_shape, features.keypoints, features.descriptors)
    if query.method != database.method or query.norm != database.norm:
        raise ValueError("Cannot match incompatible local descriptor methods")
    nq, nd = len(query.keypoints), len(database.keypoints)
    indices = np.full((nq, 2), -1, np.int64)
    distances = np.full((nq, 2), np.inf, np.float64)
    reverse = np.full(nd, -1, np.int64)
    if nq and nd:
        matcher = cv2.BFMatcher(query.norm, crossCheck=False)
        for row in matcher.knnMatch(query.descriptors, database.descriptors, k=min(2, nd)):
            for rank, match in enumerate(row):
                indices[match.queryIdx, rank] = match.trainIdx
                distances[match.queryIdx, rank] = match.distance
        if compute_reverse:
            for match in matcher.match(database.descriptors, query.descriptors):
                reverse[match.queryIdx] = match.trainIdx
    return TentativeMatches(query.method, query.norm, query.points, database.points,
                            query.image_shape, database.image_shape, indices, distances, reverse if compute_reverse else None,
                            None if compute_reverse else (query.descriptors, database.descriptors))


def filter_matches(tentative, config=None):
    config = resolve_matching(config)
    valid = (tentative.neighbors[:, 1] >= 0)
    # Only evaluate complete pairs: no inf/inf ratio and strict rejection of tied 0/0 distances.
    rows = np.flatnonzero(valid)
    rows = rows[tentative.distances[rows, 0] < config["ratio_threshold"] * tentative.distances[rows, 1]]
    columns = tentative.neighbors[rows, 0]
    if config["filter"] == "ratio_mutual":
        mutual = tentative.ensure_reverse()[columns] == rows
        rows, columns = rows[mutual], columns[mutual]
    return Correspondences(tentative.method, config, rows, columns, tentative.distances[rows, 0].copy(),
                           tentative.query_points[rows], tentative.database_points[columns],
                           tentative.query_shape, tentative.database_shape, tentative.candidate_count)
