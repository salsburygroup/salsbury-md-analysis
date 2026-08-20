"""Hydrogen-bond pattern encoding and Jaccard clustering.

This module consolidates frame-by-bond binary-matrix workflows. It
operates on explicit bond identifiers and never infers donor chemistry.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class HydrogenBondPatternError(ValueError):
    """Raised when a hydrogen-bond pattern contract is invalid."""


def encode_bond_patterns(
    frame_bonds: Sequence[Iterable[str]], bond_ids: Sequence[str] | None = None
) -> Tuple[List[str], np.ndarray]:
    """Encode observed bond-ID sets as a deterministic frames by bonds matrix."""

    observed = [set(map(str, bonds)) for bonds in frame_bonds]
    if not observed:
        raise HydrogenBondPatternError("at least one frame is required")
    identifiers = sorted(set().union(*observed)) if bond_ids is None else list(map(str, bond_ids))
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise HydrogenBondPatternError("bond_ids must be nonempty and unique")
    allowed = set(identifiers)
    unknown = sorted(set().union(*observed).difference(allowed))
    if unknown:
        raise HydrogenBondPatternError("observations contain undeclared bond IDs: " + ", ".join(unknown))
    matrix = np.asarray([[bond in bonds for bond in identifiers] for bonds in observed], dtype=bool)
    return identifiers, matrix


def jaccard_distance_matrix(patterns: Sequence[Sequence[bool]]) -> np.ndarray:
    """Return the symmetric Jaccard distance matrix, defining two empty sets as equal."""

    matrix = np.asarray(patterns, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HydrogenBondPatternError("patterns must be a nonempty two-dimensional matrix")
    intersections = matrix.astype(int) @ matrix.astype(int).T
    counts = matrix.sum(axis=1)
    unions = counts[:, None] + counts[None, :] - intersections
    distances = np.zeros_like(intersections, dtype=float)
    np.divide(1.0 * (unions - intersections), unions, out=distances, where=unions > 0)
    np.fill_diagonal(distances, 0.0)
    return distances


def pam_jaccard(
    patterns: Sequence[Sequence[bool]], cluster_count: int, maximum_iterations: int = 100
) -> Dict[str, object]:
    """Deterministic partitioning around medoids using Jaccard distance."""

    distances = jaccard_distance_matrix(patterns)
    n_frames = distances.shape[0]
    if isinstance(cluster_count, bool) or not 1 <= int(cluster_count) <= n_frames:
        raise HydrogenBondPatternError("cluster_count must be between one and the frame count")
    if maximum_iterations < 1:
        raise HydrogenBondPatternError("maximum_iterations must be positive")
    medoids = [int(np.argmin(distances.sum(axis=1)))]
    while len(medoids) < int(cluster_count):
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    for iteration in range(maximum_iterations):
        labels = np.argmin(distances[:, medoids], axis=1)
        updated = []
        for cluster in range(len(medoids)):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                updated.append(medoids[cluster])
                continue
            local = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(local))]))
        if updated == medoids:
            break
        medoids = updated
    labels = np.argmin(distances[:, medoids], axis=1)
    return {
        "method": "PAM-Jaccard",
        "labels": labels.astype(int).tolist(),
        "medoid_frame_indices": medoids,
        "cluster_sizes": [int(np.sum(labels == index)) for index in range(len(medoids))],
        "within_cluster_distance_sum": float(sum(distances[row, medoids[label]] for row, label in enumerate(labels))),
        "iterations": iteration + 1,
    }


def hdbscan_jaccard(
    patterns: Sequence[Sequence[bool]], minimum_cluster_size: int, minimum_samples: int | None = None
) -> Dict[str, object]:
    """Run the reference HDBSCAN package on a precomputed Jaccard matrix."""

    distances = jaccard_distance_matrix(patterns)
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise HydrogenBondPatternError("hdbscan is required for HDBSCAN-Jaccard") from exc
    estimator = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(minimum_cluster_size),
        min_samples=None if minimum_samples is None else int(minimum_samples),
    )
    labels = estimator.fit_predict(distances)
    return {
        "method": "HDBSCAN-Jaccard",
        "labels": labels.astype(int).tolist(),
        "probabilities": estimator.probabilities_.astype(float).tolist(),
        "noise_count": int(np.sum(labels < 0)),
        "cluster_count": len(set(labels.tolist()).difference({-1})),
    }
