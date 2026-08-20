"""Observed and average representative structures for aligned ensembles."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


class RepresentativeStructureError(ValueError):
    """Raised when representative-structure inputs are invalid."""


def _ensemble(coordinates: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] != 3:
        raise RepresentativeStructureError("coordinates must have shape frames by atoms by three")
    if not np.isfinite(values).all():
        raise RepresentativeStructureError("coordinates must be finite")
    return values


def representative_structures(
    coordinates: Sequence[Sequence[Sequence[float]]],
    frame_weights: Sequence[float] | None = None,
    within_rmsd_standard_deviations: float = 1.0,
) -> Dict[str, object]:
    """Return mean, closest-to-mean, medoid, and central observed frames.

    Input frames must already share a validated atom order and alignment basis.
    The arithmetic mean is labeled separately because it need not be an observed
    or chemically valid structure.
    """

    values = _ensemble(coordinates)
    frame_count = values.shape[0]
    if frame_weights is None:
        weights = np.ones(frame_count, dtype=float)
    else:
        weights = np.asarray(frame_weights, dtype=float)
        if weights.shape != (frame_count,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise RepresentativeStructureError("frame_weights must be finite, nonnegative, and frame-resolved")
    if weights.sum() <= 0.0:
        raise RepresentativeStructureError("frame_weights must have positive total weight")
    if within_rmsd_standard_deviations < 0.0:
        raise RepresentativeStructureError("within_rmsd_standard_deviations must be nonnegative")
    mean = np.average(values, axis=0, weights=weights)
    distances_to_mean = np.sqrt(np.mean((values - mean) ** 2, axis=(1, 2)))
    pairwise = np.sqrt(np.mean((values[:, None] - values[None, :]) ** 2, axis=(2, 3)))
    medoid_index = int(np.argmin(np.average(pairwise, axis=1, weights=weights)))
    closest_index = int(np.argmin(distances_to_mean))
    threshold = float(
        np.average(distances_to_mean, weights=weights)
        + within_rmsd_standard_deviations
        * np.sqrt(np.average((distances_to_mean - np.average(distances_to_mean, weights=weights)) ** 2, weights=weights))
    )
    central = np.flatnonzero(distances_to_mean <= threshold).astype(int).tolist()
    return {
        "arithmetic_mean_coordinates": mean.tolist(),
        "closest_to_mean_frame_index": closest_index,
        "medoid_frame_index": medoid_index,
        "rmsd_to_mean": distances_to_mean.tolist(),
        "central_frame_indices": central,
        "central_rmsd_threshold": threshold,
        "frame_count": frame_count,
        "alignment_required": True,
    }
