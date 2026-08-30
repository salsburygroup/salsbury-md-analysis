"""Additional clustering families with explicit algorithm identities."""

from __future__ import annotations

import importlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .clustering import (
    ClusteringAnalysisError,
    _canonicalize,
    _standardize,
    adjusted_rand_index,
    silhouette_score,
    silhouette_score_report,
)
from .feature_matrix import load_feature_matrix, parse_feature_selection
from .frame_sampling import integer_stride_indices, uniform_indices
from .manifests import ManifestValidationError, load_json
from .trajectory_features import TrajectoryFeatureError
from .state_populations import summarize_state_populations
from .upstream_cache import load_cached_project_report
from .validation import positive_integer
from .provenance import stable_json_sha256


Vector = Tuple[float, ...]


class AlternativeClusteringError(ValueError):
    """Raised when an alternative clustering request is invalid."""


def _array(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise AlternativeClusteringError(
            "feature vectors must be a two-dimensional array with at least two rows"
        )
    if not np.isfinite(values).all():
        raise AlternativeClusteringError("feature vectors contain non-finite values")
    return values


def _distance_matrix(values: np.ndarray, p: float = 2.0) -> np.ndarray:
    if not math.isfinite(p) or p < 1.0:
        raise AlternativeClusteringError("Minkowski p must be finite and at least 1")
    delta = np.abs(values[:, None, :] - values[None, :, :])
    return np.sum(delta ** p, axis=2) ** (1.0 / p)


def _canonical_result(
    values: np.ndarray,
    assignments: Sequence[int],
    centers: Sequence[Sequence[float]],
) -> Tuple[List[int], List[Vector], List[int]]:
    labels, ordered = _canonicalize(
        [int(value) for value in assignments],
        [tuple(float(x) for x in center) for center in centers],
    )
    sizes = [labels.count(index) for index in range(len(ordered))]
    if sum(sizes) != values.shape[0]:
        raise AlternativeClusteringError("clustering did not assign every observation")
    return labels, ordered, sizes


def run_pam(
    vectors: Sequence[Sequence[float]],
    k: int,
    maximum_iterations: int = 100,
    p: float = 2.0,
    *,
    _distances: np.ndarray | None = None,
) -> Dict[str, object]:
    """Run deterministic BUILD-initialized partition around medoids."""

    values = _array(vectors)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 < k <= len(values):
        raise AlternativeClusteringError("PAM k must be between 2 and observation count")
    maximum_iterations = positive_integer(maximum_iterations, "maximum_iterations")
    distances = _distance_matrix(values, p) if _distances is None else _distances
    if (
        not isinstance(distances, np.ndarray)
        or distances.shape != (len(values), len(values))
        or not np.isfinite(distances).all()
    ):
        raise AlternativeClusteringError("precomputed PAM distances are invalid")
    first = min(range(len(values)), key=lambda index: (float(distances[index].sum()), index))
    medoids = [first]
    while len(medoids) < k:
        candidates = [index for index in range(len(values)) if index not in medoids]
        medoids.append(max(
            candidates,
            key=lambda index: (min(float(distances[index, medoid]) for medoid in medoids), -index),
        ))

    assignments: List[int] = []
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        assignments = [
            min(range(k), key=lambda label: (float(distances[index, medoids[label]]), label))
            for index in range(len(values))
        ]
        if len(set(assignments)) != k:
            raise AlternativeClusteringError("PAM produced an empty cluster")
        updated = []
        for label in range(k):
            members = np.asarray([
                index for index, value in enumerate(assignments) if value == label
            ], dtype=int)
            costs = distances[np.ix_(members, members)].sum(axis=1)
            updated.append(int(members[int(np.argmin(costs))]))
        if updated == medoids:
            converged = True
            break
        medoids = updated
    centers = values[medoids]
    labels, ordered_centers, sizes = _canonical_result(values, assignments, centers)
    center_to_medoid = {
        tuple(float(x) for x in values[index]): index for index in medoids
    }
    ordered_medoids = [center_to_medoid[center] for center in ordered_centers]
    objective = sum(
        float(distances[index, ordered_medoids[label]])
        for index, label in enumerate(labels)
    )
    return {
        "algorithm": "partition_around_medoids",
        "valid": converged,
        "converged": converged,
        "iteration_count": iteration,
        "k": k,
        "minkowski_p": float(p),
        "assignments": labels,
        "medoid_indices": ordered_medoids,
        "centers": ordered_centers,
        "cluster_sizes": sizes,
        "objective_distance_sum": objective,
    }


def run_mwpam(
    vectors: Sequence[Sequence[float]],
    k: int,
    p: float = 2.0,
    maximum_iterations: int = 100,
    dispersion_floor: float = 1.0e-12,
    *,
    _initial_distances: np.ndarray | None = None,
) -> Dict[str, object]:
    """Run Minkowski-weighted PAM with cluster-specific feature weights."""

    if not math.isfinite(p) or p <= 1.0:
        raise AlternativeClusteringError("MWPAM requires finite p greater than 1")
    if not math.isfinite(dispersion_floor) or dispersion_floor <= 0.0:
        raise AlternativeClusteringError("dispersion_floor must be finite and positive")
    values = _array(vectors)
    base = run_pam(
        values.tolist(), k, maximum_iterations, p,
        _distances=_initial_distances,
    )
    medoids = [int(value) for value in base["medoid_indices"]]
    weights = np.full((k, values.shape[1]), 1.0 / values.shape[1], dtype=float)
    previous: List[int] = []
    assignments: List[int] = []
    converged = False
    iteration = 0
    for iteration in range(1, positive_integer(maximum_iterations, "maximum_iterations") + 1):
        weighted = np.empty((len(values), k), dtype=float)
        for label, medoid in enumerate(medoids):
            weighted[:, label] = np.sum(
                (weights[label] ** p) * np.abs(values - values[medoid]) ** p,
                axis=1,
            )
        assignments = [
            min(range(k), key=lambda label: (float(weighted[index, label]), label))
            for index in range(len(values))
        ]
        if len(set(assignments)) != k:
            raise AlternativeClusteringError("MWPAM produced an empty cluster")
        updated = []
        for label in range(k):
            members = np.asarray([
                index for index, value in enumerate(assignments) if value == label
            ], dtype=int)
            cluster = values[members]
            pairwise = np.abs(
                cluster[:, None, :] - cluster[None, :, :]
            ) ** p
            costs = np.sum(
                pairwise * (weights[label][None, None, :] ** p), axis=(1, 2)
            )
            updated.append(int(members[int(np.argmin(costs))]))
            dispersion = np.sum(
                np.abs(values[members] - values[updated[-1]]) ** p,
                axis=0,
            )
            inverse = np.maximum(dispersion, dispersion_floor) ** (-1.0 / (p - 1.0))
            weights[label] = inverse / inverse.sum()
        if assignments == previous and updated == medoids:
            converged = True
            break
        previous = assignments[:]
        medoids = updated
    centers = values[medoids]
    labels, ordered_centers, sizes = _canonical_result(values, assignments, centers)
    order = [
        next(index for index, center in enumerate(centers) if tuple(center) == target)
        for target in ordered_centers
    ]
    return {
        "algorithm": "minkowski_weighted_partition_around_medoids",
        "valid": converged,
        "converged": converged,
        "iteration_count": iteration,
        "k": k,
        "minkowski_p": float(p),
        "assignments": labels,
        "medoid_indices": [medoids[index] for index in order],
        "centers": ordered_centers,
        "feature_weights": [weights[index].tolist() for index in order],
        "cluster_sizes": sizes,
    }


def run_ward(
    vectors: Sequence[Sequence[float]],
    k: int,
    *,
    _linkage: np.ndarray | None = None,
) -> Dict[str, object]:
    """Run deterministic Euclidean Ward agglomerative clustering."""

    values = _array(vectors)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 < k <= len(values):
        raise AlternativeClusteringError("Ward k must be between 2 and observation count")
    hierarchy = importlib.import_module("scipy.cluster.hierarchy")
    linkage = (
        hierarchy.linkage(values, method="ward", metric="euclidean", optimal_ordering=True)
        if _linkage is None else np.asarray(_linkage, dtype=float)
    )
    if linkage.shape != (len(values) - 1, 4) or not np.isfinite(linkage).all():
        raise AlternativeClusteringError("precomputed Ward linkage is invalid")
    raw = hierarchy.fcluster(linkage, t=k, criterion="maxclust") - 1
    raw_labels = [int(value) for value in raw]
    unique = sorted(set(raw_labels))
    centers = [values[np.asarray(raw_labels) == label].mean(axis=0) for label in unique]
    remap = {label: index for index, label in enumerate(unique)}
    labels, ordered, sizes = _canonical_result(
        values, [remap[label] for label in raw_labels], centers
    )
    return {
        "algorithm": "ward_agglomerative",
        "valid": len(ordered) == k,
        "k": len(ordered),
        "assignments": labels,
        "centers": ordered,
        "cluster_sizes": sizes,
        "linkage": linkage.tolist(),
    }


def run_quality_threshold(
    vectors: Sequence[Sequence[float]],
    cutoff: float,
    *,
    _distances: np.ndarray | None = None,
) -> Dict[str, object]:
    """Run a greedy radius-neighborhood quality-threshold partition."""

    values = _array(vectors)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise AlternativeClusteringError("quality-threshold cutoff must be positive")
    distances = _distance_matrix(values) if _distances is None else _distances
    if (
        not isinstance(distances, np.ndarray)
        or distances.shape != (len(values), len(values))
        or not np.isfinite(distances).all()
    ):
        raise AlternativeClusteringError(
            "precomputed quality-threshold distances are invalid"
        )
    remaining = set(range(len(values)))
    labels = [-1] * len(values)
    centers = []
    label = 0
    while remaining:
        center = max(
            remaining,
            key=lambda index: (
                sum(float(distances[index, other]) < cutoff for other in remaining),
                -index,
            ),
        )
        members = sorted(
            other for other in remaining if float(distances[center, other]) < cutoff
        )
        if len(members) == 1:
            break
        for index in members:
            labels[index] = label
        remaining.difference_update(members)
        centers.append(center)
        label += 1
    return {
        "algorithm": "quality_threshold",
        "valid": True,
        "cutoff": float(cutoff),
        "assignments": labels,
        "center_indices": centers,
        "cluster_sizes": [labels.count(index) for index in range(len(centers))],
        "noise_count": labels.count(-1),
    }


def partitioned_local_depths(
    vectors: Sequence[Sequence[float]], maximum_observations: int = 500
) -> Dict[str, object]:
    """Calculate the PaLD cohesion matrix."""

    values = _array(vectors)
    maximum_observations = positive_integer(maximum_observations, "maximum_observations")
    if len(values) > maximum_observations:
        raise AlternativeClusteringError("PaLD maximum_observations gate exceeded")
    distances = _distance_matrix(values)
    count = len(values)
    accum = np.zeros((count, count), dtype=float)
    for left in range(count - 1):
        for right in range(left + 1, count):
            local = [
                index for index in range(count)
                if distances[index, left] <= distances[right, left]
                or distances[index, right] <= distances[left, right]
            ]
            scale = 1.0 / len(local)
            for index in local:
                if distances[index, left] < distances[index, right]:
                    accum[left, index] += scale
                elif distances[index, left] > distances[index, right]:
                    accum[right, index] += scale
                else:
                    accum[left, index] += 0.5 * scale
                    accum[right, index] += 0.5 * scale
    cohesion = accum / (count - 1)
    return {
        "algorithm": "partitioned_local_depth",
        "observation_count": count,
        "distance_matrix": distances.tolist(),
        "cohesion_matrix": cohesion.tolist(),
    }


def calinski_harabasz_score(
    vectors: Sequence[Sequence[float]], assignments: Sequence[int]
) -> float:
    values = _array(vectors)
    labels = np.asarray(assignments, dtype=int)
    if labels.shape != (len(values),):
        raise AlternativeClusteringError("assignment count does not match observations")
    unique = sorted(set(labels.tolist()) - {-1})
    if not 1 < len(unique) < len(values):
        raise AlternativeClusteringError("Calinski-Harabasz requires 2..n-1 clusters")
    selected = labels >= 0
    overall = values[selected].mean(axis=0)
    between = 0.0
    within = 0.0
    for label in unique:
        cluster = values[labels == label]
        center = cluster.mean(axis=0)
        between += len(cluster) * float(np.sum((center - overall) ** 2))
        within += float(np.sum((cluster - center) ** 2))
    if within <= 0.0:
        return math.inf
    observations = int(selected.sum())
    return (between / (len(unique) - 1)) / (within / (observations - len(unique)))


def davies_bouldin_score(
    vectors: Sequence[Sequence[float]], assignments: Sequence[int]
) -> float:
    values = _array(vectors)
    labels = np.asarray(assignments, dtype=int)
    unique = sorted(set(labels.tolist()) - {-1})
    if len(unique) < 2:
        raise AlternativeClusteringError("Davies-Bouldin requires at least two clusters")
    centers = []
    scatters = []
    for label in unique:
        cluster = values[labels == label]
        if not len(cluster):
            raise AlternativeClusteringError("empty cluster in Davies-Bouldin input")
        center = cluster.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.linalg.norm(cluster - center, axis=1).mean()))
    ratios = []
    for left in range(len(unique)):
        candidates = []
        for right in range(len(unique)):
            if left == right:
                continue
            separation = float(np.linalg.norm(centers[left] - centers[right]))
            if separation <= 0.0:
                return math.inf
            candidates.append((scatters[left] + scatters[right]) / separation)
        ratios.append(max(candidates))
    return sum(ratios) / len(ratios)


def _balanced_uniform_sample(
    metadata: Sequence[Mapping[str, object]],
    maximum_observations_per_replica: int,
) -> Tuple[List[int], Dict[str, object]]:
    """Select one regular stride across every replica-member segment."""

    budget = positive_integer(
        maximum_observations_per_replica,
        "fit_sampling.maximum_observations_per_replica",
        error_type=AlternativeClusteringError,
    )
    grouped: Dict[Tuple[str, str], Dict[Tuple[str | None, str], List[int]]] = {}
    for index, record in enumerate(metadata):
        key = (str(record["system_id"]), str(record["replica_id"]))
        member_id = str(record["member_id"]) if "member_id" in record else None
        segment_id = str(record["segment_id"])
        grouped.setdefault(key, {}).setdefault(
            (member_id, segment_id), []
        ).append(index)
    if not grouped:
        raise AlternativeClusteringError("feature matrix contains no replica identities")
    for trajectory_groups in grouped.values():
        for indices in trajectory_groups.values():
            indices.sort(key=lambda index: int(metadata[index]["source_frame_index"]))
    if any(budget < len(trajectory_groups) for trajectory_groups in grouped.values()):
        raise AlternativeClusteringError(
            "fit budget must retain at least one observation per replica-member segment"
        )
    stride = max(
        1,
        max(
            math.ceil(
                sum(len(indices) for indices in trajectory_groups.values()) / budget
            )
            for trajectory_groups in grouped.values()
        ),
    )
    while any(
        sum(math.ceil(len(indices) / stride) for indices in trajectory_groups.values())
        > budget
        for trajectory_groups in grouped.values()
    ):
        stride += 1
    selected: List[int] = []
    replicas = []
    trajectory_reports = []
    for (system_id, replica_id), trajectory_groups in grouped.items():
        source_count = sum(len(indices) for indices in trajectory_groups.values())
        chosen = []
        member_totals: Dict[str | None, List[int]] = {}
        for (member_id, segment_id), indices in sorted(
            trajectory_groups.items(),
            key=lambda item: (
                "" if item[0][0] is None else item[0][0], item[0][1]
            ),
        ):
            segment_chosen = [
                indices[position]
                for position in sorted(integer_stride_indices(len(indices), stride))
            ]
            chosen.extend(segment_chosen)
            totals = member_totals.setdefault(member_id, [0, 0])
            totals[0] += len(indices)
            totals[1] += len(segment_chosen)
            trajectory_reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "member_id": member_id,
                "segment_id": segment_id,
                "source_observation_count": len(indices),
                "selected_observation_count": len(segment_chosen),
                "selected_source_matrix_indices": segment_chosen,
            })
        member_reports = [
            {
                "member_id": member_id,
                "source_observation_count": counts[0],
                "selected_observation_count": counts[1],
            }
            for member_id, counts in sorted(
                member_totals.items(), key=lambda item: "" if item[0] is None else item[0]
            )
        ]
        chosen.sort()
        selected.extend(chosen)
        replicas.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "source_observation_count": source_count,
            "selected_observation_count": len(chosen),
            "selected_source_matrix_indices": chosen,
            "first_selected_source_matrix_index": chosen[0],
            "last_selected_source_matrix_index": chosen[-1],
            "members": member_reports,
        })
    selected.sort()
    return selected, {
        "mode": "common_regular_stride_per_replica_member_segment_v1",
        "maximum_observations_per_replica": budget,
        "source_frame_stride": stride,
        "source_observation_count": len(metadata),
        "selected_observation_count": len(selected),
        "selected_fraction": len(selected) / len(metadata),
        "selection_order": (
            "one common integer stride within each replica-member segment; each "
            "physical segment boundary remains explicit for sampled-state kinetics"
        ),
        "selected_source_matrix_indices": selected,
        "replicas": replicas,
        "trajectory_groups": trajectory_reports,
    }


def _integer_stride_sample(
    metadata: Sequence[Mapping[str, object]],
    stride: int,
) -> Tuple[List[int], Dict[str, object]]:
    """Select one exact stride on every concatenated replica-member timeline."""

    resolved_stride = positive_integer(
        stride,
        "fit_sampling.integer_stride",
        error_type=AlternativeClusteringError,
    )
    grouped: Dict[
        Tuple[str, str], Dict[str | None, List[int]]
    ] = {}
    for index, record in enumerate(metadata):
        replica_key = (str(record["system_id"]), str(record["replica_id"]))
        member_id = str(record["member_id"]) if "member_id" in record else None
        grouped.setdefault(replica_key, {}).setdefault(member_id, []).append(index)
    if not grouped:
        raise AlternativeClusteringError("feature matrix contains no replica identities")

    selected: List[int] = []
    replicas = []
    segment_reports = []
    for (system_id, replica_id), member_timelines in grouped.items():
        replica_chosen: List[int] = []
        member_reports = []
        for member_id, timeline in sorted(
            member_timelines.items(),
            key=lambda item: "" if item[0] is None else item[0],
        ):
            # Feature-matrix order is the authoritative concatenated trajectory
            # order.  Do not restart the stride at a segment boundary.
            chosen = [
                timeline[position]
                for position in sorted(
                    integer_stride_indices(len(timeline), resolved_stride)
                )
            ]
            replica_chosen.extend(chosen)
            by_segment: Dict[str, List[int]] = {}
            selected_by_segment: Dict[str, List[int]] = {}
            chosen_set = set(chosen)
            for matrix_index in timeline:
                segment_id = str(metadata[matrix_index]["segment_id"])
                by_segment.setdefault(segment_id, []).append(matrix_index)
                if matrix_index in chosen_set:
                    selected_by_segment.setdefault(segment_id, []).append(matrix_index)
            for segment_id, source_indices in by_segment.items():
                segment_chosen = selected_by_segment.get(segment_id, [])
                segment_reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "member_id": member_id,
                    "segment_id": segment_id,
                    "source_observation_count": len(source_indices),
                    "selected_observation_count": len(segment_chosen),
                    "selected_source_matrix_indices": segment_chosen,
                })
            member_reports.append({
                "member_id": member_id,
                "source_observation_count": len(timeline),
                "selected_observation_count": len(chosen),
                "selected_source_matrix_indices": chosen,
            })
        replica_chosen.sort()
        selected.extend(replica_chosen)
        replicas.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "source_observation_count": sum(
                len(timeline) for timeline in member_timelines.values()
            ),
            "selected_observation_count": len(replica_chosen),
            "selected_source_matrix_indices": replica_chosen,
            "first_selected_source_matrix_index": replica_chosen[0],
            "last_selected_source_matrix_index": replica_chosen[-1],
            "members": member_reports,
        })
    selected.sort()
    return selected, {
        "mode": "integer_stride_per_replica_member_timeline_v1",
        "source_frame_stride": resolved_stride,
        "source_observation_count": len(metadata),
        "selected_observation_count": len(selected),
        "selected_fraction": len(selected) / len(metadata),
        "selection_order": (
            "one exact integer stride over each concatenated replica-member "
            "timeline; frame zero retained; no random draw; segment boundaries "
            "do not restart the stride"
        ),
        "selected_source_matrix_indices": selected,
        "replicas": replicas,
        "trajectory_groups": segment_reports,
    }


def _nearest_center_assignments(
    values: np.ndarray,
    centers: Sequence[Sequence[float]],
    *,
    p: float = 2.0,
    feature_weights: Sequence[Sequence[float]] | None = None,
) -> List[int]:
    """Assign observations to fitted centers with deterministic ties."""

    center_array = np.asarray(centers, dtype=float)
    if center_array.ndim != 2 or center_array.shape[1] != values.shape[1]:
        raise AlternativeClusteringError("fitted centers do not match feature dimensions")
    delta = np.abs(values[:, None, :] - center_array[None, :, :]) ** p
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=float)
        if weights.shape != center_array.shape:
            raise AlternativeClusteringError("MWPAM feature weights do not match centers")
        distances = np.sum(delta * (weights[None, :, :] ** p), axis=2)
    else:
        distances = np.sum(delta, axis=2)
    return [int(value) for value in np.argmin(distances, axis=1)]


def _extend_partition(
    algorithm: str,
    result: Mapping[str, object],
    full_values: np.ndarray,
    parameters: Mapping[str, object],
) -> Tuple[List[int] | None, Dict[str, object]]:
    """Apply a fitted sampled partition to the complete feature matrix."""

    predicted = result.get("_full_assignments")
    if isinstance(predicted, list):
        return [int(value) for value in predicted], {
            "method": "fitted_model_predict_v1",
            "status": "exact_for_fitted_sample_model",
            "scope": "all source observations",
        }
    centers = result.get("centers")
    if algorithm == "quality_threshold":
        if not isinstance(centers, list):
            return None, {"method": None, "status": "unavailable"}
        labels = _nearest_center_assignments(full_values, centers)
        center_array = np.asarray(centers, dtype=float)
        distances = np.linalg.norm(
            full_values - center_array[np.asarray(labels, dtype=int)], axis=1
        )
        cutoff = float(parameters["quality_threshold_cutoff"])
        labels = [label if distance < cutoff else -1 for label, distance in zip(labels, distances)]
        return labels, {
            "method": "nearest_sampled_qt_center_with_declared_cutoff_v1",
            "status": "deterministic_approximation_to_full_refit",
            "scope": "all source observations",
        }
    if not isinstance(centers, list):
        return None, {"method": None, "status": "unavailable"}
    if algorithm == "pam":
        labels = _nearest_center_assignments(
            full_values, centers, p=float(parameters["minkowski_p"])
        )
        status = "exact_assignment_to_fitted_sample_medoids"
        method = "nearest_sampled_medoid_v1"
    elif algorithm == "mwpam":
        weights = result.get("feature_weights")
        if not isinstance(weights, list):
            raise AlternativeClusteringError("MWPAM result has no feature weights")
        labels = _nearest_center_assignments(
            full_values,
            centers,
            p=float(parameters["minkowski_p"]),
            feature_weights=weights,
        )
        status = "exact_assignment_to_fitted_sample_weighted_medoids"
        method = "nearest_sampled_weighted_medoid_v1"
    else:
        labels = _nearest_center_assignments(full_values, centers)
        status = "deterministic_approximation_to_full_refit"
        method = "nearest_sampled_cluster_center_v1"
    return labels, {
        "method": method,
        "status": status,
        "scope": "all source observations",
    }


def _sklearn_partition(
    algorithm: str,
    values: np.ndarray,
    k: int,
    seed: int,
    settings: Mapping[str, object],
    assignment_values: np.ndarray | None = None,
) -> Dict[str, object]:
    try:
        cluster = importlib.import_module("sklearn.cluster")
        mixture = importlib.import_module("sklearn.mixture")
    except ImportError as exc:
        raise AlternativeClusteringError(
            f"{algorithm} requires the optional scikit-learn dependency"
        ) from exc
    if algorithm == "gaussian_mixture":
        model = mixture.GaussianMixture(
            n_components=k, covariance_type="tied", n_init=5, random_state=seed
        )
        raw = model.fit_predict(values)
        centers = model.means_
        diagnostics = {"aic": float(model.aic(values)), "bic": float(model.bic(values))}
    elif algorithm == "variational_gaussian_mixture":
        model = mixture.BayesianGaussianMixture(
            n_components=k, covariance_type="tied", n_init=3, random_state=seed
        )
        raw = model.fit_predict(values)
        centers = model.means_[sorted(set(raw.tolist()))]
        lower_bound = np.asarray(model.lower_bound_, dtype=float).reshape(-1)
        if lower_bound.size != 1:
            raise AlternativeClusteringError(
                "variational Gaussian mixture returned a nonscalar lower bound"
            )
        diagnostics = {"lower_bound": float(lower_bound[0])}
    elif algorithm == "affinity_propagation":
        model = cluster.AffinityPropagation(
            damping=float(settings.get("affinity_damping", 0.75)), random_state=seed
        )
        raw = model.fit_predict(values)
        centers = model.cluster_centers_
        diagnostics = {"damping": float(model.damping)}
    elif algorithm == "mean_shift":
        bandwidth = settings.get("mean_shift_bandwidth")
        model = cluster.MeanShift(
            bandwidth=None if bandwidth is None else float(bandwidth), cluster_all=True
        )
        raw = model.fit_predict(values)
        centers = model.cluster_centers_
        diagnostics = {"bandwidth": None if bandwidth is None else float(bandwidth)}
    else:  # pragma: no cover - guarded by settings
        raise AlternativeClusteringError(f"unknown scikit-learn algorithm: {algorithm}")
    raw_labels = [int(value) for value in raw]
    unique = sorted(set(raw_labels))
    remap = {label: index for index, label in enumerate(unique)}
    center_vectors = [tuple(float(x) for x in centers[index]) for index in range(len(unique))]
    order = sorted(range(len(center_vectors)), key=lambda index: center_vectors[index])
    canonical = {old: new for new, old in enumerate(order)}
    labels = [canonical[remap[label]] for label in raw_labels]
    ordered = [center_vectors[index] for index in order]
    sizes = [labels.count(index) for index in range(len(ordered))]
    result = {
        "algorithm": algorithm,
        "valid": len(ordered) >= 1,
        "k": len(ordered),
        "assignments": labels,
        "centers": ordered,
        "cluster_sizes": sizes,
        **diagnostics,
    }
    if assignment_values is not None:
        predicted = [int(value) for value in model.predict(assignment_values)]
        if any(label not in remap for label in predicted):
            raise AlternativeClusteringError(
                f"{algorithm} predicted a cluster absent from its fitted partition"
            )
        result["_full_assignments"] = [
            canonical[remap[label]] for label in predicted
        ]
    return result


_ALGORITHMS = {
    "pam", "mwpam", "ward", "gaussian_mixture",
    "variational_gaussian_mixture", "affinity_propagation", "mean_shift",
    "quality_threshold",
}


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("alternative_clustering") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise AlternativeClusteringError(
            "definitions.alternative_clustering must be an object"
        )
    required = {
        "feature_source", "standardize_features",
        "algorithms", "k", "random_seed", "maximum_iterations",
        "minkowski_p", "quality_threshold_cutoff", "maximum_observations",
    }
    missing = sorted(required.difference(raw))
    grid_fields = {
        "k_values", "random_seeds", "minkowski_p_values",
        "quality_threshold_cutoffs", "affinity_damping_values",
        "mean_shift_bandwidth_values", "maximum_exact_silhouette_observations",
    }
    unknown = sorted(set(raw).difference(
        required | {
            "component_indices", "trajectory_feature_columns",
            "affinity_damping", "mean_shift_bandwidth",
            "fit_sampling",
        } | grid_fields
    ))
    if missing or unknown:
        raise AlternativeClusteringError(
            "alternative clustering settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    feature_selection = parse_feature_selection(raw, AlternativeClusteringError)
    algorithms = raw["algorithms"]
    if not isinstance(algorithms, list) or not algorithms or len(set(algorithms)) != len(algorithms):
        raise AlternativeClusteringError("algorithms must be a nonempty unique array")
    if any(value not in _ALGORITHMS for value in algorithms):
        raise AlternativeClusteringError("alternative clustering algorithm is unsupported")
    normalized = {**raw, **feature_selection}

    def numeric_grid(
        name: str, fallback: object, *, integer: bool = False,
        minimum: float = 0.0, strict: bool = True, allow_none: bool = False,
    ) -> List[object]:
        values = raw.get(name, [fallback])
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise AlternativeClusteringError(f"{name} must be a nonempty unique array")
        output: List[object] = []
        for value in values:
            if value is None and allow_none:
                output.append(None)
                continue
            valid_type = (
                isinstance(value, int) and not isinstance(value, bool)
                if integer
                else isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            if not valid_type or not math.isfinite(float(value)):
                raise AlternativeClusteringError(f"{name} contains an invalid value")
            if (strict and float(value) <= minimum) or (
                not strict and float(value) < minimum
            ):
                raise AlternativeClusteringError(f"{name} contains an out-of-range value")
            output.append(int(value) if integer else float(value))
        return sorted(output, key=lambda value: (-1.0 if value is None else float(value)))

    normalized["k_values"] = numeric_grid("k_values", raw["k"], integer=True, minimum=1)
    normalized["random_seeds"] = numeric_grid(
        "random_seeds", raw["random_seed"], integer=True, minimum=0, strict=False
    )
    normalized["minkowski_p_values"] = numeric_grid(
        "minkowski_p_values", raw["minkowski_p"], minimum=1
    )
    normalized["quality_threshold_cutoffs"] = numeric_grid(
        "quality_threshold_cutoffs", raw["quality_threshold_cutoff"], minimum=0
    )
    normalized["affinity_damping_values"] = numeric_grid(
        "affinity_damping_values", raw.get("affinity_damping", 0.75), minimum=0
    )
    if any(float(value) >= 1.0 for value in normalized["affinity_damping_values"]):
        raise AlternativeClusteringError("affinity_damping_values must be less than 1")
    normalized["mean_shift_bandwidth_values"] = numeric_grid(
        "mean_shift_bandwidth_values", raw.get("mean_shift_bandwidth"),
        minimum=0, allow_none=True,
    )
    normalized["maximum_exact_silhouette_observations"] = positive_integer(
        raw.get("maximum_exact_silhouette_observations", raw["maximum_observations"]),
        "maximum_exact_silhouette_observations",
    )
    sampling = raw.get("fit_sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise AlternativeClusteringError("fit_sampling must be an object")
        mode = sampling.get("mode")
        if mode == "algorithm_specific_integer_stride_v1":
            required_sampling = {
                "mode", "target_wall_hours", "member_observation_multiplier",
                "source_physical_frames_per_replica", "full_observation_count",
                "algorithm_plans", "scientific_boundary",
            }
            if set(sampling) != required_sampling:
                raise AlternativeClusteringError(
                    "algorithm-specific fit_sampling fields are incomplete or unknown"
                )
            plans = sampling.get("algorithm_plans")
            if not isinstance(plans, dict) or set(plans) != set(algorithms):
                raise AlternativeClusteringError(
                    "fit_sampling.algorithm_plans must match the requested algorithms"
                )
            normalized_plans: Dict[str, object] = {}
            for algorithm, raw_plan in plans.items():
                if not isinstance(raw_plan, dict):
                    raise AlternativeClusteringError(
                        f"fit plan for {algorithm} must be an object"
                    )
                execution = raw_plan.get("execution")
                if execution == "skip":
                    required_plan = {
                        "execution", "skip_reason", "full_observation_count",
                        "fit_observation_ceiling", "complexity_class",
                        "time_exponent", "calibration_status",
                    }
                    if set(raw_plan) != required_plan:
                        raise AlternativeClusteringError(
                            f"skip fit plan for {algorithm} has invalid fields"
                        )
                    normalized_plans[str(algorithm)] = dict(raw_plan)
                    continue
                required_plan = {
                    "execution", "mode", "strides", "primary_stride",
                    "fit_observation_ceiling",
                    "selected_fit_observations_per_physical_replica",
                    "selected_fit_observation_count", "full_observation_count",
                    "complexity_class", "time_exponent", "calibration_status",
                }
                if execution != "run" or set(raw_plan) != required_plan:
                    raise AlternativeClusteringError(
                        f"run fit plan for {algorithm} has invalid fields"
                    )
                strides = raw_plan.get("strides")
                primary = raw_plan.get("primary_stride")
                if (
                    not isinstance(strides, list)
                    or not strides
                    or len(set(strides)) != len(strides)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in strides
                    )
                    or isinstance(primary, bool)
                    or not isinstance(primary, int)
                    or primary not in strides
                ):
                    raise AlternativeClusteringError(
                        f"fit plan for {algorithm} has invalid integer strides"
                    )
                normalized_plans[str(algorithm)] = {
                    **dict(raw_plan),
                    "strides": sorted(strides, reverse=True),
                }
            normalized["fit_sampling"] = {
                **dict(sampling),
                "algorithm_plans": normalized_plans,
            }
        elif mode == "integer_stride_per_replica_member_v1":
            if set(sampling) != {"mode", "strides", "primary_stride"}:
                raise AlternativeClusteringError(
                    "integer-stride fit_sampling requires mode, strides, and "
                    "primary_stride"
                )
            strides = sampling.get("strides")
            if (
                not isinstance(strides, list)
                or not strides
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in strides
                )
                or len(set(strides)) != len(strides)
            ):
                raise AlternativeClusteringError(
                    "fit_sampling.strides must contain unique positive integers"
                )
            primary = sampling.get("primary_stride")
            if (
                isinstance(primary, bool)
                or not isinstance(primary, int)
                or primary not in strides
            ):
                raise AlternativeClusteringError(
                    "fit_sampling.primary_stride must occur in strides"
                )
            normalized["fit_sampling"] = {
                "mode": mode,
                "strides": sorted(strides, reverse=True),
                "primary_stride": primary,
            }
        elif mode == "balanced_uniform_per_replica_sensitivity_v1":
            # Read-only compatibility for frozen historical manifests. New
            # generic plans never emit this budget-derived selector.
            if set(sampling) != {
                "mode", "budgets_per_replica", "primary_budget_per_replica",
            }:
                raise AlternativeClusteringError(
                    "legacy fit_sampling requires mode, budgets_per_replica, "
                    "and primary_budget_per_replica"
                )
            budgets = sampling.get("budgets_per_replica")
            if (
                not isinstance(budgets, list)
                or not budgets
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in budgets
                )
                or len(set(budgets)) != len(budgets)
            ):
                raise AlternativeClusteringError(
                    "fit_sampling.budgets_per_replica must contain unique positive integers"
                )
            primary = sampling.get("primary_budget_per_replica")
            if (
                isinstance(primary, bool)
                or not isinstance(primary, int)
                or primary not in budgets
            ):
                raise AlternativeClusteringError(
                    "fit_sampling.primary_budget_per_replica must occur in "
                    "budgets_per_replica"
                )
            normalized["fit_sampling"] = {
                "mode": mode,
                "budgets_per_replica": sorted(budgets),
                "primary_budget_per_replica": primary,
            }
        else:
            raise AlternativeClusteringError(
                "fit_sampling.mode must be integer_stride_per_replica_member_v1"
            )
    return normalized


def _algorithm_parameter_rows(
    algorithm: str, settings: Mapping[str, object]
) -> List[Dict[str, object]]:
    if algorithm in {"pam", "mwpam"}:
        return [
            {"k": k, "minkowski_p": p}
            for k, p in itertools.product(
                settings["k_values"], settings["minkowski_p_values"]
            )
        ]
    if algorithm in {
        "ward", "gaussian_mixture", "variational_gaussian_mixture"
    }:
        seeds = (
            settings["random_seeds"]
            if "mixture" in algorithm else [settings["random_seeds"][0]]
        )
        return [
            {"k": k, "random_seed": seed}
            for k, seed in itertools.product(settings["k_values"], seeds)
        ]
    if algorithm == "affinity_propagation":
        return [
            {"random_seed": seed, "affinity_damping": damping}
            for seed, damping in itertools.product(
                settings["random_seeds"], settings["affinity_damping_values"]
            )
        ]
    if algorithm == "mean_shift":
        return [
            {"mean_shift_bandwidth": bandwidth}
            for bandwidth in settings["mean_shift_bandwidth_values"]
        ]
    if algorithm == "quality_threshold":
        return [
            {"quality_threshold_cutoff": cutoff}
            for cutoff in settings["quality_threshold_cutoffs"]
        ]
    return [{}]


def _run_sampled_sweep(
    settings: Mapping[str, object],
    metadata: Sequence[Mapping[str, object]],
    transformed: Sequence[Vector],
    fit_indices: Sequence[int],
    sampling_report: Mapping[str, object],
    *,
    emit_full_records: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    maximum = positive_integer(
        settings["maximum_observations"],
        "maximum_observations",
        error_type=AlternativeClusteringError,
    )
    if len(fit_indices) > maximum:
        raise AlternativeClusteringError(
            "sampled fit exceeds the maximum_observations algorithm gate"
        )
    full_array = np.asarray(transformed, dtype=float)
    fit_values = [transformed[index] for index in fit_indices]
    maximum_iterations = positive_integer(
        settings["maximum_iterations"],
        "maximum_iterations",
        error_type=AlternativeClusteringError,
    )
    selected_results = []
    parameter_sweeps = []
    distance_cache: Dict[float, np.ndarray] = {}
    ward_linkage: np.ndarray | None = None
    for algorithm in settings["algorithms"]:
        if (
            algorithm in {"ward", "quality_threshold"}
            and len(fit_indices) != len(transformed)
        ):
            parameter_sweeps.append({
                "algorithm": algorithm,
                "execution_status": "skipped",
                "skip_reason": (
                    "requires an exact fit over every observation; the planner "
                    "selected a strict subset"
                ),
                "source_observation_count": len(transformed),
                "fit_observation_count": len(fit_indices),
                "run_count": 0,
                "selected_parameters": None,
                "runs": [],
            })
            continue
        parameter_rows = _algorithm_parameter_rows(str(algorithm), settings)
        runs = []
        for parameters in parameter_rows:
            if algorithm == "pam":
                p_value = float(parameters["minkowski_p"])
                if p_value not in distance_cache:
                    distance_cache[p_value] = _distance_matrix(
                        np.asarray(fit_values), p_value
                    )
                distances = distance_cache[p_value]
                result = run_pam(
                    fit_values, int(parameters["k"]), maximum_iterations,
                    p_value, _distances=distances,
                )
            elif algorithm == "mwpam":
                p_value = float(parameters["minkowski_p"])
                if p_value not in distance_cache:
                    distance_cache[p_value] = _distance_matrix(
                        np.asarray(fit_values), p_value
                    )
                distances = distance_cache[p_value]
                result = run_mwpam(
                    fit_values, int(parameters["k"]),
                    p_value, maximum_iterations,
                    _initial_distances=distances,
                )
            elif algorithm == "ward":
                if ward_linkage is None:
                    hierarchy = importlib.import_module("scipy.cluster.hierarchy")
                    ward_linkage = hierarchy.linkage(
                        np.asarray(fit_values), method="ward", metric="euclidean",
                        optimal_ordering=True,
                    )
                result = run_ward(
                    fit_values, int(parameters["k"]), _linkage=ward_linkage
                )
            elif algorithm == "quality_threshold":
                if 2.0 not in distance_cache:
                    distance_cache[2.0] = _distance_matrix(
                        np.asarray(fit_values), 2.0
                    )
                distances = distance_cache[2.0]
                result = run_quality_threshold(
                    fit_values, float(parameters["quality_threshold_cutoff"]),
                    _distances=distances,
                )
            else:
                run_settings = {**settings, **parameters}
                result = _sklearn_partition(
                    algorithm,
                    np.asarray(fit_values),
                    int(parameters.get("k", settings["k_values"][0])),
                    int(parameters.get("random_seed", settings["random_seeds"][0])),
                    run_settings,
                    assignment_values=full_array,
                )
            result["parameters"] = parameters
            assignments = result.get("assignments")
            if isinstance(assignments, list) and len(set(assignments) - {-1}) >= 2:
                retained = [index for index, value in enumerate(assignments) if value >= 0]
                retained_vectors = [fit_values[index] for index in retained]
                retained_labels = [assignments[index] for index in retained]
                silhouette_evaluation = silhouette_score_report(
                    retained_vectors,
                    retained_labels,
                    int(settings["maximum_exact_silhouette_observations"]),
                    int(parameters.get("random_seed", settings["random_seeds"][0])),
                )
                result["silhouette"] = silhouette_evaluation["score"]
                result["silhouette_evaluation"] = silhouette_evaluation
                result["retained_fraction"] = len(retained) / len(assignments)
                result["calinski_harabasz"] = calinski_harabasz_score(
                    retained_vectors, retained_labels
                )
                result["davies_bouldin"] = davies_bouldin_score(
                    retained_vectors, retained_labels
                )
            runs.append(result)

        eligible = [
            result for result in runs
            if result.get("valid") is True and isinstance(result.get("silhouette"), float)
            and (
                algorithm != "quality_threshold"
                or float(result.get("retained_fraction", 0.0)) >= 1.0
            )
        ]
        selected = None
        if eligible:
            selected = sorted(
                eligible,
                key=lambda result: (
                    -float(result["silhouette"]),
                    -float(result.get("retained_fraction", 1.0)),
                    json.dumps(result["parameters"], sort_keys=True),
                ),
            )[0]
            selected["requested_algorithm"] = algorithm
            fit_labels = selected.get("assignments")
            if not isinstance(fit_labels, list) or len(fit_labels) != len(fit_indices):
                raise AlternativeClusteringError(
                    f"{algorithm} selected fit lacks complete sampled label identities"
                )
            selected["fit_assignment_observation_count"] = len(fit_labels)
            selected["fit_assignment_retained_count"] = sum(
                isinstance(label, int) and not isinstance(label, bool) and label >= 0
                for label in fit_labels
            )
            if emit_full_records:
                selected["fit_frame_assignments"] = [
                    {
                        **metadata[source_index],
                        "fit_sample_index": fit_index,
                        "cluster_id": int(label) + 1 if int(label) >= 0 else None,
                        "is_noise": int(label) < 0,
                    }
                    for fit_index, (source_index, label) in enumerate(
                        zip(fit_indices, fit_labels)
                    )
                ]
            centers = selected.get("centers")
            if not isinstance(centers, list):
                center_indices = selected.get("center_indices")
                if isinstance(center_indices, list):
                    centers = [fit_values[int(index)] for index in center_indices]
                    selected["centers"] = centers
            full_labels, extension = _extend_partition(
                str(algorithm), selected, full_array, selected["parameters"]
            )
            selected["fit_sampling"] = dict(sampling_report)
            selected["fit_observation_count"] = len(fit_values)
            selected["assignment_extension"] = extension
            if full_labels is not None:
                selected["_full_assignments"] = full_labels
                selected["full_observation_count"] = len(full_labels)
                retained_full = [
                    index for index, label in enumerate(full_labels) if label >= 0
                ]
                selected["full_retained_fraction"] = (
                    len(retained_full) / len(full_labels)
                )
                selected["full_cluster_sizes"] = [
                    full_labels.count(label)
                    for label in sorted(set(full_labels) - {-1})
                ]
                if emit_full_records and len(set(full_labels) - {-1}) >= 2:
                    selected["full_partition_silhouette_evaluation"] = (
                        silhouette_score_report(
                            [transformed[index] for index in retained_full],
                            [full_labels[index] for index in retained_full],
                            int(settings["maximum_exact_silhouette_observations"]),
                            int(selected["parameters"].get(
                                "random_seed", settings["random_seeds"][0]
                            )),
                        )
                    )
                if emit_full_records:
                    frame_assignments = []
                    for record, vector, label in zip(metadata, transformed, full_labels):
                        distance = None
                        if (
                            label >= 0
                            and isinstance(centers, list)
                            and label < len(centers)
                        ):
                            distance = float(np.sum(
                                (np.asarray(vector) - np.asarray(centers[label])) ** 2
                            ))
                        frame_assignments.append({
                            **record,
                            "cluster_id": int(label) + 1 if label >= 0 else None,
                            "is_noise": label < 0,
                            "squared_distance_in_clustering_space": distance,
                        })
                    selected["frame_assignments"] = frame_assignments
                    selected["state_population_comparison"] = summarize_state_populations(
                        frame_assignments, "cluster_id"
                    )
            selected_results.append(selected)
        for run in runs:
            if run is not selected:
                run.pop("_full_assignments", None)
        parameter_sweeps.append({
            "algorithm": algorithm,
            "execution_status": "complete" if selected else "skipped",
            "skip_reason": (
                None if selected else (
                    "no quality-threshold parameter assigned every observation"
                    if algorithm == "quality_threshold" else
                    "no valid parameter candidate passed the algorithm gates"
                )
            ),
            "run_count": len(runs),
            "selection_rule": (
                "maximum exact or seeded-estimate silhouette, then retained fraction, "
                "then lexicographically smallest declared parameter record"
            ),
            "selected_parameters": selected.get("parameters") if selected else None,
            "runs": runs,
        })
    return selected_results, parameter_sweeps


def _selected_summary(result: Mapping[str, object]) -> Dict[str, object]:
    return {
        "algorithm": result.get("requested_algorithm"),
        "selected_parameters": result.get("parameters"),
        "fit_observation_count": result.get("fit_observation_count"),
        "fit_silhouette": result.get("silhouette"),
        "fit_silhouette_evaluation": result.get("silhouette_evaluation"),
        "fit_retained_fraction": result.get("retained_fraction"),
        "full_observation_count": result.get("full_observation_count"),
        "full_cluster_sizes": result.get("full_cluster_sizes"),
        "full_retained_fraction": result.get("full_retained_fraction"),
        "assignment_extension": result.get("assignment_extension"),
    }


def alternative_clustering_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "alternative_clustering",
        source,
        hash_content=hash_content,
        error_type=AlternativeClusteringError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    settings = _settings(project)
    feature_report, metadata, vectors, feature_contract = load_feature_matrix(
        source,
        settings,
        hash_content=hash_content,
        error_type=AlternativeClusteringError,
    )
    transformed, means, scales = _standardize(
        vectors, bool(settings["standardize_features"])
    )
    sampling = settings.get("fit_sampling")
    budget_results = []
    sensitivity = []
    if (
        isinstance(sampling, dict)
        and sampling.get("mode") == "algorithm_specific_integer_stride_v1"
    ):
        primary_results = []
        primary_sweeps = []
        primary_sampling_by_algorithm: Dict[str, object] = {}
        for algorithm in settings["algorithms"]:
            plan = sampling["algorithm_plans"][algorithm]
            if plan["execution"] == "skip":
                primary_sampling_by_algorithm[str(algorithm)] = dict(plan)
                primary_sweeps.append({
                    "algorithm": algorithm,
                    "execution_status": "skipped",
                    "skip_reason": plan["skip_reason"],
                    "source_observation_count": len(vectors),
                    "fit_observation_count": 0,
                    "run_count": 0,
                    "selected_parameters": None,
                    "runs": [],
                })
                continue
            algorithm_budget_results = []
            primary_stride = int(plan["primary_stride"])
            for stride in plan["strides"]:
                indices, sampling_report = _integer_stride_sample(
                    metadata, int(stride)
                )
                if len(indices) > int(plan["fit_observation_ceiling"]):
                    raise AlternativeClusteringError(
                        f"{algorithm} integer stride exceeds its fit ceiling"
                    )
                algorithm_settings = {
                    **settings,
                    "algorithms": [algorithm],
                    "maximum_observations": max(1, len(indices)),
                }
                selected, sweeps = _run_sampled_sweep(
                    algorithm_settings,
                    metadata,
                    transformed,
                    indices,
                    sampling_report,
                    emit_full_records=int(stride) == primary_stride,
                )
                result = {
                    "maximum_observations_per_replica": None,
                    "integer_stride": int(stride),
                    "is_primary": int(stride) == primary_stride,
                    "sampling": sampling_report,
                    "algorithm_results": selected,
                    "parameter_sweeps": sweeps,
                }
                algorithm_budget_results.append(result)
                budget_results.append(result)
            algorithm_primary = next(
                result for result in algorithm_budget_results
                if result["is_primary"]
            )
            primary_sampling_by_algorithm[str(algorithm)] = {
                **dict(plan),
                "resolved_sampling": algorithm_primary["sampling"],
            }
            primary_results.extend(algorithm_primary["algorithm_results"])
            primary_sweeps.extend(algorithm_primary["parameter_sweeps"])
            primary_reference = (
                algorithm_primary["algorithm_results"][0]
                if algorithm_primary["algorithm_results"] else None
            )
            for result in algorithm_budget_results:
                if result is algorithm_primary:
                    continue
                candidate = (
                    result["algorithm_results"][0]
                    if result["algorithm_results"] else None
                )
                labels = (
                    candidate.get("_full_assignments")
                    if isinstance(candidate, dict) else None
                )
                reference_labels = (
                    primary_reference.get("_full_assignments")
                    if isinstance(primary_reference, dict) else None
                )
                adjusted = None
                if isinstance(labels, list) and isinstance(reference_labels, list):
                    adjusted = adjusted_rand_index(reference_labels, labels)
                sensitivity.append({
                    "algorithm": algorithm,
                    "integer_stride": result["integer_stride"],
                    "sampling": result["sampling"],
                    "selected_algorithm_summaries": (
                        [_selected_summary(candidate)]
                        if isinstance(candidate, dict) else []
                    ),
                    "comparisons_to_primary": [{
                        "algorithm": algorithm,
                        "selected_parameters": (
                            candidate.get("parameters")
                            if isinstance(candidate, dict) else None
                        ),
                        "primary_selected_parameters": (
                            primary_reference.get("parameters")
                            if isinstance(primary_reference, dict) else None
                        ),
                        "selected_parameters_match_primary": (
                            isinstance(candidate, dict)
                            and isinstance(primary_reference, dict)
                            and candidate.get("parameters")
                            == primary_reference.get("parameters")
                        ),
                        "full_partition_adjusted_rand_to_primary": adjusted,
                    }],
                    "parameter_sweeps": result["parameter_sweeps"],
                })
        primary_sampling = {
            "mode": "algorithm_specific_integer_stride_v1",
            "selected_observation_count": max(
                (
                    int(value["resolved_sampling"]["selected_observation_count"])
                    for value in primary_sampling_by_algorithm.values()
                    if isinstance(value, dict) and "resolved_sampling" in value
                ),
                default=0,
            ),
            "algorithms": primary_sampling_by_algorithm,
        }
        primary = {
            "sampling": primary_sampling,
            "algorithm_results": primary_results,
            "parameter_sweeps": primary_sweeps,
        }
    else:
        plans = []
        if isinstance(sampling, dict):
            if sampling["mode"] == "integer_stride_per_replica_member_v1":
                primary_stride = int(sampling["primary_stride"])
                for stride in sampling["strides"]:
                    indices, report = _integer_stride_sample(metadata, int(stride))
                    plans.append((
                        int(stride), indices, report, int(stride) == primary_stride
                    ))
            else:
                primary_budget = int(sampling["primary_budget_per_replica"])
                for budget in sampling["budgets_per_replica"]:
                    indices, report = _balanced_uniform_sample(metadata, int(budget))
                    plans.append((
                        int(budget), indices, report, int(budget) == primary_budget
                    ))
        else:
            maximum = positive_integer(
                settings["maximum_observations"],
                "maximum_observations",
                error_type=AlternativeClusteringError,
            )
            if len(vectors) > maximum:
                raise AlternativeClusteringError(
                    "maximum_observations gate exceeded and fit_sampling is not declared"
                )
            indices = list(range(len(vectors)))
            plans.append((None, indices, {
                "mode": "all_observations_v1",
                "source_observation_count": len(vectors),
                "selected_observation_count": len(vectors),
                "selected_fraction": 1.0,
                "selected_source_matrix_indices": indices,
            }, True))
        for budget, indices, sampling_report, is_primary in plans:
            selected, sweeps = _run_sampled_sweep(
                settings,
                metadata,
                transformed,
                indices,
                sampling_report,
                emit_full_records=is_primary,
            )
            budget_results.append({
                "maximum_observations_per_replica": budget,
                "integer_stride": sampling_report.get("source_frame_stride"),
                "is_primary": is_primary,
                "sampling": sampling_report,
                "algorithm_results": selected,
                "parameter_sweeps": sweeps,
            })
        primary = next(result for result in budget_results if result["is_primary"])
        primary_by_algorithm = {
            str(result["requested_algorithm"]): result
            for result in primary["algorithm_results"]
        }
        for budget_result in budget_results:
            if budget_result is primary:
                continue
            comparisons = []
            for result in budget_result["algorithm_results"]:
                algorithm = str(result["requested_algorithm"])
                reference = primary_by_algorithm.get(algorithm)
                labels = result.get("_full_assignments")
                reference_labels = (
                    reference.get("_full_assignments")
                    if isinstance(reference, dict) else None
                )
                comparison = {
                    "algorithm": algorithm,
                    "selected_parameters": result.get("parameters"),
                    "primary_selected_parameters": (
                        reference.get("parameters")
                        if isinstance(reference, dict) else None
                    ),
                    "selected_parameters_match_primary": (
                        isinstance(reference, dict)
                        and result.get("parameters") == reference.get("parameters")
                    ),
                    "full_partition_adjusted_rand_to_primary": None,
                }
                if isinstance(labels, list) and isinstance(reference_labels, list):
                    comparison["full_partition_adjusted_rand_to_primary"] = (
                        adjusted_rand_index(reference_labels, labels)
                    )
                comparisons.append(comparison)
            sensitivity.append({
                "maximum_observations_per_replica": budget_result[
                    "maximum_observations_per_replica"
                ],
                "sampling": budget_result["sampling"],
                "selected_algorithm_summaries": [
                    _selected_summary(result)
                    for result in budget_result["algorithm_results"]
                ],
                "comparisons_to_primary": comparisons,
                "parameter_sweeps": budget_result["parameter_sweeps"],
            })
    for budget_result in budget_results:
        for result in budget_result["algorithm_results"]:
            result.pop("_full_assignments", None)
    workload_settings = {
        key: value for key, value in settings.items() if key != "fit_sampling"
    }
    workload_signature = stable_json_sha256({
        "module_id": "alternative_clustering",
        "settings_excluding_fit_sampling": workload_settings,
        "feature_contract": feature_contract,
        "full_assignment_observation_count": len(vectors),
    })
    return {
        "module_id": "alternative_clustering",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": feature_report["project_manifest_sha256"],
        "system_manifest_path": feature_report["system_manifest_path"],
        "system_manifest_sha256": feature_report["system_manifest_sha256"],
        "input_content_signature_sha256": feature_report["input_content_signature_sha256"],
        "settings": settings,
        "feature_contract": feature_contract,
        "feature_standardization": {"means": means, "scales": scales},
        "observation_count": len(vectors),
        "full_assignment_observation_count": len(vectors),
        "workload_signature_sha256": workload_signature,
        "fit_observation_count": primary["sampling"]["selected_observation_count"],
        "algorithm_fit_observation_counts": {
            str(result["requested_algorithm"]): int(result["fit_observation_count"])
            for result in primary["algorithm_results"]
        },
        "feature_count": len(vectors[0]),
        "frame_identity": metadata,
        "fit_sampling": primary["sampling"],
        "algorithm_results": primary["algorithm_results"],
        "parameter_sweeps": primary["parameter_sweeps"],
        "skipped_algorithms": [
            {
                "algorithm": row.get("algorithm"),
                "reason": row.get("skip_reason"),
                "source_observation_count": row.get(
                    "source_observation_count", len(vectors)
                ),
                "fit_observation_count": row.get(
                    "fit_observation_count",
                    primary["sampling"]["selected_observation_count"],
                ),
            }
            for row in primary["parameter_sweeps"]
            if row.get("execution_status") == "skipped"
        ],
        "sampling_sensitivity": sensitivity,
        "error_count": 0,
        "warning_count": sum(
            issue.get("severity") == "warning"
            for issue in feature_report.get("issues", [])
            if isinstance(issue, dict)
        ),
        "issues": [
            issue for issue in feature_report.get("issues", [])
            if isinstance(issue, dict)
        ],
        "limitations": [
            "Every algorithm retains its own name; partitions are not relabeled as KMeans.",
            "Validation scores describe geometric separation and do not establish metastability or kinetics.",
            "Ward and quality-threshold are skipped unless their exact fit covers every observation; quality-threshold is also skipped unless every observation is assigned.",
            "Other sample-fitted partitions report their deterministic fit frames and algorithm-specific all-frame assignment rule; approximate extensions are not equivalent to a full refit.",
            "Sample-fitted algorithms may use different planner-selected integer strides; validation comparisons therefore report each fit count explicitly.",
            "Cross-budget adjusted Rand agreement measures partition sensitivity, not physical convergence, metastability, or kinetics.",
            "PaLD is provided by the separate pald_community_analysis module and is never ranked as a conventional partition method.",
        ],
    }


def alternative_clustering_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return alternative_clustering_project(project_path, hash_content=hash_content)
    except (
        AlternativeClusteringError, ClusteringAnalysisError, TrajectoryFeatureError,
        ManifestValidationError, ImportError, OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "alternative_clustering",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "ALTERNATIVE_CLUSTERING_INVALID", "message": message}
                for message in messages
            ],
        }
