"""Deterministic clustering over declared common-PCA features."""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .feature_matrix import load_feature_matrix, parse_feature_selection
from .pca import PCAAnalysisError, common_pca_project
from .trajectory_features import TrajectoryFeatureError
from .upstream_cache import load_cached_project_report
from .state_populations import summarize_state_populations
from .validation import positive_integer


Vector = Tuple[float, ...]


class ClusteringAnalysisError(ValueError):
    """Raised when clustering configuration or numerical execution is unsafe."""


_positive_integer = partial(positive_integer, error_type=ClusteringAnalysisError)


def _kmeans_settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise ClusteringAnalysisError("project definitions.clustering_kmeans is required")
    raw = definitions.get("clustering_kmeans")
    if not isinstance(raw, dict):
        raise ClusteringAnalysisError("definitions.clustering_kmeans must be an object")
    required = {
        "feature_source",
        "standardize_features",
        "k_values",
        "random_seeds",
        "maximum_iterations",
        "center_tolerance",
        "minimum_cluster_size",
        "maximum_silhouette_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(
        required | {"component_indices", "trajectory_feature_columns"}
    ))
    if missing:
        raise ClusteringAnalysisError(
            "definitions.clustering_kmeans is missing required fields: "
            + ", ".join(missing)
        )
    if unknown:
        raise ClusteringAnalysisError(
            "definitions.clustering_kmeans contains unknown fields: "
            + ", ".join(unknown)
        )
    feature_selection = parse_feature_selection(raw, ClusteringAnalysisError)
    k_values = raw["k_values"]
    if (
        not isinstance(k_values, list)
        or not k_values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in k_values)
        or len(set(k_values)) != len(k_values)
    ):
        raise ClusteringAnalysisError("k_values must contain unique integers of at least 2")
    seeds = raw["random_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ClusteringAnalysisError(
            "random_seeds must contain at least two unique nonnegative integers"
        )
    if not isinstance(raw["standardize_features"], bool):
        raise ClusteringAnalysisError("standardize_features must be boolean")
    tolerance = raw["center_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ClusteringAnalysisError("center_tolerance must be finite and positive")
    return {
        **feature_selection,
        "standardize_features": raw["standardize_features"],
        "k_values": sorted(k_values),
        "random_seeds": list(seeds),
        "maximum_iterations": _positive_integer(raw["maximum_iterations"], "maximum_iterations"),
        "center_tolerance": float(tolerance),
        "minimum_cluster_size": _positive_integer(raw["minimum_cluster_size"], "minimum_cluster_size"),
        "maximum_silhouette_observations": _positive_integer(
            raw["maximum_silhouette_observations"],
            "maximum_silhouette_observations",
        ),
    }


def _projection_records(
    pca_report: Mapping[str, object], component_indices: Sequence[int]
) -> Tuple[List[Dict[str, object]], List[Vector]]:
    metadata: List[Dict[str, object]] = []
    vectors: List[Vector] = []
    systems = pca_report.get("systems")
    assert isinstance(systems, list)
    zero_based = [value - 1 for value in component_indices]
    for system in systems:
        assert isinstance(system, dict)
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                projections = segment["projections"]
                assert isinstance(projections, list)
                for projection in projections:
                    assert isinstance(projection, dict)
                    scores = projection["scores_angstrom"]
                    assert isinstance(scores, list)
                    if max(zero_based) >= len(scores):
                        raise ClusteringAnalysisError(
                            "component_indices exceed components returned by common_pca"
                        )
                    vector = tuple(float(scores[index]) for index in zero_based)
                    if not all(math.isfinite(value) for value in vector):
                        raise ClusteringAnalysisError("PCA feature vector is non-finite")
                    vectors.append(vector)
                    metadata.append({
                        "system_id": str(system["system_id"]),
                        "replica_id": str(replica["replica_id"]),
                        "segment_id": str(segment["segment_id"]),
                        "source_frame_index": projection["source_frame_index"],
                        **(
                            {"member_id": str(projection["member_id"])}
                            if "member_id" in projection else {}
                        ),
                        **(
                            {"sample_index": projection["sample_index"]}
                            if "sample_index" in projection
                            else {
                                "time": projection["time"],
                                "time_unit": projection["time_unit"],
                            }
                        ),
                    })
    if not vectors:
        raise ClusteringAnalysisError("common_pca produced no feature records")
    return metadata, vectors


def _standardize(
    vectors: Sequence[Vector], enabled: bool
) -> Tuple[List[Vector], Vector, Vector]:
    feature_count = len(vectors[0])
    if any(len(vector) != feature_count for vector in vectors):
        raise ClusteringAnalysisError("feature vectors have inconsistent dimensions")
    means = tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(feature_count)
    )
    scales = tuple(
        math.sqrt(
            sum((vector[index] - means[index]) ** 2 for vector in vectors)
            / len(vectors)
        )
        for index in range(feature_count)
    )
    if enabled and any(scale <= 1.0e-15 for scale in scales):
        raise ClusteringAnalysisError(
            "standardization encountered a zero-variance selected component"
        )
    if enabled:
        transformed = [
            tuple((value - means[index]) / scales[index] for index, value in enumerate(vector))
            for vector in vectors
        ]
    else:
        transformed = [tuple(vector) for vector in vectors]
        scales = tuple(1.0 for _ in range(feature_count))
        means = tuple(0.0 for _ in range(feature_count))
    return transformed, means, scales


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _initialize_kmeans_pp(vectors: Sequence[Vector], k: int, seed: int) -> List[Vector]:
    generator = random.Random(seed)
    centers = [vectors[generator.randrange(len(vectors))]]
    while len(centers) < k:
        distances = [min(_squared_distance(vector, center) for center in centers) for vector in vectors]
        total = sum(distances)
        if total <= 0.0:
            for vector in vectors:
                if vector not in centers:
                    centers.append(vector)
                    break
            else:
                raise ClusteringAnalysisError(
                    f"data contain fewer than {k} distinct feature vectors"
                )
            continue
        threshold = generator.random() * total
        cumulative = 0.0
        selected = vectors[-1]
        for vector, distance in zip(vectors, distances):
            cumulative += distance
            if cumulative >= threshold:
                selected = vector
                break
        if selected in centers:
            selected = max(
                (vector for vector in vectors if vector not in centers),
                key=lambda vector: min(_squared_distance(vector, center) for center in centers),
            )
        centers.append(selected)
    return [tuple(center) for center in centers]


def _canonicalize(
    assignments: Sequence[int], centers: Sequence[Vector]
) -> Tuple[List[int], List[Vector]]:
    order = sorted(range(len(centers)), key=lambda index: centers[index])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[value] for value in assignments], [centers[index] for index in order]


def run_kmeans(
    vectors: Sequence[Vector],
    k: int,
    seed: int,
    maximum_iterations: int,
    center_tolerance: float,
) -> Dict[str, object]:
    """Run one seeded KMeans++/Lloyd partition with deterministic ties."""

    if k > len(vectors):
        raise ClusteringAnalysisError("k cannot exceed observation count")
    centers = _initialize_kmeans_pp(vectors, k, seed)
    assignments = [-1] * len(vectors)
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        next_assignments = [
            min(range(k), key=lambda index: (_squared_distance(vector, centers[index]), index))
            for vector in vectors
        ]
        groups = [[vector for vector, label in zip(vectors, next_assignments) if label == index] for index in range(k)]
        if any(not group for group in groups):
            return {
                "valid": False,
                "failure": "empty_cluster",
                "seed": seed,
                "k": k,
                "iteration_count": iteration,
            }
        next_centers = [
            tuple(sum(vector[axis] for vector in group) / len(group) for axis in range(len(vectors[0])))
            for group in groups
        ]
        movement = max(math.sqrt(_squared_distance(a, b)) for a, b in zip(centers, next_centers))
        assignments = next_assignments
        centers = next_centers
        if movement <= center_tolerance:
            converged = True
            break
    assignments, centers = _canonicalize(assignments, centers)
    cluster_sizes = [sum(label == index for label in assignments) for index in range(k)]
    inertia = sum(
        _squared_distance(vector, centers[label])
        for vector, label in zip(vectors, assignments)
    )
    return {
        "valid": converged,
        "failure": None if converged else "maximum_iterations",
        "seed": seed,
        "k": k,
        "iteration_count": iteration,
        "converged": converged,
        "assignments": assignments,
        "centers": centers,
        "cluster_sizes": cluster_sizes,
        "inertia": inertia,
    }


def _silhouette_for_indices(
    vectors: Sequence[Vector],
    assignments: Sequence[int],
    evaluated_indices: Sequence[int],
) -> float:
    """Return mean silhouettes for declared observations against the full partition."""

    labels = sorted(set(assignments))
    if len(labels) < 2:
        raise ClusteringAnalysisError("silhouette requires at least two clusters")
    matrix = np.asarray(vectors, dtype=float)
    label_array = np.asarray(assignments)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ClusteringAnalysisError("silhouette vectors must form a finite matrix")
    clusters = {
        label: np.flatnonzero(label_array == label) for label in labels
    }
    scores = []
    for index in evaluated_indices:
        distances = np.linalg.norm(matrix - matrix[index], axis=1)
        own = clusters[assignments[index]]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a = float(distances[own].sum()) / (len(own) - 1)
        b = min(
            float(distances[clusters[label]].mean())
            for label in labels if label != assignments[index]
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return sum(scores) / len(scores)


def silhouette_score(vectors: Sequence[Vector], assignments: Sequence[int]) -> float:
    """Return the exact mean silhouette for one complete partition."""

    return _silhouette_for_indices(vectors, assignments, list(range(len(vectors))))


def silhouette_score_report(
    vectors: Sequence[Vector],
    assignments: Sequence[int],
    maximum_exact_observations: int,
    random_seed: int = 0,
) -> Dict[str, object]:
    """Return an exact or seeded observation-subsampled silhouette estimate.

    Estimated silhouettes evaluate a deterministic random subset of focal
    observations against every member of the complete fitted partition.  They
    are therefore not the silhouette of a refitted or sample-only partition.
    """

    maximum = _positive_integer(
        maximum_exact_observations, "maximum_exact_observations"
    )
    if len(vectors) != len(assignments) or not vectors:
        raise ClusteringAnalysisError(
            "silhouette vectors and assignments must have equal nonzero length"
        )
    if len(vectors) <= maximum:
        indices = list(range(len(vectors)))
        method = "exact_all_observations"
        seed: Optional[int] = None
    else:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ClusteringAnalysisError("silhouette random_seed must be an integer")
        indices = sorted(random.Random(random_seed).sample(range(len(vectors)), maximum))
        method = "seeded_focal_observation_subsample_against_full_partition"
        seed = random_seed
    return {
        "score": _silhouette_for_indices(vectors, assignments, indices),
        "method": method,
        "estimated": len(indices) < len(vectors),
        "total_observation_count": len(vectors),
        "evaluated_observation_count": len(indices),
        "random_seed": seed,
        "evaluated_observation_indices": indices,
    }


def adjusted_rand_index(first: Sequence[int], second: Sequence[int]) -> float:
    """Return the adjusted Rand index for two partitions of the same records."""

    if len(first) != len(second) or not first:
        raise ClusteringAnalysisError("ARI partitions must have equal nonzero length")
    first_labels = sorted(set(first))
    second_labels = sorted(set(second))
    table = {
        (left, right): sum(a == left and b == right for a, b in zip(first, second))
        for left in first_labels for right in second_labels
    }
    choose_two = lambda value: value * (value - 1) / 2
    sum_cells = sum(choose_two(value) for value in table.values())
    row_sums = [sum(table[(left, right)] for right in second_labels) for left in first_labels]
    column_sums = [sum(table[(left, right)] for left in first_labels) for right in second_labels]
    total_pairs = choose_two(len(first))
    if total_pairs == 0:
        return 1.0
    expected = sum(choose_two(value) for value in row_sums) * sum(
        choose_two(value) for value in column_sums
    ) / total_pairs
    maximum = 0.5 * (
        sum(choose_two(value) for value in row_sums)
        + sum(choose_two(value) for value in column_sums)
    )
    denominator = maximum - expected
    return (sum_cells - expected) / denominator if denominator else 1.0


def clustering_kmeans_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Scan a declared KMeans grid and select a complete stable partition."""

    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "clustering_kmeans",
        source,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    settings = _kmeans_settings(project)
    feature_report, metadata, raw_vectors, feature_contract = load_feature_matrix(
        source,
        settings,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    vectors, means, scales = _standardize(
        raw_vectors, bool(settings["standardize_features"])
    )
    if max(settings["k_values"]) > len(vectors):  # type: ignore[arg-type]
        raise ClusteringAnalysisError("k_values cannot exceed observation count")
    diagnostics = []
    candidates = []
    for k in settings["k_values"]:  # type: ignore[union-attr]
        runs = [
            run_kmeans(
                vectors,
                int(k),
                int(seed),
                int(settings["maximum_iterations"]),
                float(settings["center_tolerance"]),
            )
            for seed in settings["random_seeds"]  # type: ignore[union-attr]
        ]
        valid_runs = [
            run for run in runs
            if run["valid"]
            and min(run["cluster_sizes"]) >= int(settings["minimum_cluster_size"])  # type: ignore[arg-type]
        ]
        best = min(valid_runs, key=lambda run: (run["inertia"], run["seed"])) if valid_runs else None
        if best is not None:
            silhouette_evaluation = silhouette_score_report(
                vectors,
                best["assignments"],  # type: ignore[arg-type]
                int(settings["maximum_silhouette_observations"]),
                min(settings["random_seeds"]),  # type: ignore[arg-type]
            )
            silhouette = float(silhouette_evaluation["score"])
            stability_values = [
                adjusted_rand_index(best["assignments"], run["assignments"])  # type: ignore[arg-type]
                for run in valid_runs if run is not best
            ]
            stability = (
                sum(stability_values) / len(stability_values)
                if stability_values else 1.0
            )
            candidate = {
                "k": int(k),
                "best_run": best,
                "silhouette": silhouette,
                "silhouette_evaluation": silhouette_evaluation,
                "mean_adjusted_rand_to_best": stability,
                "valid_seed_count": len(valid_runs),
            }
            candidates.append(candidate)
            diagnostics.append({
                "k": int(k),
                "eligible": True,
                "selected_seed": best["seed"],
                "selected_inertia": best["inertia"],
                "selected_cluster_sizes": best["cluster_sizes"],
                "silhouette": silhouette,
                "mean_adjusted_rand_to_best": stability,
                "valid_seed_count": len(valid_runs),
                "runs": [
                    {key: run.get(key) for key in ("seed", "valid", "failure", "iteration_count", "inertia", "cluster_sizes")}
                    for run in runs
                ],
            })
        else:
            diagnostics.append({
                "k": int(k),
                "eligible": False,
                "reason": "no converged seed passed minimum_cluster_size",
                "valid_seed_count": 0,
                "runs": [
                    {key: run.get(key) for key in ("seed", "valid", "failure", "iteration_count", "inertia", "cluster_sizes")}
                    for run in runs
                ],
            })
    if not candidates:
        raise ClusteringAnalysisError(
            "no KMeans grid candidate passed convergence and occupancy gates"
        )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["silhouette"],
            candidate["mean_adjusted_rand_to_best"],
            -candidate["k"],
        ),
    )
    best_run = selected["best_run"]
    assignments = best_run["assignments"]
    centers = best_run["centers"]
    raw_centers = [
        [center[index] * scales[index] + means[index] for index in range(len(center))]
        for center in centers
    ]
    assignment_rows = []
    for record, raw_vector, vector, label in zip(
        metadata, raw_vectors, vectors, assignments
    ):
        assignment_rows.append({
            **record,
            "feature_values": list(raw_vector),
            **(
                {"features_angstrom": list(raw_vector)}
                if settings["feature_source"] == "common_pca" else {}
            ),
            "cluster_id": int(label) + 1,
            "squared_distance_in_clustering_space": _squared_distance(
                vector, centers[label]
            ),
        })
    issues = [
        issue for issue in feature_report.get("issues", []) if isinstance(issue, dict)
    ]
    return {
        "module_id": "clustering_kmeans",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": feature_report["project_manifest_sha256"],
        "system_manifest_path": feature_report["system_manifest_path"],
        "system_manifest_sha256": feature_report["system_manifest_sha256"],
        "input_content_signature_sha256": feature_report["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "feature_contract": feature_contract | {
            "standardization_means": list(means),
            "standardization_scales": list(scales),
        },
        "selection_rule": (
            "maximum exact or prespecified seeded-estimate silhouette, then maximum mean ARI to selected-seed partition, "
            "then smaller k; only converged complete partitions meeting minimum_cluster_size are eligible"
        ),
        "grid_diagnostics": diagnostics,
        "selected_model": {
            "k": selected["k"],
            "seed": best_run["seed"],
            "iteration_count": best_run["iteration_count"],
            "inertia": best_run["inertia"],
            "silhouette": selected["silhouette"],
            "silhouette_evaluation": selected["silhouette_evaluation"],
            "mean_adjusted_rand_to_best": selected["mean_adjusted_rand_to_best"],
            "cluster_sizes": best_run["cluster_sizes"],
            "centers_in_input_units": raw_centers,
            **(
                {"centers_angstrom": raw_centers}
                if settings["feature_source"] == "common_pca" else {}
            ),
            "centers_in_clustering_space": [list(center) for center in centers],
        },
        "assignments": assignment_rows,
        "state_population_comparison": summarize_state_populations(
            assignment_rows, "cluster_id"
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "KMeans assumes complete convex Voronoi partitions and Euclidean geometry in the declared feature space.",
            "Silhouette and seed stability do not establish physical metastability, kinetics, or convergence.",
            "Association of cluster labels with system, replica, or preparation identity is a reported scientific characteristic of the fitted partition, not a technical failure or a rule for discarding assignments, populations, representatives, or trajectories.",
            "The k grid, feature definitions, standardization, seeds, and occupancy gate require sensitivity analysis.",
            "Frame assignments are not independent observations for uncertainty estimation.",
            "Technical completion does not establish scientific validity.",
        ],
    }


def clustering_kmeans_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Return a machine-readable failure rather than an uncaught exception."""

    try:
        return clustering_kmeans_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, PCAAnalysisError, TrajectoryFeatureError,
        ClusteringAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "clustering_kmeans",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "KMEANS_INVALID", "message": message}
                for message in messages
            ],
        }


def _imwkmeans_settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise ClusteringAnalysisError("project definitions.clustering_imwkmeans is required")
    raw = definitions.get("clustering_imwkmeans")
    if not isinstance(raw, dict):
        raise ClusteringAnalysisError("definitions.clustering_imwkmeans must be an object")
    required = {
        "feature_source", "standardize_features",
        "k_values", "minkowski_p_values", "initialization_ranks",
        "maximum_iterations", "objective_tolerance", "minimum_cluster_size",
        "weight_dispersion_floor", "maximum_silhouette_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(
        required | {"component_indices", "trajectory_feature_columns"}
    ))
    if missing:
        raise ClusteringAnalysisError(
            "definitions.clustering_imwkmeans is missing required fields: "
            + ", ".join(missing)
        )
    if unknown:
        raise ClusteringAnalysisError(
            "definitions.clustering_imwkmeans contains unknown fields: "
            + ", ".join(unknown)
        )
    feature_selection = parse_feature_selection(raw, ClusteringAnalysisError)
    k_values = raw["k_values"]
    if (
        not isinstance(k_values, list) or not k_values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in k_values)
        or len(set(k_values)) != len(k_values)
    ):
        raise ClusteringAnalysisError("k_values must contain unique integers of at least 2")
    p_values = raw["minkowski_p_values"]
    if (
        not isinstance(p_values, list) or not p_values
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 1.0
            for value in p_values
        )
        or len({float(value) for value in p_values}) != len(p_values)
    ):
        raise ClusteringAnalysisError(
            "minkowski_p_values must contain unique finite values greater than 1"
        )
    ranks = raw["initialization_ranks"]
    if (
        not isinstance(ranks, list) or len(ranks) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ranks)
        or len(set(ranks)) != len(ranks)
    ):
        raise ClusteringAnalysisError(
            "initialization_ranks must contain at least two unique nonnegative integers"
        )
    if not isinstance(raw["standardize_features"], bool):
        raise ClusteringAnalysisError("standardize_features must be boolean")
    for label in ("objective_tolerance", "weight_dispersion_floor"):
        value = raw[label]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise ClusteringAnalysisError(f"{label} must be finite and positive")
    return {
        **feature_selection,
        "standardize_features": raw["standardize_features"],
        "k_values": sorted(k_values),
        "minkowski_p_values": sorted(float(value) for value in p_values),
        "initialization_ranks": list(ranks),
        "maximum_iterations": _positive_integer(raw["maximum_iterations"], "maximum_iterations"),
        "objective_tolerance": float(raw["objective_tolerance"]),
        "minimum_cluster_size": _positive_integer(raw["minimum_cluster_size"], "minimum_cluster_size"),
        "weight_dispersion_floor": float(raw["weight_dispersion_floor"]),
        "maximum_silhouette_observations": _positive_integer(
            raw["maximum_silhouette_observations"],
            "maximum_silhouette_observations",
        ),
    }


def _lp_center(values: Sequence[float], p: float) -> float:
    if p == 2.0:
        return sum(values) / len(values)
    lower = min(values)
    upper = max(values)
    if lower == upper:
        return lower
    for _ in range(100):
        middle = (lower + upper) / 2.0
        derivative = sum(
            (1.0 if middle > value else -1.0 if middle < value else 0.0)
            * abs(middle - value) ** (p - 1.0)
            for value in values
        )
        if derivative > 0.0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2.0


def _minkowski_distance_power(
    vector: Sequence[float],
    center: Sequence[float],
    weights: Sequence[float],
    p: float,
) -> float:
    return sum(
        (weight ** p) * (abs(value - center_value) ** p)
        for value, center_value, weight in zip(vector, center, weights)
    )


def _intelligent_centers(
    vectors: Sequence[Vector], k: int, p: float, initialization_rank: int
) -> List[Vector]:
    global_center = tuple(
        _lp_center([vector[index] for vector in vectors], p)
        for index in range(len(vectors[0]))
    )
    equal_weights = tuple(1.0 / len(vectors[0]) for _ in vectors[0])
    ordered = sorted(
        range(len(vectors)),
        key=lambda index: (
            -_minkowski_distance_power(vectors[index], global_center, equal_weights, p),
            index,
        ),
    )
    first = ordered[initialization_rank % len(ordered)]
    selected = [first]
    while len(selected) < k:
        candidates = [index for index in range(len(vectors)) if index not in selected]
        if not candidates:
            raise ClusteringAnalysisError(f"data contain fewer than {k} distinct initial centers")
        next_index = max(
            candidates,
            key=lambda index: (
                min(
                    _minkowski_distance_power(
                        vectors[index], vectors[center_index], equal_weights, p
                    )
                    for center_index in selected
                ),
                -index,
            ),
        )
        selected.append(next_index)
    return [vectors[index] for index in selected]


def run_imwkmeans(
    vectors: Sequence[Vector],
    k: int,
    p: float,
    initialization_rank: int,
    maximum_iterations: int,
    objective_tolerance: float,
    dispersion_floor: float,
) -> Dict[str, object]:
    """Run intelligent Minkowski weighted KMeans with cluster feature weights."""

    if k > len(vectors):
        raise ClusteringAnalysisError("k cannot exceed observation count")
    feature_count = len(vectors[0])
    centers = _intelligent_centers(vectors, k, p, initialization_rank)
    weights = [tuple(1.0 / feature_count for _ in range(feature_count)) for _ in range(k)]
    assignments = [-1] * len(vectors)
    previous_objective: Optional[float] = None
    converged = False
    objective = math.inf
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        next_assignments = [
            min(
                range(k),
                key=lambda cluster: (
                    _minkowski_distance_power(vector, centers[cluster], weights[cluster], p),
                    cluster,
                ),
            )
            for vector in vectors
        ]
        groups = [
            [vector for vector, label in zip(vectors, next_assignments) if label == cluster]
            for cluster in range(k)
        ]
        if any(not group for group in groups):
            return {
                "valid": False,
                "failure": "empty_cluster",
                "k": k,
                "p": p,
                "initialization_rank": initialization_rank,
                "iteration_count": iteration,
            }
        next_centers = [
            tuple(
                _lp_center([vector[feature] for vector in group], p)
                for feature in range(feature_count)
            )
            for group in groups
        ]
        next_weights = []
        for group, center in zip(groups, next_centers):
            dispersions = [
                max(
                    dispersion_floor,
                    sum(abs(vector[feature] - center[feature]) ** p for vector in group),
                )
                for feature in range(feature_count)
            ]
            raw = [value ** (-1.0 / (p - 1.0)) for value in dispersions]
            total = sum(raw)
            next_weights.append(tuple(value / total for value in raw))
        objective = sum(
            _minkowski_distance_power(
                vector, next_centers[label], next_weights[label], p
            )
            for vector, label in zip(vectors, next_assignments)
        )
        assignment_stable = next_assignments == assignments
        relative_change = (
            math.inf
            if previous_objective is None
            else abs(previous_objective - objective) / max(1.0, abs(previous_objective))
        )
        assignments = next_assignments
        centers = next_centers
        weights = next_weights
        if assignment_stable or relative_change <= objective_tolerance:
            converged = True
            break
        previous_objective = objective
    order = sorted(range(k), key=lambda index: centers[index])
    remap = {old: new for new, old in enumerate(order)}
    assignments = [remap[label] for label in assignments]
    centers = [centers[index] for index in order]
    weights = [weights[index] for index in order]
    sizes = [sum(label == cluster for label in assignments) for cluster in range(k)]
    return {
        "valid": converged,
        "failure": None if converged else "maximum_iterations",
        "k": k,
        "p": p,
        "initialization_rank": initialization_rank,
        "iteration_count": iteration,
        "converged": converged,
        "objective": objective,
        "assignments": assignments,
        "centers": centers,
        "feature_weights": weights,
        "cluster_sizes": sizes,
    }


_IMWKMEANS_CHECKPOINT_ENV = "SALSBURY_MD_ANALYSIS_IMWKMEANS_CHECKPOINT"
_IMWKMEANS_CHECKPOINT_SCHEMA = "salsbury-imwkmeans-grid-checkpoint-v1"


def _imwkmeans_checkpoint_root() -> Optional[Path]:
    raw = os.environ.get(_IMWKMEANS_CHECKPOINT_ENV)
    if raw is None or not raw.strip():
        return None
    root = Path(raw).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise ClusteringAnalysisError(
            f"{_IMWKMEANS_CHECKPOINT_ENV} is not a directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _imwkmeans_checkpoint_signature(
    feature_report: Mapping[str, object],
    settings: Mapping[str, object],
) -> str:
    payload = {
        "schema": _IMWKMEANS_CHECKPOINT_SCHEMA,
        "project_manifest_sha256": feature_report.get("project_manifest_sha256"),
        "system_manifest_sha256": feature_report.get("system_manifest_sha256"),
        "input_content_signature_sha256": feature_report.get(
            "input_content_signature_sha256"
        ),
        "settings": settings,
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _imwkmeans_checkpoint_path(root: Path, k: int, p: float) -> Path:
    p_token = float(p).hex().replace("+", "p").replace("-", "m").replace(".", "d")
    return root / f"k-{int(k)}-p-{p_token}.json"


def _load_imwkmeans_checkpoint(
    path: Path,
    signature: str,
    k: int,
    p: float,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    try:
        checkpoint = load_json(path)
    except (OSError, ValueError) as exc:
        raise ClusteringAnalysisError(
            f"invalid iMWK-Means checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ClusteringAnalysisError(f"iMWK-Means checkpoint is not an object: {path}")
    if checkpoint.get("schema") != _IMWKMEANS_CHECKPOINT_SCHEMA:
        raise ClusteringAnalysisError(f"iMWK-Means checkpoint has wrong schema: {path}")
    if checkpoint.get("checkpoint_signature_sha256") != signature:
        raise ClusteringAnalysisError(
            f"iMWK-Means checkpoint does not match current inputs/settings: {path}"
        )
    if checkpoint.get("k") != int(k) or checkpoint.get("p") != float(p):
        raise ClusteringAnalysisError(
            f"iMWK-Means checkpoint grid identity does not match its filename: {path}"
        )
    candidate = checkpoint.get("candidate")
    diagnostic = checkpoint.get("diagnostic")
    if candidate is not None and not isinstance(candidate, dict):
        raise ClusteringAnalysisError(f"iMWK-Means checkpoint candidate is invalid: {path}")
    if not isinstance(diagnostic, dict):
        raise ClusteringAnalysisError(f"iMWK-Means checkpoint diagnostic is invalid: {path}")
    return candidate, diagnostic


def _write_imwkmeans_checkpoint(
    path: Path,
    signature: str,
    k: int,
    p: float,
    candidate: Optional[Mapping[str, object]],
    diagnostic: Mapping[str, object],
) -> None:
    payload = {
        "schema": _IMWKMEANS_CHECKPOINT_SCHEMA,
        "checkpoint_signature_sha256": signature,
        "k": int(k),
        "p": float(p),
        "candidate": candidate,
        "diagnostic": diagnostic,
    }
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ClusteringAnalysisError(
            f"could not install iMWK-Means checkpoint {path}: {exc}"
        ) from exc


def clustering_imwkmeans_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run iMWK-Means k/p/initialization sensitivity scans."""

    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "clustering_imwkmeans",
        source,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    settings = _imwkmeans_settings(project)
    feature_report, metadata, raw_vectors, feature_contract = load_feature_matrix(
        source,
        settings,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    vectors, means, scales = _standardize(
        raw_vectors, bool(settings["standardize_features"])
    )
    checkpoint_root = _imwkmeans_checkpoint_root()
    checkpoint_signature = _imwkmeans_checkpoint_signature(feature_report, settings)
    restored_checkpoint_count = 0
    written_checkpoint_count = 0
    candidates = []
    diagnostics = []
    for k in settings["k_values"]:  # type: ignore[union-attr]
        for p in settings["minkowski_p_values"]:  # type: ignore[union-attr]
            checkpoint_path = (
                _imwkmeans_checkpoint_path(checkpoint_root, int(k), float(p))
                if checkpoint_root is not None else None
            )
            if checkpoint_path is not None and checkpoint_path.exists():
                candidate, diagnostic = _load_imwkmeans_checkpoint(
                    checkpoint_path, checkpoint_signature, int(k), float(p)
                )
                restored_checkpoint_count += 1
            else:
                runs = [
                    run_imwkmeans(
                        vectors,
                        int(k),
                        float(p),
                        int(rank),
                        int(settings["maximum_iterations"]),
                        float(settings["objective_tolerance"]),
                        float(settings["weight_dispersion_floor"]),
                    )
                    for rank in settings["initialization_ranks"]  # type: ignore[union-attr]
                ]
                valid = [
                    run for run in runs
                    if run["valid"]
                    and min(run["cluster_sizes"]) >= int(settings["minimum_cluster_size"])  # type: ignore[arg-type]
                ]
                best = min(
                    valid,
                    key=lambda run: (run["objective"], run["initialization_rank"]),
                ) if valid else None
                if best is None:
                    candidate = None
                    diagnostic = {
                    "k": int(k), "p": float(p), "eligible": False,
                    "reason": "no converged initialization passed occupancy gates",
                    }
                else:
                    silhouette_evaluation = silhouette_score_report(
                        vectors,
                        best["assignments"],  # type: ignore[arg-type]
                        int(settings["maximum_silhouette_observations"]),
                        min(settings["initialization_ranks"]),  # type: ignore[arg-type]
                    )
                    silhouette = float(silhouette_evaluation["score"])
                    ari_values = [
                        adjusted_rand_index(best["assignments"], run["assignments"])  # type: ignore[arg-type]
                        for run in valid if run is not best
                    ]
                    stability = sum(ari_values) / len(ari_values) if ari_values else 1.0
                    candidate = {
                        "k": int(k), "p": float(p), "best_run": best,
                        "silhouette": silhouette,
                        "silhouette_evaluation": silhouette_evaluation,
                        "mean_adjusted_rand_to_best": stability,
                        "valid_initialization_count": len(valid),
                    }
                    diagnostic = {
                        "k": int(k), "p": float(p), "eligible": True,
                        "selected_initialization_rank": best["initialization_rank"],
                        "selected_objective": best["objective"],
                        "selected_cluster_sizes": best["cluster_sizes"],
                        "silhouette": silhouette,
                        "mean_adjusted_rand_to_best": stability,
                        "valid_initialization_count": len(valid),
                    }
                if checkpoint_path is not None:
                    _write_imwkmeans_checkpoint(
                        checkpoint_path, checkpoint_signature, int(k), float(p),
                        candidate, diagnostic,
                    )
                    written_checkpoint_count += 1
            diagnostics.append(diagnostic)
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        raise ClusteringAnalysisError("no iMWK-Means grid candidate passed all gates")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["silhouette"],
            candidate["mean_adjusted_rand_to_best"],
            -candidate["k"],
            -candidate["p"],
        ),
    )
    run = selected["best_run"]
    centers = run["centers"]
    assignments = run["assignments"]
    raw_centers = [
        [center[index] * scales[index] + means[index] for index in range(len(center))]
        for center in centers
    ]
    assignment_rows = [
        {
            **record,
            "feature_values": list(raw_vector),
            **(
                {"features_angstrom": list(raw_vector)}
                if settings["feature_source"] == "common_pca" else {}
            ),
            "cluster_id": int(label) + 1,
            "weighted_minkowski_distance_power": _minkowski_distance_power(
                vector,
                centers[label],
                run["feature_weights"][label],
                float(run["p"]),
            ),
        }
        for record, raw_vector, vector, label in zip(
            metadata, raw_vectors, vectors, assignments
        )
    ]
    issues = [
        issue for issue in feature_report.get("issues", []) if isinstance(issue, dict)
    ]
    return {
        "module_id": "clustering_imwkmeans",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": feature_report["project_manifest_sha256"],
        "system_manifest_path": feature_report["system_manifest_path"],
        "system_manifest_sha256": feature_report["system_manifest_sha256"],
        "input_content_signature_sha256": feature_report["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "algorithm_contract": {
            "name": "intelligent Minkowski weighted KMeans",
            "initialization": "ranked farthest observation from global Lp center, then deterministic farthest-first",
            "cluster_center": "coordinate-wise Lp minimizer",
            "feature_weight": "inverse within-cluster Lp dispersion normalized to sum one",
            "distance": "sum(weight_feature^p * absolute_deviation^p)",
        },
        "checkpoint_restart": {
            "enabled": checkpoint_root is not None,
            "environment_variable": _IMWKMEANS_CHECKPOINT_ENV,
            "checkpoint_directory": (
                str(checkpoint_root) if checkpoint_root is not None else None
            ),
            "checkpoint_signature_sha256": checkpoint_signature,
            "granularity": "one complete k/p grid candidate including all initialization ranks",
            "restored_candidate_count": restored_checkpoint_count,
            "written_candidate_count": written_checkpoint_count,
            "atomic_install": True,
        },
        "feature_contract": feature_contract | {
            "standardization_means": list(means),
            "standardization_scales": list(scales),
        },
        "selection_rule": (
            "maximum exact or prespecified seeded-estimate Euclidean silhouette, then maximum initialization ARI stability, "
            "then smaller k and p among converged occupancy-gated iMWK-Means models"
        ),
        "grid_diagnostics": diagnostics,
        "selected_model": {
            "k": selected["k"],
            "p": selected["p"],
            "initialization_rank": run["initialization_rank"],
            "iteration_count": run["iteration_count"],
            "objective": run["objective"],
            "silhouette": selected["silhouette"],
            "silhouette_evaluation": selected["silhouette_evaluation"],
            "mean_adjusted_rand_to_best": selected["mean_adjusted_rand_to_best"],
            "cluster_sizes": run["cluster_sizes"],
            "centers_in_input_units": raw_centers,
            **(
                {"centers_angstrom": raw_centers}
                if settings["feature_source"] == "common_pca" else {}
            ),
            "centers_in_clustering_space": [list(center) for center in centers],
            "feature_weights": [list(weights) for weights in run["feature_weights"]],
        },
        "assignments": assignment_rows,
        "state_population_comparison": summarize_state_populations(
            assignment_rows, "cluster_id"
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Feature weights are cluster-specific descriptors and are not mechanistic importance scores.",
            "Silhouette is evaluated in Euclidean clustering space for cross-method comparison, while the fitted objective is weighted Minkowski.",
            "Initialization, p, k, feature definitions, standardization, and dispersion floors require sensitivity analysis.",
            "Cluster partitions do not establish metastability, kinetics, convergence, or scientific validity.",
        ],
    }


def clustering_imwkmeans_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return clustering_imwkmeans_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, PCAAnalysisError, TrajectoryFeatureError,
        ClusteringAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "clustering_imwkmeans",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "IMWKMEANS_INVALID", "message": message}
                for message in messages
            ],
        }


def _hdbscan_settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("clustering_hdbscan") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise ClusteringAnalysisError("definitions.clustering_hdbscan must be an object")
    required = {
        "feature_source", "standardize_features",
        "minimum_cluster_sizes", "minimum_samples_values",
        "cluster_selection_method", "allow_single_cluster",
        "minimum_retained_fraction", "maximum_silhouette_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(
        required | {"component_indices", "trajectory_feature_columns"}
    ))
    if missing:
        raise ClusteringAnalysisError("HDBSCAN settings missing: " + ", ".join(missing))
    if unknown:
        raise ClusteringAnalysisError("HDBSCAN settings contain unknown fields: " + ", ".join(unknown))
    feature_selection = parse_feature_selection(raw, ClusteringAnalysisError)
    for label in ("minimum_cluster_sizes", "minimum_samples_values"):
        values = raw[label]
        if (
            not isinstance(values, list) or not values
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ClusteringAnalysisError(f"{label} must contain unique positive integers")
    if raw["cluster_selection_method"] not in {"eom", "leaf"}:
        raise ClusteringAnalysisError("cluster_selection_method must be eom or leaf")
    if not isinstance(raw["allow_single_cluster"], bool) or not isinstance(raw["standardize_features"], bool):
        raise ClusteringAnalysisError("HDBSCAN boolean settings are malformed")
    retained = raw["minimum_retained_fraction"]
    if isinstance(retained, bool) or not isinstance(retained, (int, float)) or not 0.0 < float(retained) <= 1.0:
        raise ClusteringAnalysisError("minimum_retained_fraction must be in (0, 1]")
    maximum = _positive_integer(raw["maximum_silhouette_observations"], "maximum_silhouette_observations")
    return {
        **feature_selection,
        "standardize_features": raw["standardize_features"],
        "minimum_cluster_sizes": sorted(raw["minimum_cluster_sizes"]),
        "minimum_samples_values": sorted(raw["minimum_samples_values"]),
        "cluster_selection_method": raw["cluster_selection_method"],
        "allow_single_cluster": raw["allow_single_cluster"],
        "minimum_retained_fraction": float(retained),
        "maximum_silhouette_observations": maximum,
    }


def _canonical_hdbscan_labels(
    vectors: Sequence[Vector], labels: Sequence[int]
) -> Tuple[List[int], List[Vector]]:
    raw_labels = sorted(set(label for label in labels if label >= 0))
    centers = []
    for label in raw_labels:
        members = [vector for vector, assigned in zip(vectors, labels) if assigned == label]
        centers.append(tuple(
            sum(vector[feature] for vector in members) / len(members)
            for feature in range(len(vectors[0]))
        ))
    order = sorted(range(len(raw_labels)), key=lambda index: centers[index])
    remap = {raw_labels[old]: new for new, old in enumerate(order)}
    return [remap[label] if label >= 0 else -1 for label in labels], [centers[index] for index in order]


def clustering_hdbscan_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run the reference hdbscan package as a noise-aware sensitivity scan."""

    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "clustering_hdbscan",
        source,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    settings = _hdbscan_settings(project)
    try:
        package = importlib.import_module("hdbscan")
    except ImportError as exc:
        raise ClusteringAnalysisError(
            "optional dependency hdbscan is unavailable; install the hdbscan extra to run this module"
        ) from exc
    if not hasattr(package, "HDBSCAN"):
        raise ClusteringAnalysisError("imported hdbscan package does not expose HDBSCAN")
    feature_report, metadata, raw_vectors, feature_contract = load_feature_matrix(
        source,
        settings,
        hash_content=hash_content,
        error_type=ClusteringAnalysisError,
    )
    vectors, means, scales = _standardize(raw_vectors, bool(settings["standardize_features"]))
    candidates = []
    diagnostics = []
    for minimum_cluster_size in settings["minimum_cluster_sizes"]:  # type: ignore[union-attr]
        for minimum_samples in settings["minimum_samples_values"]:  # type: ignore[union-attr]
            model = package.HDBSCAN(
                min_cluster_size=int(minimum_cluster_size),
                min_samples=int(minimum_samples),
                metric="euclidean",
                cluster_selection_method=str(settings["cluster_selection_method"]),
                allow_single_cluster=bool(settings["allow_single_cluster"]),
                core_dist_n_jobs=1,
            )
            raw_labels = [int(value) for value in model.fit_predict(vectors)]
            if len(raw_labels) != len(vectors):
                raise ClusteringAnalysisError("HDBSCAN returned an assignment count mismatch")
            labels, centers = _canonical_hdbscan_labels(vectors, raw_labels)
            retained_indices = [index for index, label in enumerate(labels) if label >= 0]
            retained_fraction = len(retained_indices) / len(labels)
            cluster_count = len(set(label for label in labels if label >= 0))
            cluster_sizes = [labels.count(label) for label in range(cluster_count)]
            score = None
            silhouette_evaluation = None
            if cluster_count >= 2 and len(retained_indices) > cluster_count:
                silhouette_evaluation = silhouette_score_report(
                    [vectors[index] for index in retained_indices],
                    [labels[index] for index in retained_indices],
                    int(settings["maximum_silhouette_observations"]),
                    0,
                )
                score = float(silhouette_evaluation["score"])
            eligible = (
                retained_fraction >= float(settings["minimum_retained_fraction"])
                and cluster_count >= (1 if settings["allow_single_cluster"] else 2)
                and score is not None
            )
            diagnostic = {
                "minimum_cluster_size": int(minimum_cluster_size),
                "minimum_samples": int(minimum_samples), "eligible": eligible,
                "cluster_count": cluster_count, "cluster_sizes": cluster_sizes,
                "retained_count": len(retained_indices), "noise_count": labels.count(-1),
                "retained_fraction": retained_fraction,
                "retained_only_silhouette": score,
                "silhouette_evaluation": silhouette_evaluation,
            }
            diagnostics.append(diagnostic)
            if eligible:
                candidates.append({**diagnostic, "labels": labels, "centers": centers})
    if not candidates:
        raise ClusteringAnalysisError("no HDBSCAN grid candidate passed retained-coverage and cluster gates")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["retained_only_silhouette"], candidate["retained_fraction"],
            -candidate["minimum_cluster_size"], -candidate["minimum_samples"],
        ),
    )
    labels = selected["labels"]
    centers = selected["centers"]
    assignment_rows = [{
        **record, "feature_values": list(raw_vector),
        **(
            {"features_angstrom": list(raw_vector)}
            if settings["feature_source"] == "common_pca" else {}
        ),
        "cluster_id": int(label) + 1 if label >= 0 else None,
        "is_noise": label < 0,
        "squared_distance_in_clustering_space": (
            _squared_distance(vector, centers[label]) if label >= 0 else None
        ),
    } for record, raw_vector, vector, label in zip(
        metadata, raw_vectors, vectors, labels
    )]
    issues = [issue for issue in feature_report.get("issues", []) if isinstance(issue, dict)]
    issues.append({
        "severity": "warning", "code": "NOISE_EXCLUDING_SENSITIVITY_PARTITION",
        "location": str(source),
        "message": "HDBSCAN labels include noise and are not a complete state assignment for kinetics",
    })
    try:
        hdbscan_version = importlib_metadata.version("hdbscan")
    except importlib_metadata.PackageNotFoundError:
        hdbscan_version = getattr(package, "__version__", "unknown")
    return {
        "module_id": "clustering_hdbscan", "technical_status": "complete",
        "scientific_status": "not evaluated", "project_manifest_path": str(source),
        "project_manifest_sha256": feature_report["project_manifest_sha256"],
        "system_manifest_path": feature_report["system_manifest_path"],
        "system_manifest_sha256": feature_report["system_manifest_sha256"],
        "input_content_signature_sha256": feature_report["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "implementation": {
            "package": "hdbscan", "version": hdbscan_version,
            "class": "hdbscan.HDBSCAN", "metric": "euclidean",
        },
        "settings": settings,
        "feature_contract": feature_contract | {
            "standardization_means": list(means),
            "standardization_scales": list(scales),
        },
        "grid_diagnostics": diagnostics,
        "selection_rule": "maximum exact or prespecified seeded-estimate retained-only silhouette, then retained fraction, then smaller parameters",
        "selected_model": {
            key: selected[key] for key in (
                "minimum_cluster_size", "minimum_samples", "cluster_count", "cluster_sizes",
                "retained_count", "noise_count", "retained_fraction", "retained_only_silhouette",
            )
        } | {"silhouette_evaluation": selected["silhouette_evaluation"]},
        "assignments": assignment_rows,
        "state_population_comparison": summarize_state_populations(
            assignment_rows, "cluster_id"
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Noise-excluding partitions are sensitivity analyses and are not complete kinetic state assignments.",
            "Silhouette is evaluated only on retained observations and must be read with retained coverage.",
            "Density parameters, standardization, and feature definitions require sensitivity analysis.",
            "Clusters do not establish metastability, convergence, kinetics, or scientific validity.",
        ],
    }


def clustering_hdbscan_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return clustering_hdbscan_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, PCAAnalysisError, TrajectoryFeatureError,
        ClusteringAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "clustering_hdbscan", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "HDBSCAN_INVALID", "message": message} for message in messages],
        }
