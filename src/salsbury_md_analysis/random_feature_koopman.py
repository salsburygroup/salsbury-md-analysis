"""Random-Fourier-feature approximation of nonlinear molecular kinetics.

The implementation follows the TICA-RFF construction: prescaled linear TICA
coordinates are mapped through an isotropic Gaussian random Fourier dictionary,
then a reversible time-lagged covariance problem is solved in that dictionary.
Several prespecified feature-map seeds are evaluated for every hyperparameter
candidate.  No model is selected unless held-out VAMP-E and the recovered slow
subspaces pass declared stability gates.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .tica import (
    TICAAnalysisError,
    fit_tica,
    project_tica,
    time_lagged_independent_component_analysis_project,
)
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class RandomFeatureKoopmanError(ValueError):
    """Raised when nonlinear kinetic estimation cannot be evaluated safely."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("random_feature_koopman")
        if isinstance(definitions, dict) else None
    )
    required = {
        "source_module", "component_indices", "lag_frames", "component_count",
        "random_feature_counts", "bandwidth_scales", "random_seeds",
        "cross_validation_folds", "covariance_regularization",
        "covariance_eigenvalue_cutoff", "minimum_pairs_per_segment",
        "maximum_bandwidth_observations", "maximum_feature_matrix_elements",
        "maximum_seed_vamp_e_relative_range", "minimum_seed_subspace_similarity",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise RandomFeatureKoopmanError(
            "definitions.random_feature_koopman fields do not match the contract"
        )
    if raw["source_module"] != "time_lagged_independent_component_analysis":
        raise RandomFeatureKoopmanError(
            "source_module must be time_lagged_independent_component_analysis"
        )
    components = raw["component_indices"]
    if (
        not isinstance(components, list) or len(components) < 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in components
        )
        or len(set(components)) != len(components)
    ):
        raise RandomFeatureKoopmanError(
            "component_indices must contain at least two unique positive integers"
        )
    feature_counts = raw["random_feature_counts"]
    if (
        not isinstance(feature_counts, list) or not feature_counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 4
            for value in feature_counts
        )
        or len(set(feature_counts)) != len(feature_counts)
    ):
        raise RandomFeatureKoopmanError(
            "random_feature_counts must contain unique integers of at least four"
        )
    bandwidths = raw["bandwidth_scales"]
    if (
        not isinstance(bandwidths, list) or not bandwidths
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0
            for value in bandwidths
        )
        or len(set(float(value) for value in bandwidths)) != len(bandwidths)
    ):
        raise RandomFeatureKoopmanError(
            "bandwidth_scales must contain unique finite positive numbers"
        )
    seeds = raw["random_seeds"]
    if (
        not isinstance(seeds, list) or len(seeds) < 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise RandomFeatureKoopmanError(
            "random_seeds must contain at least three unique nonnegative integers"
        )
    for name, allow_zero in (
        ("covariance_regularization", True),
        ("covariance_eigenvalue_cutoff", False),
        ("maximum_seed_vamp_e_relative_range", True),
        ("minimum_seed_subspace_similarity", True),
    ):
        value = raw[name]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            relation = "nonnegative" if allow_zero else "positive"
            raise RandomFeatureKoopmanError(f"{name} must be finite and {relation}")
    if float(raw["minimum_seed_subspace_similarity"]) > 1.0:
        raise RandomFeatureKoopmanError(
            "minimum_seed_subspace_similarity cannot exceed one"
        )
    result = dict(raw)
    result["component_indices"] = list(components)
    result["random_feature_counts"] = sorted(int(value) for value in feature_counts)
    result["bandwidth_scales"] = sorted(float(value) for value in bandwidths)
    result["random_seeds"] = [int(value) for value in seeds]
    for name in (
        "lag_frames", "component_count", "cross_validation_folds",
        "minimum_pairs_per_segment", "maximum_bandwidth_observations",
        "maximum_feature_matrix_elements",
    ):
        result[name] = positive_integer(
            raw[name], name, error_type=RandomFeatureKoopmanError
        )
    if int(result["cross_validation_folds"]) < 2:
        raise RandomFeatureKoopmanError(
            "cross_validation_folds must be at least two"
        )
    if int(result["component_count"]) > min(result["random_feature_counts"]):
        raise RandomFeatureKoopmanError(
            "component_count cannot exceed the smallest random feature count"
        )
    result["covariance_regularization"] = float(raw["covariance_regularization"])
    result["covariance_eigenvalue_cutoff"] = float(
        raw["covariance_eigenvalue_cutoff"]
    )
    result["maximum_seed_vamp_e_relative_range"] = float(
        raw["maximum_seed_vamp_e_relative_range"]
    )
    result["minimum_seed_subspace_similarity"] = float(
        raw["minimum_seed_subspace_similarity"]
    )
    return result


def random_fourier_features(
    values: np.ndarray, *, feature_count: int, bandwidth: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an isotropic-Gaussian random Fourier feature map."""

    if values.ndim != 2 or not np.isfinite(values).all() or values.shape[1] == 0:
        raise RandomFeatureKoopmanError("random-feature input must be a finite matrix")
    if feature_count < 1 or not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise RandomFeatureKoopmanError("random-feature dimensions and bandwidth are invalid")
    generator = np.random.default_rng(seed)
    frequencies = generator.normal(
        0.0, 1.0 / bandwidth, size=(feature_count, values.shape[1])
    )
    phases = generator.uniform(0.0, 2.0 * math.pi, size=feature_count)
    mapped = math.sqrt(2.0 / feature_count) * np.cos(
        values @ frequencies.T + phases
    )
    return mapped, frequencies, phases


def _source_segments(
    report: Mapping[str, object], component_indices: Sequence[int],
    lag_frames: int, minimum_pairs: int,
) -> Tuple[list[np.ndarray], list[Dict[str, object]], float, str]:
    raw_segments = report.get("segments")
    if not isinstance(raw_segments, list):
        raise RandomFeatureKoopmanError("TICA report has no segments")
    zero_based = [value - 1 for value in component_indices]
    arrays: list[np.ndarray] = []
    metadata: list[Dict[str, object]] = []
    intervals: list[float] = []
    time_units = set()
    for segment in raw_segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("projections"), list):
            raise RandomFeatureKoopmanError("TICA segment is malformed")
        projections = sorted(
            segment["projections"], key=lambda row: int(row["source_frame_index"])
        )
        if len(projections) - lag_frames < minimum_pairs:
            raise RandomFeatureKoopmanError(
                f"{segment.get('system_id')}/{segment.get('replica_id')}/"
                f"{segment.get('segment_id')} has fewer than minimum_pairs_per_segment"
            )
        rows = []
        times = []
        for projection in projections:
            if not isinstance(projection, dict) or not isinstance(projection.get("scores"), list):
                raise RandomFeatureKoopmanError("TICA projection is malformed")
            scores = projection["scores"]
            if max(zero_based) >= len(scores):
                raise RandomFeatureKoopmanError(
                    "component_indices exceed TICA projection dimensions"
                )
            rows.append([float(scores[index]) for index in zero_based])
            times.append(float(projection["time"]))
            time_units.add(str(projection["time_unit"]))
        array = np.asarray(rows, dtype=float)
        if not np.isfinite(array).all():
            raise RandomFeatureKoopmanError("TICA projection contains non-finite values")
        differences = np.diff(np.asarray(times, dtype=float))
        if differences.size == 0 or np.any(differences <= 0.0):
            raise RandomFeatureKoopmanError("TICA projection times must increase")
        interval = float(differences[0])
        if np.max(np.abs(differences - interval)) > 1.0e-9 * max(1.0, abs(interval)):
            raise RandomFeatureKoopmanError("TICA projection interval must be constant")
        intervals.append(interval)
        arrays.append(array)
        metadata.append({
            "system_id": str(segment["system_id"]),
            "replica_id": str(segment["replica_id"]),
            "segment_id": str(segment["segment_id"]),
            **({"member_id": str(segment["member_id"])} if "member_id" in segment else {}),
            "source_frame_indices": [int(row["source_frame_index"]) for row in projections],
            "times": times,
        })
    if not arrays:
        raise RandomFeatureKoopmanError("TICA report produced no nonlinear input segments")
    if len(time_units) != 1:
        raise RandomFeatureKoopmanError("TICA segments mix physical time units")
    reference = intervals[0]
    if any(abs(value - reference) > 1.0e-9 * max(1.0, abs(reference)) for value in intervals):
        raise RandomFeatureKoopmanError("TICA segments mix evaluated frame intervals")
    return arrays, metadata, reference, next(iter(time_units))


def _deterministic_bandwidth(
    arrays: Sequence[np.ndarray], maximum_observations: int,
) -> Tuple[float, Dict[str, object]]:
    values = np.concatenate(arrays, axis=0)
    if len(values) > maximum_observations:
        indices = np.linspace(
            0, len(values) - 1, maximum_observations, dtype=int
        )
        sampled = values[indices]
        method = "deterministic_evenly_spaced_observation_subset"
    else:
        indices = np.arange(len(values), dtype=int)
        sampled = values
        method = "all_observations"
    distances = []
    for left in range(len(sampled)):
        delta = sampled[left + 1:] - sampled[left]
        if len(delta):
            distances.extend(np.linalg.norm(delta, axis=1).tolist())
    positive = np.asarray([value for value in distances if value > 0.0], dtype=float)
    if positive.size == 0:
        raise RandomFeatureKoopmanError("nonlinear input has no nonzero pairwise distance")
    median = float(np.median(positive))
    return median, {
        "method": method,
        "total_observation_count": len(values),
        "evaluated_observation_count": len(sampled),
        "evaluated_observation_indices": indices.tolist(),
        "positive_pair_distance_count": int(positive.size),
        "median_pairwise_distance": median,
    }


def _inverse_sqrt(
    matrix: np.ndarray, regularization: float, eigenvalue_cutoff: float,
) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    scale = float(np.trace(symmetric)) / max(1, symmetric.shape[0])
    absolute = regularization * max(scale, np.finfo(float).eps)
    values, vectors = np.linalg.eigh(symmetric + absolute * np.eye(len(symmetric)))
    threshold = max(
        float(values.max()) * eigenvalue_cutoff, np.finfo(float).eps
    )
    retained = values > threshold
    if not retained.any():
        raise RandomFeatureKoopmanError("VAMP covariance has zero numerical rank")
    return vectors[:, retained] / np.sqrt(values[retained])


def _covariances(
    left: np.ndarray, right: np.ndarray,
    left_mean: np.ndarray | None = None, right_mean: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if left_mean is None:
        left_mean = left.mean(axis=0)
    if right_mean is None:
        right_mean = right.mean(axis=0)
    l = left - left_mean
    r = right - right_mean
    count = len(l)
    return (
        l.T @ l / count, l.T @ r / count, r.T @ r / count,
        left_mean, right_mean,
    )


def _cross_validated_vamp(
    segments: Sequence[np.ndarray], lag_frames: int, fold_count: int,
    regularization: float, eigenvalue_cutoff: float, component_count: int,
) -> Dict[str, object]:
    pairs = []
    for segment_index, values in enumerate(segments):
        pair_count = len(values) - lag_frames
        for index in range(pair_count):
            fold = min(fold_count - 1, (index * fold_count) // pair_count)
            pairs.append((fold, segment_index, index, values[index], values[index + lag_frames]))
    if len(pairs) < 2 * fold_count:
        return {
            "status": "not_calculable",
            "reason": "fewer than two lag pairs per requested fold",
            "fold_count": fold_count, "transition_pair_count": len(pairs),
        }
    folds = []
    for fold in range(fold_count):
        training = [row for row in pairs if row[0] != fold]
        testing = [row for row in pairs if row[0] == fold]
        if not training or not testing:
            continue
        train_left = np.asarray([row[3] for row in training], dtype=float)
        train_right = np.asarray([row[4] for row in training], dtype=float)
        c00, c01, c11, mean_left, mean_right = _covariances(
            train_left, train_right
        )
        left_whitener = _inverse_sqrt(
            c00, regularization, eigenvalue_cutoff
        )
        right_whitener = _inverse_sqrt(
            c11, regularization, eigenvalue_cutoff
        )
        koopman = left_whitener.T @ c01 @ right_whitener
        left_singular, singular_values, right_singular_t = np.linalg.svd(
            koopman, full_matrices=False
        )
        retained = min(component_count, len(singular_values))
        singular_values = singular_values[:retained]
        left_functions = left_whitener @ left_singular[:, :retained]
        right_functions = right_whitener @ right_singular_t.T[:, :retained]
        test_left = np.asarray([row[3] for row in testing], dtype=float)
        test_right = np.asarray([row[4] for row in testing], dtype=float)
        test_c00, test_c01, test_c11, _, _ = _covariances(
            test_left, test_right, mean_left, mean_right
        )
        singular = np.diag(singular_values)
        cross = left_functions.T @ test_c01 @ right_functions
        left_metric = left_functions.T @ test_c00 @ left_functions
        right_metric = right_functions.T @ test_c11 @ right_functions
        vamp_e = float(np.trace(
            2.0 * singular @ cross
            - singular @ left_metric @ singular @ right_metric
        ))
        vamp_2 = float(np.sum(singular_values ** 2))
        if not math.isfinite(vamp_e) or not math.isfinite(vamp_2):
            raise RandomFeatureKoopmanError("VAMP cross-validation produced a non-finite score")
        folds.append({
            "fold": fold,
            "training_pair_count": len(training),
            "testing_pair_count": len(testing),
            "training_vamp2": vamp_2,
            "heldout_vamp_e": vamp_e,
        })
    if len(folds) != fold_count:
        return {
            "status": "not_calculable", "reason": "one or more folds were empty",
            "fold_count": fold_count, "transition_pair_count": len(pairs),
            "folds": folds,
        }
    values = np.asarray([float(row["heldout_vamp_e"]) for row in folds])
    return {
        "status": "complete", "fold_count": fold_count,
        "transition_pair_count": len(pairs),
        "fold_assignment": "contiguous within-segment time blocks",
        "mean_training_vamp2": float(np.mean([
            float(row["training_vamp2"]) for row in folds
        ])),
        "mean_heldout_vamp_e": float(values.mean()),
        "standard_deviation_heldout_vamp_e": float(values.std(ddof=0)),
        "folds": folds,
    }


def _subspace_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = first - first.mean(axis=0)
    right = second - second.mean(axis=0)
    q_left, _ = np.linalg.qr(left, mode="reduced")
    q_right, _ = np.linalg.qr(right, mode="reduced")
    singular = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return float(singular.min()) if len(singular) else 0.0


def fit_random_feature_koopman(
    source_segments: Sequence[np.ndarray], settings: Mapping[str, object],
) -> Dict[str, object]:
    """Scan RFF hyperparameters and apply multi-seed stability gates."""

    total_observations = sum(len(values) for values in source_segments)
    combined = np.concatenate(source_segments, axis=0)
    mean = combined.mean(axis=0)
    scale = combined.std(axis=0)
    if np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise RandomFeatureKoopmanError("selected TICA inputs contain a zero-variance component")
    standardized = [(values - mean) / scale for values in source_segments]
    base_bandwidth, bandwidth_evidence = _deterministic_bandwidth(
        standardized, int(settings["maximum_bandwidth_observations"])
    )
    candidates = []
    internal_models: Dict[Tuple[int, float, int], Dict[str, object]] = {}
    for feature_count in settings["random_feature_counts"]:  # type: ignore[index]
        if total_observations * int(feature_count) > int(
            settings["maximum_feature_matrix_elements"]
        ):
            candidates.append({
                "random_feature_count": int(feature_count),
                "eligible": False,
                "reason": "maximum_feature_matrix_elements_exceeded",
            })
            continue
        for bandwidth_scale in settings["bandwidth_scales"]:  # type: ignore[index]
            bandwidth = base_bandwidth * float(bandwidth_scale)
            runs = []
            for seed in settings["random_seeds"]:  # type: ignore[index]
                mapped_segments = []
                frequencies = phases = None
                for values in standardized:
                    mapped, current_frequencies, current_phases = random_fourier_features(
                        values, feature_count=int(feature_count),
                        bandwidth=bandwidth, seed=int(seed),
                    )
                    mapped_segments.append(mapped)
                    if frequencies is None:
                        frequencies, phases = current_frequencies, current_phases
                    elif not (
                        np.array_equal(frequencies, current_frequencies)
                        and np.array_equal(phases, current_phases)
                    ):
                        raise RandomFeatureKoopmanError(
                            "one feature-map seed produced inconsistent segment dictionaries"
                        )
                try:
                    model = fit_tica(
                        [values.tolist() for values in mapped_segments],
                        lag_frames=int(settings["lag_frames"]),
                        component_count=int(settings["component_count"]),
                        covariance_regularization=float(
                            settings["covariance_regularization"]
                        ),
                        covariance_eigenvalue_cutoff=float(
                            settings["covariance_eigenvalue_cutoff"]
                        ),
                    )
                    projections = [
                        np.asarray(project_tica(
                            values.tolist(), model["mean"], model["eigenvectors"]
                        ), dtype=float)
                        for values in mapped_segments
                    ]
                    cv = _cross_validated_vamp(
                        mapped_segments, int(settings["lag_frames"]),
                        int(settings["cross_validation_folds"]),
                        float(settings["covariance_regularization"]),
                        float(settings["covariance_eigenvalue_cutoff"]),
                        int(settings["component_count"]),
                    )
                except (
                    TICAAnalysisError, RandomFeatureKoopmanError,
                    np.linalg.LinAlgError,
                ) as exc:
                    runs.append({
                        "random_seed": int(seed),
                        "fit_status": "failed",
                        "failure_reason": str(exc),
                        "mean_heldout_vamp_e": None,
                        "cross_validated_vamp": {
                            "status": "not_calculable",
                            "reason": "feature-map fit or validation failed",
                        },
                    })
                    continue
                run = {
                    "random_seed": int(seed),
                    "fit_status": "complete",
                    "eigenvalues": list(model["eigenvalues"]),
                    "retained_covariance_rank": model["retained_covariance_rank"],
                    "mean_heldout_vamp_e": cv.get("mean_heldout_vamp_e"),
                    "cross_validated_vamp": cv,
                }
                runs.append(run)
                internal_models[(int(feature_count), float(bandwidth_scale), int(seed))] = {
                    "model": model, "projections": projections,
                    "frequencies": frequencies, "phases": phases,
                }
            cv_complete = len(runs) == len(settings["random_seeds"]) and all(
                row.get("fit_status") == "complete"
                and
                row["cross_validated_vamp"].get("status") == "complete"
                for row in runs
            )
            vamp_values = [
                float(row["mean_heldout_vamp_e"]) for row in runs
                if row["mean_heldout_vamp_e"] is not None
            ]
            relative_range = math.inf
            if len(vamp_values) == len(runs):
                denominator = max(abs(float(np.median(vamp_values))), 1.0e-12)
                relative_range = (max(vamp_values) - min(vamp_values)) / denominator
            similarities = []
            if cv_complete:
                for left_seed, right_seed in combinations(settings["random_seeds"], 2):  # type: ignore[arg-type]
                    left = np.concatenate(internal_models[(
                        int(feature_count), float(bandwidth_scale), int(left_seed)
                    )]["projections"], axis=0)
                    right = np.concatenate(internal_models[(
                        int(feature_count), float(bandwidth_scale), int(right_seed)
                    )]["projections"], axis=0)
                    similarities.append({
                        "first_seed": int(left_seed), "second_seed": int(right_seed),
                        "minimum_canonical_subspace_similarity": _subspace_similarity(
                            left, right
                        ),
                    })
            minimum_similarity = min(
                float(row["minimum_canonical_subspace_similarity"])
                for row in similarities
            ) if similarities else 1.0
            passes = (
                cv_complete
                and relative_range <= float(
                    settings["maximum_seed_vamp_e_relative_range"]
                )
                and minimum_similarity >= float(
                    settings["minimum_seed_subspace_similarity"]
                )
            )
            candidates.append({
                "random_feature_count": int(feature_count),
                "bandwidth_scale": float(bandwidth_scale),
                "bandwidth": bandwidth,
                "eligible": passes,
                "candidate_fit_status": (
                    "complete" if cv_complete else "incomplete_seed_runs"
                ),
                "mean_seed_heldout_vamp_e": (
                    float(np.mean(vamp_values)) if vamp_values else None
                ),
                "seed_heldout_vamp_e_relative_range": relative_range,
                "minimum_seed_subspace_similarity": minimum_similarity,
                "seed_stability_gate": {
                    "status": "passed" if passes else "failed",
                    "configured_random_seeds": list(settings["random_seeds"]),
                    "maximum_vamp_e_relative_range": settings[
                        "maximum_seed_vamp_e_relative_range"
                    ],
                    "minimum_subspace_similarity": settings[
                        "minimum_seed_subspace_similarity"
                    ],
                    "observed_vamp_e_relative_range": relative_range,
                    "observed_minimum_subspace_similarity": minimum_similarity,
                    "pairwise_subspace_similarities": similarities,
                },
                "seed_runs": runs,
            })
    eligible = [row for row in candidates if row.get("eligible") is True]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_seed_heldout_vamp_e"]),
            -int(row["random_feature_count"]),
            -abs(float(row["bandwidth_scale"]) - 1.0),
        ),
    ) if eligible else None
    result: Dict[str, object] = {
        "input_standardization": {
            "mean": mean.tolist(), "population_standard_deviation": scale.tolist(),
        },
        "base_bandwidth": base_bandwidth,
        "bandwidth_evidence": bandwidth_evidence,
        "hyperparameter_candidates": candidates,
        "eligible_candidate_count": len(eligible),
        "selection_status": "selected_stable_candidate" if selected else "no_stable_candidate",
        "configured_random_seeds": list(settings["random_seeds"]),
    }
    if selected is None:
        return result
    primary_seed = int(settings["random_seeds"][0])  # type: ignore[index]
    key = (
        int(selected["random_feature_count"]),
        float(selected["bandwidth_scale"]), primary_seed,
    )
    primary = internal_models[key]
    model = primary["model"]
    result.update({
        "selected_hyperparameters": {
            "random_feature_count": selected["random_feature_count"],
            "bandwidth_scale": selected["bandwidth_scale"],
            "bandwidth": selected["bandwidth"],
            "primary_prespecified_seed": primary_seed,
            "selection_score": selected["mean_seed_heldout_vamp_e"],
            "selection_score_definition": "mean held-out VAMP-E across prespecified seeds",
        },
        "selected_seed_stability_gate": selected["seed_stability_gate"],
        "selected_model": model,
        "selected_primary_projections": [
            values.tolist() for values in primary["projections"]
        ],
        "random_feature_dictionary": {
            "frequencies": primary["frequencies"].tolist(),
            "phases": primary["phases"].tolist(),
            "normalization": math.sqrt(
                2.0 / int(selected["random_feature_count"])
            ),
            "kernel": "isotropic_gaussian_rbf",
        },
    })
    return result


def random_feature_koopman_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    tica_report = load_cached_project_report(
        "time_lagged_independent_component_analysis", source,
        hash_content=hash_content, error_type=RandomFeatureKoopmanError,
    )
    if tica_report is None:
        tica_report = time_lagged_independent_component_analysis_project(
            source, hash_content=hash_content
        )
    arrays, metadata, interval, time_unit = _source_segments(
        tica_report, settings["component_indices"],  # type: ignore[arg-type]
        int(settings["lag_frames"]), int(settings["minimum_pairs_per_segment"]),
    )
    fitted = fit_random_feature_koopman(arrays, settings)
    physical_frame_identities = {
        (
            str(segment["system_id"]), str(segment["replica_id"]),
            str(segment["segment_id"]), int(source_frame_index),
        )
        for segment in metadata
        for source_frame_index in segment["source_frame_indices"]
    }
    issues = [issue for issue in tica_report.get("issues", []) if isinstance(issue, dict)]
    segments = []
    components = []
    availability = "available"
    if fitted["selection_status"] == "selected_stable_candidate":
        projections = fitted.pop("selected_primary_projections")
        model = fitted.pop("selected_model")
        lag_time = interval * int(settings["lag_frames"])
        for index, (value, residual) in enumerate(zip(
            model["eigenvalues"], model["generalized_eigen_residual_norms"]
        ), start=1):
            magnitude = abs(float(value))
            timescale = (
                -lag_time / math.log(magnitude)
                if 0.0 < magnitude < 1.0 else None
            )
            components.append({
                "component_index": index, "eigenvalue": float(value),
                "implied_timescale": timescale, "time_unit": time_unit,
                "generalized_eigen_residual_norm": float(residual),
            })
        for segment, projection_rows in zip(metadata, projections):
            segments.append({
                "system_id": segment["system_id"],
                "replica_id": segment["replica_id"],
                "segment_id": segment["segment_id"],
                **({"member_id": segment["member_id"]} if "member_id" in segment else {}),
                "projections": [{
                    "source_frame_index": frame_index,
                    "time": time, "time_unit": time_unit, "scores": scores,
                } for frame_index, time, scores in zip(
                    segment["source_frame_indices"], segment["times"], projection_rows
                )],
            })
    else:
        availability = "not_available"
        issues.append({
            "severity": "warning", "code": "RANDOM_FEATURE_KOOPMAN_SEED_GATE_FAILED",
            "message": "No hyperparameter candidate passed both prespecified seed-stability gates.",
        })
    return {
        "module_id": "random_feature_koopman",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "availability_status": availability,
        "availability_reason": (
            None if availability == "available" else "no_stable_hyperparameter_candidate"
        ),
        "project_manifest_path": str(source),
        "project_manifest_sha256": tica_report["project_manifest_sha256"],
        "system_manifest_path": tica_report["system_manifest_path"],
        "system_manifest_sha256": tica_report["system_manifest_sha256"],
        "input_content_signature_sha256": tica_report.get(
            "input_content_signature_sha256"
        ),
        "content_hashes_included": hash_content,
        "settings": settings,
        "feature_lineage": {
            "module_id": "time_lagged_independent_component_analysis",
            "component_indices": settings["component_indices"],
        },
        "lag_time": interval * int(settings["lag_frames"]),
        "time_unit": time_unit,
        "pair_count": sum(len(values) - int(settings["lag_frames"]) for values in arrays),
        "observation_accounting": {
            "source_physical_frame_count": len(physical_frame_identities),
            "symmetry_expanded_observation_count": sum(
                len(values) for values in arrays
            ),
            "kinetic_trajectory_count": len(arrays),
            "member_observations_are_independent_replicas": False,
        },
        "components": components, "segments": segments,
        **fitted,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Random Fourier features approximate an isotropic Gaussian kernel and do not guarantee that the chosen kernel is physically appropriate.",
            "Every candidate must pass held-out VAMP-E and slow-subspace stability gates across prespecified feature-map seeds.",
            "The first prespecified seed supplies report projections only after the seed gate passes; seeds are not searched for the most favorable model.",
            "Hyperparameter selection reuses the same contiguous validation folds and therefore remains a sensitivity analysis rather than independent confirmation.",
            "Lag pairs never cross system, replica, segment, or member boundaries.",
            "Nonlinear coordinates and implied timescales do not establish metastability, convergence, Markovianity, mechanism, or causality.",
        ],
    }


def random_feature_koopman_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return random_feature_koopman_project(project_path, hash_content=hash_content)
    except (
        RandomFeatureKoopmanError, TICAAnalysisError,
        ManifestValidationError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "random_feature_koopman",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "RANDOM_FEATURE_KOOPMAN_INVALID",
                "message": message,
            } for message in messages],
        }
