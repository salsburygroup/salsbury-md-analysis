"""Segment-safe discrete-state transition-model diagnostics.

This module deliberately treats Markov modeling as a validation exercise.  It
never joins transitions across segment boundaries and never promotes a fitted
model to a kinetic interpretation merely because the numerical estimator ran.
"""

from __future__ import annotations

import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .alternative_clustering import (
    AlternativeClusteringError,
    alternative_clustering_project,
)

from .clustering import (
    ClusteringAnalysisError,
    clustering_hdbscan_project,
    clustering_imwkmeans_project,
    clustering_kmeans_project,
)
from .manifests import ManifestValidationError, load_json
from .pca import PCAAnalysisError
from .pca_fes import PCAFESAnalysisError, pca_fes_basins_project
from .validation import positive_integer


class MSMAnalysisError(ValueError):
    """Raised when a discrete-state or kinetics contract fails closed."""


_positive_integer = partial(positive_integer, error_type=MSMAnalysisError)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise MSMAnalysisError("project definitions.markov_state_models is required")
    raw = definitions.get("markov_state_models")
    if not isinstance(raw, dict):
        raise MSMAnalysisError("definitions.markov_state_models must be an object")
    required = {
        "lag_frames", "estimators", "minimum_transition_count",
        "maximum_states", "ck_multiples", "maximum_ck_rmse",
    }
    missing = sorted(required.difference(raw))
    optional = {
        "assignment_source", "assignment_sources", "vamp_cross_validation_folds",
        "vamp_regularization", "maximum_implied_timescale_relative_range",
        "bootstrap_repeats", "bootstrap_block_length_frames",
        "bootstrap_confidence_level", "random_seed",
    }
    unknown = sorted(set(raw).difference(required | optional))
    if missing:
        raise MSMAnalysisError(
            "definitions.markov_state_models is missing required fields: " + ", ".join(missing)
        )
    if unknown:
        raise MSMAnalysisError(
            "definitions.markov_state_models contains unknown fields: " + ", ".join(unknown)
        )
    has_single = "assignment_source" in raw
    has_multiple = "assignment_sources" in raw
    if has_single == has_multiple:
        raise MSMAnalysisError(
            "declare exactly one of assignment_source or assignment_sources"
        )
    if has_single and raw["assignment_source"] not in {
        "clustering_kmeans", "clustering_imwkmeans"
    }:
        raise MSMAnalysisError(
            "assignment_source must be clustering_kmeans or clustering_imwkmeans"
        )
    assignment_sources = raw.get("assignment_sources")
    if has_multiple and (
        not isinstance(assignment_sources, list)
        or len(assignment_sources) != 2
        or any(not isinstance(value, str) for value in assignment_sources)
        or set(assignment_sources) != {"best_clustering", "pca_fes_basins"}
    ):
        raise MSMAnalysisError(
            "assignment_sources must contain best_clustering and pca_fes_basins"
        )
    lag_frames = raw["lag_frames"]
    if (
        not isinstance(lag_frames, list) or not lag_frames
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in lag_frames)
        or len(set(lag_frames)) != len(lag_frames)
    ):
        raise MSMAnalysisError("lag_frames must contain unique positive integers")
    estimators = raw["estimators"]
    allowed_estimators = {"reversible_symmetrized", "nonreversible_mle"}
    if (
        not isinstance(estimators, list) or not estimators
        or any(value not in allowed_estimators for value in estimators)
        or len(set(estimators)) != len(estimators)
    ):
        raise MSMAnalysisError(
            "estimators must contain unique reversible_symmetrized and/or nonreversible_mle values"
        )
    multiples = raw["ck_multiples"]
    if (
        not isinstance(multiples, list) or not multiples
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in multiples)
        or len(set(multiples)) != len(multiples)
    ):
        raise MSMAnalysisError("ck_multiples must contain unique integers of at least 2")
    maximum_ck_rmse = raw["maximum_ck_rmse"]
    if (
        isinstance(maximum_ck_rmse, bool) or not isinstance(maximum_ck_rmse, (int, float))
        or not math.isfinite(float(maximum_ck_rmse)) or float(maximum_ck_rmse) <= 0.0
    ):
        raise MSMAnalysisError("maximum_ck_rmse must be finite and positive")
    folds = raw.get("vamp_cross_validation_folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise MSMAnalysisError(
            "vamp_cross_validation_folds must be an integer of at least 2"
        )
    vamp_regularization = raw.get("vamp_regularization", 1.0e-8)
    if (
        isinstance(vamp_regularization, bool)
        or not isinstance(vamp_regularization, (int, float))
        or not math.isfinite(float(vamp_regularization))
        or float(vamp_regularization) <= 0.0
    ):
        raise MSMAnalysisError("vamp_regularization must be finite and positive")
    maximum_its_range = raw.get("maximum_implied_timescale_relative_range", 0.5)
    if (
        isinstance(maximum_its_range, bool)
        or not isinstance(maximum_its_range, (int, float))
        or not math.isfinite(float(maximum_its_range))
        or float(maximum_its_range) <= 0.0
    ):
        raise MSMAnalysisError(
            "maximum_implied_timescale_relative_range must be finite and positive"
        )
    bootstrap_repeats = raw.get("bootstrap_repeats", 0)
    if (
        isinstance(bootstrap_repeats, bool)
        or not isinstance(bootstrap_repeats, int)
        or bootstrap_repeats < 0
    ):
        raise MSMAnalysisError("bootstrap_repeats must be a nonnegative integer")
    bootstrap_block_length = raw.get("bootstrap_block_length_frames", 100)
    if (
        isinstance(bootstrap_block_length, bool)
        or not isinstance(bootstrap_block_length, int)
        or bootstrap_block_length < 2
    ):
        raise MSMAnalysisError(
            "bootstrap_block_length_frames must be an integer of at least 2"
        )
    confidence = raw.get("bootstrap_confidence_level", 0.95)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise MSMAnalysisError(
            "bootstrap_confidence_level must be strictly between zero and one"
        )
    random_seed = raw.get("random_seed", 0)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise MSMAnalysisError("random_seed must be a nonnegative integer")
    return {
        **(
            {"assignment_source": raw["assignment_source"]}
            if has_single else {"assignment_sources": list(assignment_sources)}
        ),
        "lag_frames": sorted(lag_frames),
        "estimators": list(estimators),
        "minimum_transition_count": _positive_integer(
            raw["minimum_transition_count"], "minimum_transition_count"
        ),
        "maximum_states": _positive_integer(raw["maximum_states"], "maximum_states"),
        "ck_multiples": sorted(multiples),
        "maximum_ck_rmse": float(maximum_ck_rmse),
        "vamp_cross_validation_folds": folds,
        "vamp_regularization": float(vamp_regularization),
        "maximum_implied_timescale_relative_range": float(maximum_its_range),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_block_length_frames": bootstrap_block_length,
        "bootstrap_confidence_level": float(confidence),
        "random_seed": random_seed,
    }


def _group_assignments(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[List[List[int]], float, str, int]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = {}
    states = set()
    intervals = []
    time_unit: Optional[str] = None
    for row in rows:
        try:
            key = (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]),
                *((str(row["member_id"]),) if "member_id" in row else ()),
            )
            frame = int(row["source_frame_index"])
            state = int(row["cluster_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MSMAnalysisError("clustering assignments lack segment/frame/state identities") from exc
        if state <= 0:
            raise MSMAnalysisError("complete assignments require positive cluster_id values")
        if "time" not in row or "time_unit" not in row:
            raise MSMAnalysisError(
                "Markov-state kinetics require physical time on every assignment; sample-index ensembles are unsupported"
            )
        row_unit = str(row["time_unit"])
        if time_unit is None:
            time_unit = row_unit
        elif row_unit != time_unit:
            raise MSMAnalysisError("all assignments must use one physical time unit")
        groups.setdefault(key, []).append({**row, "source_frame_index": frame, "cluster_id": state})
        states.add(state)
    if not rows:
        raise MSMAnalysisError("assignment source returned no observations")
    expected_states = set(range(1, max(states) + 1))
    if states != expected_states:
        raise MSMAnalysisError("cluster IDs must be complete and contiguous from 1")
    discrete = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: int(row["source_frame_index"]))
        frame_ids = [int(row["source_frame_index"]) for row in ordered]
        if len(frame_ids) != len(set(frame_ids)):
            raise MSMAnalysisError(f"duplicate source frame in segment {'/'.join(key)}")
        times = [float(row["time"]) for row in ordered]
        if any(not math.isfinite(value) for value in times):
            raise MSMAnalysisError("assignment times must be finite")
        for left, right in zip(times, times[1:]):
            difference = right - left
            if difference <= 0.0:
                raise MSMAnalysisError(f"non-increasing time in segment {'/'.join(key)}")
            intervals.append(difference)
        discrete.append([int(row["cluster_id"]) - 1 for row in ordered])
    if not intervals:
        raise MSMAnalysisError("at least one segment must contain two physically timed observations")
    interval = intervals[0]
    if any(abs(value - interval) > max(1.0e-9, abs(interval) * 1.0e-9) for value in intervals[1:]):
        raise MSMAnalysisError("all assignment intervals must match for frame-lag kinetics")
    return discrete, interval, time_unit or "unknown", len(states)


def _count_transitions(
    trajectories: Sequence[Sequence[int]], state_count: int, lag: int
) -> Tuple[List[List[int]], int]:
    counts = [[0 for _ in range(state_count)] for _ in range(state_count)]
    total = 0
    for trajectory in trajectories:
        for index in range(0, max(0, len(trajectory) - lag)):
            counts[trajectory[index]][trajectory[index + lag]] += 1
            total += 1
    return counts, total


def _connected_components(counts: Sequence[Sequence[int]]) -> List[List[int]]:
    state_count = len(counts)
    unseen = set(range(state_count))
    components = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component = []
        unseen.remove(root)
        while stack:
            state = stack.pop()
            component.append(state)
            neighbors = {
                other for other in range(state_count)
                if counts[state][other] > 0 or counts[other][state] > 0
            }
            for other in sorted(neighbors & unseen, reverse=True):
                unseen.remove(other)
                stack.append(other)
        components.append(sorted(state + 1 for state in component))
    return sorted(components, key=lambda values: (values[0], len(values)))


def _normalize_rows(values: Sequence[Sequence[float]]) -> List[List[float]]:
    matrix = []
    for index, row in enumerate(values):
        total = sum(row)
        if total <= 0.0:
            raise MSMAnalysisError(f"state {index + 1} has no outgoing transition counts")
        matrix.append([value / total for value in row])
    return matrix


def _stationary_power(matrix: Sequence[Sequence[float]]) -> Tuple[List[float], bool, int]:
    count = len(matrix)
    values = [1.0 / count for _ in range(count)]
    for iteration in range(1, 100001):
        updated = [
            sum(values[left] * matrix[left][right] for left in range(count))
            for right in range(count)
        ]
        total = sum(updated)
        if total <= 0.0:
            return values, False, iteration
        updated = [value / total for value in updated]
        if max(abs(left - right) for left, right in zip(values, updated)) <= 1.0e-13:
            return updated, True, iteration
        values = updated
    return values, False, 100000


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> List[float]:
    """Return all eigenvalues of a real symmetric matrix using Jacobi sweeps."""

    values = [list(row) for row in matrix]
    count = len(values)
    if count == 1:
        return [values[0][0]]
    for _ in range(max(100, 100 * count * count)):
        left, right = max(
            ((i, j) for i in range(count) for j in range(i + 1, count)),
            key=lambda pair: abs(values[pair[0]][pair[1]]),
        )
        off = values[left][right]
        if abs(off) <= 1.0e-14:
            break
        angle = 0.5 * math.atan2(
            2.0 * off, values[right][right] - values[left][left]
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        old_left = values[left][left]
        old_right = values[right][right]
        values[left][left] = cosine * cosine * old_left - 2.0 * sine * cosine * off + sine * sine * old_right
        values[right][right] = sine * sine * old_left + 2.0 * sine * cosine * off + cosine * cosine * old_right
        values[left][right] = values[right][left] = 0.0
        for index in range(count):
            if index in {left, right}:
                continue
            old_i_left = values[index][left]
            old_i_right = values[index][right]
            values[index][left] = values[left][index] = cosine * old_i_left - sine * old_i_right
            values[index][right] = values[right][index] = sine * old_i_left + cosine * old_i_right
    return sorted((values[index][index] for index in range(count)), reverse=True)


def _matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> List[List[float]]:
    count = len(left)
    return [[
        sum(left[i][k] * right[k][j] for k in range(count))
        for j in range(count)
    ] for i in range(count)]


def _matrix_power(matrix: Sequence[Sequence[float]], exponent: int) -> List[List[float]]:
    count = len(matrix)
    result = [[float(i == j) for j in range(count)] for i in range(count)]
    base = [list(row) for row in matrix]
    remaining = exponent
    while remaining:
        if remaining % 2:
            result = _matmul(result, base)
        base = _matmul(base, base)
        remaining //= 2
    return result


def _transition_pairs(
    trajectories: Sequence[Sequence[int]], lag_frames: int,
) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for index in range(max(0, len(trajectory) - lag_frames)):
            pairs.append((trajectory_index, trajectory[index], trajectory[index + lag_frames]))
    return pairs


def _vamp_covariances(
    pairs: Sequence[Tuple[int, int, int]], state_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pairs:
        raise MSMAnalysisError("VAMP scoring requires at least one transition pair")
    c00 = np.zeros((state_count, state_count), dtype=float)
    c01 = np.zeros((state_count, state_count), dtype=float)
    c11 = np.zeros((state_count, state_count), dtype=float)
    for _, left, right in pairs:
        c00[left, left] += 1.0
        c01[left, right] += 1.0
        c11[right, right] += 1.0
    scale = 1.0 / len(pairs)
    return c00 * scale, c01 * scale, c11 * scale


def _inverse_sqrt(matrix: np.ndarray, regularization: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    cutoff = regularization * max(
        1.0, float(np.max(values)) if values.size else 0.0
    )
    inverse = np.asarray([
        1.0 / math.sqrt(float(value)) if value > cutoff else 0.0
        for value in values
    ])
    return (vectors * inverse) @ vectors.T


def cross_validated_vamp_report(
    trajectories: Sequence[Sequence[int]],
    state_count: int,
    lag_frames: int,
    fold_count: int,
    regularization: float,
) -> Dict[str, object]:
    """Calculate time-blocked VAMP-2 training and held-out VAMP-E scores."""

    all_pairs: List[Tuple[int, int, int, int]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        count = max(0, len(trajectory) - lag_frames)
        for index in range(count):
            fold = min(fold_count - 1, (index * fold_count) // max(1, count))
            all_pairs.append(
                (fold, trajectory_index, trajectory[index], trajectory[index + lag_frames])
            )
    if len(all_pairs) < fold_count * 2:
        return {
            "status": "not_calculable",
            "reason": "fewer than two transition pairs per requested fold",
            "lag_frames": lag_frames,
            "fold_count": fold_count,
            "transition_pair_count": len(all_pairs),
        }
    folds = []
    for fold in range(fold_count):
        training = [row[1:] for row in all_pairs if row[0] != fold]
        testing = [row[1:] for row in all_pairs if row[0] == fold]
        if not training or not testing:
            continue
        c00, c01, c11 = _vamp_covariances(training, state_count)
        left_whitener = _inverse_sqrt(c00, regularization)
        right_whitener = _inverse_sqrt(c11, regularization)
        koopman = left_whitener @ c01 @ right_whitener
        left_singular, singular_values, right_singular_t = np.linalg.svd(
            koopman, full_matrices=False
        )
        left_functions = left_whitener @ left_singular
        right_functions = right_whitener @ right_singular_t.T
        test_c00, test_c01, test_c11 = _vamp_covariances(testing, state_count)
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
            raise MSMAnalysisError("VAMP cross-validation produced a non-finite score")
        folds.append({
            "fold": fold,
            "training_transition_pair_count": len(training),
            "testing_transition_pair_count": len(testing),
            "training_vamp2": vamp_2,
            "heldout_vamp_e": vamp_e,
        })
    if len(folds) != fold_count:
        return {
            "status": "not_calculable",
            "reason": "one or more time-blocked folds were empty",
            "lag_frames": lag_frames,
            "fold_count": fold_count,
            "transition_pair_count": len(all_pairs),
            "folds": folds,
        }
    vamp_e_values = [float(row["heldout_vamp_e"]) for row in folds]
    vamp_2_values = [float(row["training_vamp2"]) for row in folds]
    return {
        "status": "complete",
        "lag_frames": lag_frames,
        "fold_count": fold_count,
        "transition_pair_count": len(all_pairs),
        "fold_assignment": (
            "contiguous time blocks within every trajectory segment; folds are pooled "
            "across replicas without leaving out an entire replica"
        ),
        "mean_training_vamp2": float(np.mean(vamp_2_values)),
        "mean_heldout_vamp_e": float(np.mean(vamp_e_values)),
        "standard_deviation_heldout_vamp_e": float(np.std(vamp_e_values)),
        "folds": folds,
    }


def _implied_timescale_stability(
    models: Sequence[Mapping[str, object]], maximum_relative_range: float,
) -> Dict[str, object]:
    reversible = [
        model for model in models
        if model.get("estimator") == "reversible_symmetrized"
        and model.get("estimable") is True
    ]
    modes = sorted({
        int(row["mode"])
        for model in reversible
        for row in model.get("implied_timescales", [])  # type: ignore[union-attr]
        if isinstance(row, dict) and row.get("implied_timescale") is not None
    })
    rows = []
    for mode in modes:
        values = [
            {
                "lag_frames": int(model["lag_frames"]),
                "implied_timescale": float(row["implied_timescale"]),
            }
            for model in reversible
            for row in model.get("implied_timescales", [])  # type: ignore[union-attr]
            if isinstance(row, dict)
            and int(row.get("mode", -1)) == mode
            and row.get("implied_timescale") is not None
        ]
        numeric = [row["implied_timescale"] for row in values]
        if len(numeric) < 2:
            continue
        median = float(np.median(numeric))
        relative_range = (
            (max(numeric) - min(numeric)) / median if median > 0.0 else math.inf
        )
        rows.append({
            "mode": mode,
            "lag_values": values,
            "relative_range": relative_range,
            "passes_declared_gate": relative_range <= maximum_relative_range,
        })
    return {
        "status": "complete" if rows else "not_calculable",
        "maximum_relative_range": maximum_relative_range,
        "modes": rows,
        "passes_declared_gate": bool(rows) and all(
            bool(row["passes_declared_gate"]) for row in rows
        ),
    }


def _time_block_bootstrap(
    trajectories: Sequence[Sequence[int]],
    state_count: int,
    lag_frames: int,
    interval: float,
    time_unit: str,
    repeats: int,
    block_length: int,
    confidence: float,
    random_seed: int,
) -> Dict[str, object]:
    if repeats == 0:
        return {"status": "disabled", "repeat_count": 0}
    blocks = [
        list(trajectory[start:start + block_length])
        for trajectory in trajectories
        for start in range(0, len(trajectory), block_length)
        if len(trajectory[start:start + block_length]) > lag_frames
    ]
    if len(blocks) < 2:
        return {
            "status": "not_calculable",
            "reason": "fewer than two trajectory-preserving time blocks",
            "repeat_count": repeats,
        }
    generator = random.Random(random_seed)
    stationary_samples: List[List[float]] = []
    timescale_samples: List[List[float | None]] = []
    for _ in range(repeats):
        sampled = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        try:
            model = estimate_transition_model(
                sampled, state_count, lag_frames, "reversible_symmetrized",
                interval, time_unit,
            )
        except MSMAnalysisError:
            continue
        stationary_samples.append([
            float(value) for value in model["stationary_distribution"]
        ])
        timescale_samples.append([
            (
                float(row["implied_timescale"])
                if row["implied_timescale"] is not None else None
            )
            for row in model["implied_timescales"]
        ])
    if not stationary_samples:
        return {
            "status": "not_calculable",
            "reason": "no bootstrap resample produced an estimable model",
            "repeat_count": repeats,
        }
    lower = (1.0 - confidence) / 2.0
    upper = 1.0 - lower
    stationary_intervals = []
    for state in range(state_count):
        values = [sample[state] for sample in stationary_samples]
        stationary_intervals.append({
            "state_id": state + 1,
            "median": float(np.median(values)),
            "lower": float(np.quantile(values, lower)),
            "upper": float(np.quantile(values, upper)),
        })
    timescale_intervals = []
    for mode in range(max(0, state_count - 1)):
        values = [
            float(sample[mode]) for sample in timescale_samples
            if mode < len(sample) and sample[mode] is not None
        ]
        if values:
            timescale_intervals.append({
                "mode": mode + 2,
                "sample_count": len(values),
                "median": float(np.median(values)),
                "lower": float(np.quantile(values, lower)),
                "upper": float(np.quantile(values, upper)),
            })
    return {
        "status": "complete",
        "requested_repeat_count": repeats,
        "successful_repeat_count": len(stationary_samples),
        "block_length_frames": block_length,
        "confidence_level": confidence,
        "lag_frames": lag_frames,
        "resampling_unit": (
            "contiguous within-segment time blocks; blocks are never joined when "
            "transition counts are calculated"
        ),
        "stationary_distribution_intervals": stationary_intervals,
        "implied_timescale_intervals": timescale_intervals,
    }


def estimate_transition_model(
    trajectories: Sequence[Sequence[int]],
    state_count: int,
    lag_frames: int,
    estimator: str,
    interval: float,
    time_unit: str,
) -> Dict[str, object]:
    """Estimate one declared model without crossing trajectory boundaries."""

    counts, transition_count = _count_transitions(trajectories, state_count, lag_frames)
    components = _connected_components(counts)
    if estimator == "reversible_symmetrized":
        fitted_counts = [[
            float(counts[i][j] + counts[j][i])
            for j in range(state_count)
        ] for i in range(state_count)]
        transition = _normalize_rows(fitted_counts)
        row_sums = [sum(row) for row in fitted_counts]
        total = sum(row_sums)
        stationary = [value / total for value in row_sums]
        symmetric = [[
            fitted_counts[i][j] / math.sqrt(row_sums[i] * row_sums[j])
            for j in range(state_count)
        ] for i in range(state_count)]
        eigenvalues = _symmetric_eigenvalues(symmetric)
        timescales = []
        lag_time = lag_frames * interval
        for mode, eigenvalue in enumerate(eigenvalues[1:], start=2):
            value = min(1.0, max(-1.0, eigenvalue))
            timescale = None
            if 0.0 < value < 1.0 - 1.0e-12:
                timescale = -lag_time / math.log(value)
            timescales.append({
                "mode": mode,
                "eigenvalue": value,
                "implied_timescale": timescale,
                "time_unit": time_unit,
            })
        stationary_converged = True
        stationary_iterations = 0
    elif estimator == "nonreversible_mle":
        transition = _normalize_rows([[float(value) for value in row] for row in counts])
        stationary, stationary_converged, stationary_iterations = _stationary_power(transition)
        eigenvalues = None
        timescales = None
    else:
        raise MSMAnalysisError(f"unsupported estimator {estimator}")
    return {
        "estimator": estimator,
        "lag_frames": lag_frames,
        "lag_time": lag_frames * interval,
        "time_unit": time_unit,
        "transition_count": transition_count,
        "count_matrix": counts,
        "transition_matrix": transition,
        "stationary_distribution": stationary,
        "stationary_iteration_converged": stationary_converged,
        "stationary_iteration_count": stationary_iterations,
        "undirected_connected_components": components,
        "connected": len(components) == 1,
        "eigenvalues": eigenvalues,
        "implied_timescales": timescales,
    }


def _evaluate_state_definition(
    rows: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
    *,
    candidate_id: str,
    family: str,
    geometric_score: float | None,
    geometric_coverage: float,
) -> Dict[str, object]:
    trajectories, interval, time_unit, state_count = _group_assignments(rows)
    if state_count > int(settings["maximum_states"]):
        raise MSMAnalysisError(
            f"{candidate_id} state count exceeds maximum_states gate"
        )
    models = []
    issues = []
    for estimator in settings["estimators"]:  # type: ignore[union-attr]
        for lag in settings["lag_frames"]:  # type: ignore[union-attr]
            try:
                model = estimate_transition_model(
                    trajectories, state_count, int(lag), str(estimator),
                    interval, time_unit,
                )
            except MSMAnalysisError as exc:
                model = {
                    "estimator": str(estimator),
                    "lag_frames": int(lag),
                    "lag_time": int(lag) * interval,
                    "time_unit": time_unit,
                    "estimable": False,
                    "failure": str(exc),
                    "transition_count": sum(
                        max(0, len(trajectory) - int(lag))
                        for trajectory in trajectories
                    ),
                    "connected": False,
                    "stationary_iteration_converged": False,
                    "passes_minimum_transition_count": False,
                }
                issues.append({
                    "severity": "warning",
                    "code": "MSM_MODEL_NOT_ESTIMABLE",
                    "location": f"{candidate_id}/{estimator}/lag-{lag}",
                    "message": str(exc),
                })
                models.append(model)
                continue
            model["estimable"] = True
            model["failure"] = None
            model["passes_minimum_transition_count"] = (
                model["transition_count"]
                >= int(settings["minimum_transition_count"])
            )
            models.append(model)
    by_key = {
        (model["estimator"], model["lag_frames"]): model
        for model in models if model.get("estimable")
    }
    ck_tests = []
    for model in models:
        if not model.get("estimable"):
            continue
        for multiple in settings["ck_multiples"]:  # type: ignore[union-attr]
            direct_lag = int(model["lag_frames"]) * int(multiple)
            direct = by_key.get((model["estimator"], direct_lag))
            if direct is None:
                continue
            predicted = _matrix_power(model["transition_matrix"], int(multiple))
            observed = direct["transition_matrix"]
            squared = [
                (predicted[i][j] - observed[i][j]) ** 2
                for i in range(state_count) for j in range(state_count)
            ]
            rmse = math.sqrt(sum(squared) / len(squared))
            ck_tests.append({
                "estimator": model["estimator"],
                "base_lag_frames": model["lag_frames"],
                "multiple": multiple,
                "direct_lag_frames": direct_lag,
                "matrix_rmse": rmse,
                "passes_declared_gate": (
                    rmse <= float(settings["maximum_ck_rmse"])
                ),
            })
    implied_stability = _implied_timescale_stability(
        models, float(settings["maximum_implied_timescale_relative_range"])
    )
    vamp = [
        cross_validated_vamp_report(
            trajectories, state_count, int(lag),
            int(settings["vamp_cross_validation_folds"]),
            float(settings["vamp_regularization"]),
        )
        for lag in settings["lag_frames"]  # type: ignore[union-attr]
    ]
    base_lags_passing_ck = [
        int(lag) for lag in settings["lag_frames"]  # type: ignore[union-attr]
        if any(int(test["base_lag_frames"]) == int(lag) for test in ck_tests)
        and all(
            bool(test["passes_declared_gate"])
            for test in ck_tests
            if int(test["base_lag_frames"]) == int(lag)
        )
    ]
    reference_lag = min(base_lags_passing_ck) if base_lags_passing_ck else max(
        int(value) for value in settings["lag_frames"]  # type: ignore[union-attr]
    )
    bootstrap = _time_block_bootstrap(
        trajectories, state_count, reference_lag, interval, time_unit,
        int(settings["bootstrap_repeats"]),
        int(settings["bootstrap_block_length_frames"]),
        float(settings["bootstrap_confidence_level"]),
        int(settings["random_seed"]),
    )
    model_gate = bool(models) and all(
        model.get("estimable") is True
        and model.get("connected") is True
        and model.get("passes_minimum_transition_count") is True
        and model.get("stationary_iteration_converged") is True
        for model in models
    )
    ck_gate = bool(ck_tests) and all(
        bool(test["passes_declared_gate"]) for test in ck_tests
    )
    vamp_gate = bool(vamp) and all(row.get("status") == "complete" for row in vamp)
    implied_gate = implied_stability.get("passes_declared_gate") is True
    validation_passed = model_gate and ck_gate and vamp_gate and implied_gate
    physical_frames = {
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        for row in rows
    }
    return {
        "candidate_id": candidate_id,
        "family": family,
        "geometric_score": geometric_score,
        "geometric_coverage_fraction": geometric_coverage,
        "kinetic_validation_status": "passed" if validation_passed else "not passed",
        "validation_gates": {
            "models_estimable_connected_and_counted": model_gate,
            "chapman_kolmogorov": ck_gate,
            "implied_timescale_stability": implied_gate,
            "cross_validated_vamp": vamp_gate,
        },
        "state_count": state_count,
        "observation_count": sum(len(values) for values in trajectories),
        "source_physical_frame_count": len(physical_frames),
        "kinetic_trajectory_count": len(trajectories),
        "frame_interval": interval,
        "time_unit": time_unit,
        "models": models,
        "chapman_kolmogorov_tests": ck_tests,
        "implied_timescale_stability": implied_stability,
        "vamp_cross_validation": vamp,
        "time_block_bootstrap": bootstrap,
        "issues": issues,
    }


def _clustering_candidates(
    source: Path, project: Mapping[str, object], hash_content: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise MSMAnalysisError("project definitions are required for clustering selection")
    candidates: List[Dict[str, object]] = []
    provenance: List[Dict[str, object]] = []
    full_coverage_tolerance = 1.0e-12

    def complete_coverage(value: float) -> bool:
        return value >= 1.0 - full_coverage_tolerance

    def sampled_kinetic_rows(
        rows: Sequence[Mapping[str, object]],
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        """Retain assigned contiguous runs without crossing noise gaps."""

        groups: Dict[Tuple[str, ...], List[Mapping[str, object]]] = {}
        for row in rows:
            try:
                key = (
                    str(row["system_id"]), str(row["replica_id"]),
                    str(row["segment_id"]),
                    *((str(row["member_id"]),) if "member_id" in row else ()),
                )
                int(row["source_frame_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MSMAnalysisError(
                    "sampled clustering assignments lack trajectory identities"
                ) from exc
            groups.setdefault(key, []).append(row)
        retained_runs: List[List[Dict[str, object]]] = []
        noise_count = 0
        isolated_assigned_count = 0
        for key in sorted(groups):
            ordered = sorted(
                groups[key], key=lambda row: int(row["source_frame_index"])
            )
            run: List[Dict[str, object]] = []

            def flush() -> None:
                nonlocal isolated_assigned_count, run
                if len(run) >= 2:
                    retained_runs.append(run)
                else:
                    isolated_assigned_count += len(run)
                run = []

            for row in ordered:
                cluster_id = row.get("cluster_id")
                if (
                    isinstance(cluster_id, int)
                    and not isinstance(cluster_id, bool)
                    and cluster_id > 0
                ):
                    run.append(dict(row))
                else:
                    noise_count += 1
                    flush()
            flush()
        present_states = sorted({
            int(row["cluster_id"])
            for run in retained_runs for row in run
        })
        remap = {state: index + 1 for index, state in enumerate(present_states)}
        normalized = []
        for run_index, run in enumerate(retained_runs, start=1):
            for row in run:
                normalized.append({
                    **row,
                    "original_cluster_id": int(row["cluster_id"]),
                    "cluster_id": remap[int(row["cluster_id"])],
                    "original_segment_id": str(row["segment_id"]),
                    "segment_id": f"{row['segment_id']}::msm-run-{run_index:06d}",
                })
        return normalized, {
            "source_assignment_observation_count": len(rows),
            "retained_kinetic_observation_count": len(normalized),
            "retained_kinetic_run_count": len(retained_runs),
            "noise_observation_count": noise_count,
            "isolated_assigned_observation_count": isolated_assigned_count,
            "retained_state_count": len(present_states),
            "noise_gaps_crossed": False,
        }

    def add_standard(
        candidate_id: str, report: Mapping[str, object],
        score_key: str, coverage: float = 1.0, *, partial_msm: bool = False,
    ) -> None:
        selected = report.get("selected_model")
        rows = report.get("assignments")
        if not isinstance(selected, dict) or not isinstance(rows, list):
            raise MSMAnalysisError(f"{candidate_id} produced no selected assignments")
        score = selected.get(score_key)
        normalized = [dict(row) for row in rows if isinstance(row, dict)]
        complete = len(normalized) == len(rows) and all(
            isinstance(row.get("cluster_id"), int)
            and not isinstance(row.get("cluster_id"), bool)
            for row in normalized
        )
        msm_rows = normalized
        msm_diagnostics: Dict[str, object] = {
            "source_assignment_observation_count": len(normalized),
            "retained_kinetic_observation_count": len(normalized),
            "retained_kinetic_run_count": None,
            "noise_observation_count": 0,
            "isolated_assigned_observation_count": 0,
            "retained_state_count": len({
                int(row["cluster_id"]) for row in normalized
                if isinstance(row.get("cluster_id"), int)
            }),
            "noise_gaps_crossed": False,
        }
        if partial_msm:
            msm_rows, msm_diagnostics = sampled_kinetic_rows(normalized)
        msm_state_count = int(msm_diagnostics["retained_state_count"])
        msm_complete = bool(msm_rows) and msm_state_count >= 2
        msm_coverage = len(msm_rows) / len(normalized) if normalized else 0.0
        eligible = (
            msm_complete if partial_msm
            else complete and complete_coverage(coverage)
        )
        primary_eligible = eligible and not partial_msm
        candidates.append({
            "candidate_id": candidate_id,
            "family": "clustering",
            "algorithm": candidate_id,
            "geometric_score": (
                float(score) if isinstance(score, (int, float))
                and not isinstance(score, bool) else None
            ),
            "coverage_fraction": coverage,
            "assignments": msm_rows,
            "msm_assignment_scope": (
                "contiguous_nonnoise_fitted_observations"
                if partial_msm else "complete_all_source_observations"
            ),
            "msm_observation_count": len(msm_rows),
            "msm_coverage_fraction": msm_coverage,
            "msm_assignment_diagnostics": msm_diagnostics,
            "msm_eligible": eligible,
            "primary_msm_selection_eligible": primary_eligible,
            "msm_role": (
                "sampled_sensitivity" if partial_msm else "primary_candidate"
            ),
            "msm_exclusion_reason": (
                None if eligible
                else (
                    "fewer than two states remain in contiguous nonnoise runs"
                    if partial_msm
                    else "partition contains noise or lacks complete assignments"
                )
            ),
            "primary_msm_exclusion_reason": (
                None if primary_eligible else (
                    "noise-censored dense-core assignments do not cover the complete trajectory"
                    if partial_msm and eligible else
                    "no MSM-eligible assignment table is available"
                )
            ),
        })
        provenance.append({
            "candidate_id": candidate_id,
            "module_id": report.get("module_id"),
            "selected_model": selected,
        })

    if "clustering_kmeans" in definitions:
        add_standard(
            "kmeans", clustering_kmeans_project(source, hash_content=hash_content),
            "silhouette",
        )
    if "clustering_imwkmeans" in definitions:
        add_standard(
            "intelligent_minkowski_weighted_kmeans",
            clustering_imwkmeans_project(source, hash_content=hash_content),
            "silhouette",
        )
    if "clustering_hdbscan" in definitions:
        report = clustering_hdbscan_project(source, hash_content=hash_content)
        selected = report.get("selected_model")
        coverage = (
            float(selected.get("retained_fraction", 0.0))
            if isinstance(selected, dict) else 0.0
        )
        add_standard(
            "hdbscan", report, "retained_only_silhouette", coverage,
            partial_msm=True,
        )
    if "alternative_clustering" in definitions:
        report = alternative_clustering_project(source, hash_content=hash_content)
        results = report.get("algorithm_results")
        if not isinstance(results, list):
            raise MSMAnalysisError("alternative clustering produced no algorithm results")
        for result in results:
            if not isinstance(result, dict):
                continue
            algorithm = str(result.get("requested_algorithm"))
            candidate_id = {
                "mwpam": "minkowski_weighted_pam",
            }.get(algorithm, algorithm)
            source_observation_count = int(report.get("observation_count", 0))
            fit_observation_count = int(result.get("fit_observation_count", 0))
            fit_rows = result.get("fit_frame_assignments")
            fit_only_algorithm = algorithm in {"ward", "quality_threshold"}
            fit_covers_all = (
                fit_only_algorithm
                and source_observation_count > 0
                and fit_observation_count == source_observation_count
                and isinstance(fit_rows, list)
                and len(fit_rows) == source_observation_count
            )
            fit_has_noise = bool(
                isinstance(fit_rows, list) and any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("cluster_id"), int)
                    or isinstance(row.get("cluster_id"), bool)
                    for row in fit_rows
                )
            )
            partial_msm = fit_only_algorithm and (
                not fit_covers_all or fit_has_noise
            )
            if partial_msm:
                candidates.append({
                    "candidate_id": candidate_id,
                    "family": "clustering",
                    "algorithm": algorithm,
                    "geometric_score": None,
                    "coverage_fraction": 0.0,
                    "assignments": [],
                    "msm_assignment_scope": "skipped_incomplete_exact_fit",
                    "msm_observation_count": 0,
                    "msm_coverage_fraction": 0.0,
                    "msm_assignment_diagnostics": None,
                    "msm_eligible": False,
                    "primary_msm_selection_eligible": False,
                    "msm_role": "skipped",
                    "msm_exclusion_reason": (
                        "Ward and quality-threshold require a complete exact "
                        "fit with no unassigned observations"
                    ),
                    "primary_msm_exclusion_reason": (
                        "incomplete exact fit was skipped"
                    ),
                })
                provenance.append({
                    "candidate_id": candidate_id,
                    "module_id": "alternative_clustering",
                    "selected_parameters": result.get("parameters"),
                    "msm_assignment_scope": "skipped_incomplete_exact_fit",
                })
                continue
            rows = (
                fit_rows if fit_only_algorithm
                else result.get("frame_assignments")
            )
            evaluation = result.get("full_partition_silhouette_evaluation")
            score = (
                evaluation.get("score") if isinstance(evaluation, dict)
                else result.get("silhouette")
            )
            coverage = float(
                (
                    result.get("retained_fraction", 0.0)
                    * fit_observation_count
                    / max(1, source_observation_count)
                )
                if fit_only_algorithm
                else result.get(
                    "full_retained_fraction", result.get("retained_fraction", 0.0)
                )
            )
            normalized = [dict(row) for row in rows if isinstance(row, dict)] \
                if isinstance(rows, list) else []
            complete = bool(normalized) and len(normalized) == len(rows) and all(
                isinstance(row.get("cluster_id"), int)
                and not isinstance(row.get("cluster_id"), bool)
                for row in normalized
            )
            msm_rows = normalized
            msm_diagnostics: Dict[str, object] = {
                "source_assignment_observation_count": len(normalized),
                "retained_kinetic_observation_count": len(normalized),
                "retained_kinetic_run_count": None,
                "noise_observation_count": 0,
                "isolated_assigned_observation_count": 0,
                "retained_state_count": len({
                    int(row["cluster_id"]) for row in normalized
                    if isinstance(row.get("cluster_id"), int)
                }),
                "noise_gaps_crossed": False,
            }
            msm_coverage = (
                len(msm_rows) / max(1, int(report.get("observation_count", 0)))
            )
            eligible = complete and complete_coverage(coverage)
            primary_eligible = eligible
            assignment_scope = (
                "complete_exact_fitted_partition"
                if fit_only_algorithm else
                "complete_all_source_observations"
            )
            candidates.append({
                "candidate_id": candidate_id,
                "family": "clustering",
                "algorithm": algorithm,
                "geometric_score": (
                    float(score) if isinstance(score, (int, float))
                    and not isinstance(score, bool) else None
                ),
                "coverage_fraction": coverage,
                "assignments": msm_rows,
                "msm_assignment_scope": assignment_scope,
                "msm_observation_count": len(msm_rows),
                "msm_coverage_fraction": msm_coverage,
                "msm_assignment_diagnostics": msm_diagnostics,
                "msm_eligible": eligible,
                "primary_msm_selection_eligible": primary_eligible,
                "msm_role": "primary_candidate",
                "msm_exclusion_reason": (
                    None if eligible
                    else "no complete all-frame partition is available"
                ),
                "primary_msm_exclusion_reason": (
                    None if primary_eligible
                    else "no MSM-eligible assignment table is available"
                ),
            })
            provenance.append({
                "candidate_id": candidate_id,
                "module_id": "alternative_clustering",
                "selected_parameters": result.get("parameters"),
                "assignment_extension": result.get("assignment_extension"),
                "msm_assignment_scope": assignment_scope,
            })
    return candidates, provenance


def _state_model_selection_key(report: Mapping[str, object]) -> Tuple[object, ...]:
    vamp_values = [
        float(row["mean_heldout_vamp_e"])
        for row in report.get("vamp_cross_validation", [])  # type: ignore[union-attr]
        if isinstance(row, dict) and row.get("status") == "complete"
    ]
    ck_values = [
        float(row["matrix_rmse"])
        for row in report.get("chapman_kolmogorov_tests", [])  # type: ignore[union-attr]
        if isinstance(row, dict)
    ]
    its_rows = report.get("implied_timescale_stability")
    its_values = [
        float(row["relative_range"])
        for row in its_rows.get("modes", [])  # type: ignore[union-attr]
        if isinstance(row, dict)
    ] if isinstance(its_rows, dict) else []
    geometric = report.get("geometric_score")
    return (
        report.get("kinetic_validation_status") == "passed",
        float(np.mean(vamp_values)) if vamp_values else -math.inf,
        -float(np.mean(ck_values)) if ck_values else -math.inf,
        -float(np.mean(its_values)) if its_values else -math.inf,
        float(geometric) if isinstance(geometric, (int, float)) else -math.inf,
        -int(report.get("state_count", 0)),
        str(report.get("candidate_id", "")),
    )


def _selection_summary(report: Mapping[str, object]) -> Dict[str, object]:
    return {
        "candidate_id": report.get("candidate_id"),
        "family": report.get("family"),
        "msm_role": report.get("msm_role"),
        "assignment_scope": report.get("assignment_scope"),
        "assignment_observation_count": report.get("assignment_observation_count"),
        "assignment_coverage_fraction": report.get("assignment_coverage_fraction"),
        "state_count": report.get("state_count"),
        "geometric_score": report.get("geometric_score"),
        "geometric_coverage_fraction": report.get("geometric_coverage_fraction"),
        "kinetic_validation_status": report.get("kinetic_validation_status"),
        "validation_gates": report.get("validation_gates"),
        "selection_key": list(_state_model_selection_key(report)),
    }


def _multi_state_markov_project(
    source: Path, project: Mapping[str, object], settings: Mapping[str, object],
    hash_content: bool,
) -> Dict[str, object]:
    candidates, provenance = _clustering_candidates(source, project, hash_content)
    geometric_candidates = [
        row for row in candidates
        if isinstance(row.get("geometric_score"), (int, float))
        and not isinstance(row.get("geometric_score"), bool)
    ]
    if not geometric_candidates:
        raise MSMAnalysisError("no enabled clustering method produced a geometric score")
    best_geometric = max(
        geometric_candidates,
        key=lambda row: (
            float(row["geometric_score"]), float(row["coverage_fraction"]),
            str(row["candidate_id"]),
        ),
    )
    clustering_models = []
    candidate_issues = []
    for candidate in candidates:
        if candidate.get("msm_eligible") is not True:
            continue
        candidate_state_count = len({
            int(row["cluster_id"])
            for row in candidate["assignments"]  # type: ignore[union-attr]
            if isinstance(row, dict)
            and isinstance(row.get("cluster_id"), int)
            and not isinstance(row.get("cluster_id"), bool)
        })
        if candidate_state_count > int(settings["maximum_states"]):
            candidate["msm_eligible"] = False
            candidate["primary_msm_selection_eligible"] = False
            candidate["msm_role"] = "skipped"
            candidate["msm_exclusion_reason"] = (
                f"state count {candidate_state_count} exceeds maximum_states "
                f"{settings['maximum_states']}"
            )
            candidate["primary_msm_exclusion_reason"] = (
                "over-fragmented partition was excluded from MSM construction"
            )
            candidate_issues.append({
                "severity": "warning",
                "code": "MSM_STATE_COUNT_EXCEEDS_LIMIT",
                "location": str(candidate["candidate_id"]),
                "message": candidate["msm_exclusion_reason"],
            })
            continue
        model = _evaluate_state_definition(
            candidate["assignments"], settings,  # type: ignore[arg-type]
            candidate_id=str(candidate["candidate_id"]), family="clustering",
            geometric_score=float(candidate["geometric_score"]),
            geometric_coverage=float(candidate["msm_coverage_fraction"]),
        )
        model.update({
            "msm_role": candidate["msm_role"],
            "primary_msm_selection_eligible": candidate[
                "primary_msm_selection_eligible"
            ],
            "assignment_scope": candidate["msm_assignment_scope"],
            "assignment_observation_count": candidate["msm_observation_count"],
            "assignment_coverage_fraction": candidate["msm_coverage_fraction"],
            "assignment_diagnostics": candidate["msm_assignment_diagnostics"],
            "interpretation_scope": (
                "conditional on retained sampled assignments; stationary populations, "
                "residence, and timescales do not describe the omitted observations"
                if candidate["msm_role"] == "sampled_sensitivity"
                else "complete enabled trajectory partition"
            ),
        })
        clustering_models.append(model)
    primary_clustering_models = [
        model for model in clustering_models
        if model.get("primary_msm_selection_eligible") is True
    ]
    sampled_sensitivity_models = [
        model for model in clustering_models
        if model.get("msm_role") == "sampled_sensitivity"
    ]
    if not primary_clustering_models:
        raise MSMAnalysisError(
            "no enabled clustering method produced a complete MSM-eligible partition"
        )
    best_clustering = max(
        primary_clustering_models, key=_state_model_selection_key
    )

    fes = pca_fes_basins_project(source, hash_content=hash_content)
    fes_rows = fes.get("frame_assignments")
    if not isinstance(fes_rows, list):
        raise MSMAnalysisError("PCA-FES produced no frame assignments")
    normalized_fes = [
        {**row, "cluster_id": row.get("basin_id")}
        for row in fes_rows if isinstance(row, dict)
    ]
    if len(normalized_fes) != len(fes_rows) or not all(
        isinstance(row.get("cluster_id"), int)
        and not isinstance(row.get("cluster_id"), bool)
        for row in normalized_fes
    ):
        raise MSMAnalysisError(
            "primary FES contains unassigned frames and cannot define a complete transition model"
        )
    fes_silhouette = fes.get("basin_silhouette")
    fes_score = (
        float(fes_silhouette["score"])
        if isinstance(fes_silhouette, dict)
        and isinstance(fes_silhouette.get("score"), (int, float))
        else None
    )
    fes_model = _evaluate_state_definition(
        normalized_fes, settings, candidate_id="pca_fes_basins",
        family="fes", geometric_score=fes_score, geometric_coverage=1.0,
    )
    fes_model.update({
        "msm_role": "separate_fes_sensitivity",
        "primary_msm_selection_eligible": False,
        "assignment_scope": "complete_all_source_observations",
        "assignment_observation_count": len(normalized_fes),
        "assignment_coverage_fraction": 1.0,
        "interpretation_scope": "complete PCA-FES basin partition",
    })

    physical_frame_count = len({
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        for row in normalized_fes
    })

    all_issues = candidate_issues + [
        issue for report in [*clustering_models, fes_model]
        for issue in report.get("issues", [])  # type: ignore[union-attr]
        if isinstance(issue, dict)
    ]
    first = fes
    definitions = project.get("definitions")
    clustering_feature_space = "unknown"
    if isinstance(definitions, dict):
        for definition_id in (
            "clustering_kmeans", "clustering_hdbscan",
            "clustering_imwkmeans", "alternative_clustering",
        ):
            definition = definitions.get(definition_id)
            if isinstance(definition, dict) and isinstance(
                definition.get("feature_source"), str
            ):
                clustering_feature_space = str(definition["feature_source"])
                break
    return {
        "module_id": "markov_state_models",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "kinetic_validation_status": (
            "passed"
            if best_clustering["kinetic_validation_status"] == "passed"
            and fes_model["kinetic_validation_status"] == "passed"
            else "not passed"
        ),
        "project_manifest_path": str(source),
        "project_manifest_sha256": first["project_manifest_sha256"],
        "system_manifest_path": first["system_manifest_path"],
        "system_manifest_sha256": first["system_manifest_sha256"],
        "input_content_signature_sha256": first["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "observation_count": len(normalized_fes),
        "observation_accounting": {
            "source_physical_frame_count": physical_frame_count,
            "symmetry_expanded_observation_count": len(normalized_fes),
            "member_observations_are_independent_replicas": False
            if any("member_id" in row for row in normalized_fes) else None,
        },
        "clustering_feature_space": clustering_feature_space,
        "fes_feature_space": "common_pca",
        "best_geometric_clustering": {
            key: best_geometric.get(key) for key in (
                "candidate_id", "geometric_score", "coverage_fraction",
                "msm_eligible", "primary_msm_selection_eligible", "msm_role",
                "msm_exclusion_reason", "primary_msm_exclusion_reason",
            )
        },
        "best_clustering_state_model": _selection_summary(best_clustering),
        "best_clustering_state_model_details": best_clustering,
        "fes_state_model": _selection_summary(fes_model),
        "fes_state_model_details": fes_model,
        "clustering_state_model_comparison": [
            _selection_summary(report)
            for report in sorted(
                primary_clustering_models,
                key=_state_model_selection_key, reverse=True
            )
        ],
        "sampled_clustering_state_model_sensitivities": [
            _selection_summary(report)
            for report in sorted(
                sampled_sensitivity_models,
                key=_state_model_selection_key, reverse=True,
            )
        ],
        "sampled_clustering_state_model_sensitivity_details": sorted(
            sampled_sensitivity_models,
            key=_state_model_selection_key, reverse=True,
        ),
        "clustering_method_inventory": [
            {
                key: candidate.get(key) for key in (
                    "candidate_id", "geometric_score", "coverage_fraction",
                    "msm_assignment_scope", "msm_observation_count",
                    "msm_coverage_fraction", "msm_assignment_diagnostics",
                    "msm_eligible", "primary_msm_selection_eligible", "msm_role",
                    "msm_exclusion_reason", "primary_msm_exclusion_reason",
                )
            }
            for candidate in candidates
        ],
        "clustering_provenance": provenance,
        "selection_rule": (
            "only complete all-frame clustering partitions enter primary selection; "
            "then kinetic gate pass, mean held-out VAMP-E, lower mean CK RMSE, "
            "lower implied-timescale relative range, geometric score, and fewer states; "
            "FES and sampled/noise-censored clustering sensitivities are reported "
            "separately and are never promoted by this ranking"
        ),
        "error_count": 0,
        "warning_count": sum(
            issue.get("severity") == "warning" for issue in all_issues
        ),
        "issues": all_issues,
        "limitations": [
            "The selected clustering MSM and FES-basin model are separate state definitions and both remain reportable.",
            "Silhouette selects geometric partitions only; it does not establish metastability or kinetics.",
            "FES basins are thermodynamic or occupancy catchments and are not assumed Markovian.",
            "VAMP-E cross-validation uses contiguous within-segment time folds and never leaves out an entire replica.",
            "Time-block bootstrap intervals quantify finite-trajectory sensitivity but do not create independent replicas.",
            "HDBSCAN dense-core models split every noise gap and are conditional on retained core observations; they cannot recover full-trajectory stationary populations, residence, or kinetics.",
            "Ward and quality-threshold are omitted when they cannot provide a complete exact assignment over every observation.",
            "PaLD community MSMs are emitted only by the separate PaLD module and never enter conventional MSM selection.",
            "Technical completion does not establish mechanistic or scientific validity.",
        ],
    }


def markov_state_models_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    if "assignment_sources" in settings:
        return _multi_state_markov_project(
            source, project, settings, hash_content
        )
    if settings["assignment_source"] == "clustering_kmeans":
        clustering = clustering_kmeans_project(source, hash_content=hash_content)
    else:
        clustering = clustering_imwkmeans_project(source, hash_content=hash_content)
    rows = clustering.get("assignments")
    if not isinstance(rows, list):
        raise MSMAnalysisError("assignment source did not return a complete assignment table")
    trajectories, interval, time_unit, state_count = _group_assignments(rows)
    physical_frame_count = len({
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        for row in rows
    })
    if state_count > int(settings["maximum_states"]):
        raise MSMAnalysisError("assignment state count exceeds maximum_states gate")
    models = []
    issues = [issue for issue in clustering.get("issues", []) if isinstance(issue, dict)]
    for estimator in settings["estimators"]:  # type: ignore[union-attr]
        for lag in settings["lag_frames"]:  # type: ignore[union-attr]
            try:
                model = estimate_transition_model(
                    trajectories, state_count, int(lag), str(estimator), interval, time_unit
                )
            except MSMAnalysisError as exc:
                model = {
                    "estimator": str(estimator),
                    "lag_frames": int(lag),
                    "lag_time": int(lag) * interval,
                    "time_unit": time_unit,
                    "estimable": False,
                    "failure": str(exc),
                    "transition_count": sum(
                        max(0, len(trajectory) - int(lag)) for trajectory in trajectories
                    ),
                    "connected": False,
                    "stationary_iteration_converged": False,
                    "passes_minimum_transition_count": False,
                }
                issues.append({
                    "severity": "warning",
                    "code": "MSM_MODEL_NOT_ESTIMABLE",
                    "location": f"{estimator}/lag-{lag}",
                    "message": str(exc),
                })
                models.append(model)
                continue
            model["estimable"] = True
            model["failure"] = None
            model["passes_minimum_transition_count"] = (
                model["transition_count"] >= int(settings["minimum_transition_count"])
            )
            if not model["connected"]:
                issues.append({
                    "severity": "warning",
                    "code": "MSM_DISCONNECTED",
                    "location": f"{estimator}/lag-{lag}",
                    "message": "transition graph is not connected",
                })
            if not model["passes_minimum_transition_count"]:
                issues.append({
                    "severity": "warning",
                    "code": "MSM_TRANSITIONS_BELOW_GATE",
                    "location": f"{estimator}/lag-{lag}",
                    "message": "transition count is below the declared validation gate",
                })
            models.append(model)
    by_key = {
        (model["estimator"], model["lag_frames"]): model
        for model in models if model.get("estimable")
    }
    ck_tests = []
    for model in models:
        if not model.get("estimable"):
            continue
        for multiple in settings["ck_multiples"]:  # type: ignore[union-attr]
            direct_lag = int(model["lag_frames"]) * int(multiple)
            direct_key = (model["estimator"], direct_lag)
            if direct_key not in by_key:
                continue
            predicted = _matrix_power(model["transition_matrix"], int(multiple))
            observed = by_key[direct_key]["transition_matrix"]
            squared = [
                (predicted[i][j] - observed[i][j]) ** 2
                for i in range(state_count) for j in range(state_count)
            ]
            rmse = math.sqrt(sum(squared) / len(squared))
            ck_tests.append({
                "estimator": model["estimator"],
                "base_lag_frames": model["lag_frames"],
                "multiple": multiple,
                "direct_lag_frames": direct_lag,
                "matrix_rmse": rmse,
                "passes_declared_gate": rmse <= float(settings["maximum_ck_rmse"]),
            })
    validation_passed = bool(models) and all(
        model.get("estimable") and model["connected"] and model["passes_minimum_transition_count"]
        and model["stationary_iteration_converged"]
        for model in models
    ) and bool(ck_tests) and all(test["passes_declared_gate"] for test in ck_tests)
    return {
        "module_id": "markov_state_models",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "kinetic_validation_status": "passed" if validation_passed else "not passed",
        "project_manifest_path": str(source),
        "project_manifest_sha256": clustering["project_manifest_sha256"],
        "system_manifest_path": clustering["system_manifest_path"],
        "system_manifest_sha256": clustering["system_manifest_sha256"],
        "input_content_signature_sha256": clustering["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "assignment_source_selected_model": clustering["selected_model"],
        "segment_count": len(trajectories),
        "observation_count": sum(len(values) for values in trajectories),
        "observation_accounting": {
            "source_physical_frame_count": physical_frame_count,
            "symmetry_expanded_observation_count": len(rows),
            "kinetic_trajectory_count": len(trajectories),
            "member_observations_are_independent_replicas": False
            if any("member_id" in row for row in rows) else None,
        },
        "state_count": state_count,
        "frame_interval": interval,
        "time_unit": time_unit,
        "segment_boundary_policy": "transitions are counted within each declared segment only",
        "models": models,
        "chapman_kolmogorov_tests": ck_tests,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "A complete state assignment is necessary but does not establish metastability.",
            "No transitions are inferred across segment boundaries.",
            "Equivalent oligomer members are separate discrete trajectories; transitions never join two member identities.",
            "Symmetry-expanded member trajectories are paired representations of the same physical frames, not additional independent replicas.",
            "Nonreversible implied timescales are omitted because complex eigenmodes require a separately validated solver.",
            "Rates, pathways, and mechanistic claims remain invalid unless lag, connectivity, sampling, and convergence gates pass on adequate data.",
            "Technical completion does not establish scientific validity.",
        ],
    }


def markov_state_models_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return markov_state_models_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, PCAAnalysisError, ClusteringAnalysisError,
        AlternativeClusteringError, PCAFESAnalysisError, MSMAnalysisError,
        OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "markov_state_models",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "kinetic_validation_status": "not passed",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "MSM_INVALID", "message": message}
                for message in messages
            ],
        }
