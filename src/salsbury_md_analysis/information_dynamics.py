"""Directional and higher-order information analyses."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .pca import common_pca_project
from .validation import positive_integer


class InformationDynamicsError(ValueError):
    """Raised when information-dynamics inputs cannot support the estimator."""


def _segments(values: Sequence[Sequence[Sequence[float]]]) -> List[np.ndarray]:
    arrays = [np.asarray(segment, dtype=float) for segment in values]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise InformationDynamicsError("features must contain two-dimensional segments")
    feature_count = arrays[0].shape[1]
    if feature_count < 1 or any(array.shape[1] != feature_count for array in arrays):
        raise InformationDynamicsError("feature dimensions must be positive and consistent")
    if any(not np.isfinite(array).all() for array in arrays):
        raise InformationDynamicsError("features contain non-finite values")
    return arrays


def _quantile_bins(arrays: Sequence[np.ndarray], bin_count: int) -> Tuple[np.ndarray, ...]:
    combined = np.concatenate(arrays, axis=0)
    edges = []
    for feature in range(combined.shape[1]):
        observed = np.unique(combined[:, feature])
        if len(observed) < 2:
            raise InformationDynamicsError(
                f"feature {feature + 1} is constant and cannot be discretized"
            )
        if len(observed) <= bin_count:
            interior = (observed[:-1] + observed[1:]) / 2.0
        else:
            quantiles = np.quantile(
                combined[:, feature], np.linspace(0.0, 1.0, bin_count + 1)
            )
            interior = np.unique(quantiles[1:-1])
        edges.append(np.concatenate(([-np.inf], interior, [np.inf])))
    return tuple(edges)


def transfer_entropy_matrix(
    feature_segments: Sequence[Sequence[Sequence[float]]],
    lag_frames: int,
    bin_count: int,
    minimum_pairs: int,
) -> Dict[str, object]:
    """Estimate segment-safe discrete transfer entropy in natural-log units.

    Matrix element ``[source][target]`` is
    ``I(target[t+lag]; source[t] | target[t])``.
    """

    arrays = _segments(feature_segments)
    lag_frames = positive_integer(lag_frames, "lag_frames")
    bin_count = positive_integer(bin_count, "bin_count")
    minimum_pairs = positive_integer(minimum_pairs, "minimum_pairs")
    edges = _quantile_bins(arrays, bin_count)
    discrete = [
        np.column_stack([
            np.digitize(array[:, feature], edges[feature][1:-1], right=False)
            for feature in range(array.shape[1])
        ]).astype(int)
        for array in arrays
    ]
    pair_count = sum(max(0, len(array) - lag_frames) for array in discrete)
    if pair_count < minimum_pairs:
        raise InformationDynamicsError(
            f"transfer entropy has {pair_count} lag pairs; minimum is {minimum_pairs}"
        )
    size = discrete[0].shape[1]
    matrix = [[0.0] * size for _ in range(size)]
    for source in range(size):
        for target in range(size):
            if source == target:
                continue
            joint: Counter[Tuple[int, int, int]] = Counter()
            target_transition: Counter[Tuple[int, int]] = Counter()
            source_target: Counter[Tuple[int, int]] = Counter()
            target_now: Counter[int] = Counter()
            for segment in discrete:
                for index in range(len(segment) - lag_frames):
                    source_now = int(segment[index, source])
                    target_current = int(segment[index, target])
                    target_next = int(segment[index + lag_frames, target])
                    joint[(target_next, target_current, source_now)] += 1
                    target_transition[(target_next, target_current)] += 1
                    source_target[(target_current, source_now)] += 1
                    target_now[target_current] += 1
            estimate = 0.0
            for (target_next, target_current, source_now), count in joint.items():
                conditional_full = count / source_target[(target_current, source_now)]
                conditional_target = (
                    target_transition[(target_next, target_current)]
                    / target_now[target_current]
                )
                estimate += (count / pair_count) * math.log(
                    conditional_full / conditional_target
                )
            matrix[source][target] = max(0.0, estimate)
    return {
        "transfer_entropy_nats": matrix,
        "lag_frames": lag_frames,
        "pair_count": pair_count,
        "bin_count_requested": bin_count,
        "bin_edges": [edge.tolist() for edge in edges],
        "direction_contract": "row source at t; column target at t+lag conditional on target at t",
    }


def lagged_cross_correlation(
    feature_segments: Sequence[Sequence[Sequence[float]]],
    lag_frames: int,
    minimum_pairs: int = 1,
) -> Dict[str, object]:
    """Calculate a segment-safe source-at-t to target-at-t+lag correlation matrix."""

    arrays = _segments(feature_segments)
    lag_frames = positive_integer(lag_frames, "lag_frames")
    minimum_pairs = positive_integer(minimum_pairs, "minimum_pairs")
    sources = [array[:-lag_frames] for array in arrays if len(array) > lag_frames]
    targets = [array[lag_frames:] for array in arrays if len(array) > lag_frames]
    pair_count = sum(len(array) for array in sources)
    if pair_count < minimum_pairs:
        raise InformationDynamicsError(
            f"lagged correlation has {pair_count} lag pairs; minimum is {minimum_pairs}"
        )
    source = np.concatenate(sources, axis=0)
    target = np.concatenate(targets, axis=0)
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = source_centered.T @ target_centered / pair_count
    source_scale = np.sqrt(np.mean(source_centered * source_centered, axis=0))
    target_scale = np.sqrt(np.mean(target_centered * target_centered, axis=0))
    denominator = np.outer(source_scale, target_scale)
    matrix = np.full_like(covariance, np.nan)
    np.divide(covariance, denominator, out=matrix, where=denominator > 1.0e-15)
    return {
        "lagged_cross_correlation": [
            [None if not math.isfinite(float(value)) else float(value) for value in row]
            for row in matrix
        ],
        "lag_frames": lag_frames,
        "pair_count": pair_count,
        "feature_count": int(matrix.shape[0]),
        "constant_source_feature_indices": [
            int(index) for index in np.where(source_scale <= 1.0e-15)[0]
        ],
        "constant_target_feature_indices": [
            int(index) for index in np.where(target_scale <= 1.0e-15)[0]
        ],
        "direction_contract": "row source feature at t; column target feature at t+lag",
        "normalization_contract": "pair-endpoint mean-centered population covariance divided by source and target population standard deviations",
    }


def coskewness_tensor(
    features: Sequence[Sequence[float]], maximum_tensor_elements: int = 1_000_000
) -> Dict[str, object]:
    """Return the standardized third central cross-moment tensor."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise InformationDynamicsError("coskewness requires a 2D feature matrix")
    if not np.isfinite(values).all():
        raise InformationDynamicsError("coskewness features contain non-finite values")
    maximum_tensor_elements = positive_integer(
        maximum_tensor_elements, "maximum_tensor_elements"
    )
    if values.shape[1] ** 3 > maximum_tensor_elements:
        raise InformationDynamicsError("coskewness maximum_tensor_elements gate exceeded")
    centered = values - values.mean(axis=0)
    scales = np.sqrt(np.mean(centered * centered, axis=0))
    valid = scales > 1.0e-15
    standardized = np.zeros_like(centered)
    standardized[:, valid] = centered[:, valid] / scales[valid]
    tensor = np.einsum(
        "ni,nj,nk->ijk", standardized, standardized, standardized, optimize=True
    ) / len(values)
    if np.any(~valid):
        invalid = np.where(~valid)[0]
        tensor[invalid, :, :] = np.nan
        tensor[:, invalid, :] = np.nan
        tensor[:, :, invalid] = np.nan
    return {
        "coskewness": [
            [
                [None if not math.isfinite(float(value)) else float(value) for value in row]
                for row in plane
            ]
            for plane in tensor
        ],
        "observation_count": int(values.shape[0]),
        "feature_count": int(values.shape[1]),
        "constant_feature_indices": [int(index) for index in np.where(~valid)[0]],
        "definition": "mean product of standardized mean-centered feature triplets",
    }


def displacement_propagator(
    coordinate_segments: Sequence[Sequence[Sequence[Sequence[float]]]],
    lag_frames: int,
    minimum_pairs: int = 1,
) -> Dict[str, object]:
    """Calculate a lagged displacement-vector propagator segment-safely."""

    arrays = [np.asarray(segment, dtype=float) for segment in coordinate_segments]
    if not arrays or any(array.ndim != 3 or array.shape[2] != 3 for array in arrays):
        raise InformationDynamicsError(
            "coordinate_segments must contain frame x atom x 3 arrays"
        )
    atom_count = arrays[0].shape[1]
    if any(array.shape[1] != atom_count or not np.isfinite(array).all() for array in arrays):
        raise InformationDynamicsError("coordinate segments are inconsistent or non-finite")
    lag_frames = positive_integer(lag_frames, "lag_frames")
    deltas = [array[:-lag_frames] - array[lag_frames:] for array in arrays if len(array) > lag_frames]
    pair_count = sum(len(delta) for delta in deltas)
    if pair_count < positive_integer(minimum_pairs, "minimum_pairs"):
        raise InformationDynamicsError("displacement propagator has too few lag pairs")
    combined = np.concatenate(deltas, axis=0)
    average_delta = combined.mean(axis=0)
    average_dot = np.einsum("nai,nbi->ab", combined, combined) / pair_count
    diagonal = np.diag(average_dot)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    normalized = np.full_like(average_dot, np.nan)
    np.divide(average_dot, denominator, out=normalized, where=denominator > 1.0e-15)
    return {
        "normalized_displacement_dot_matrix": [
            [None if not math.isfinite(float(value)) else float(value) for value in row]
            for row in normalized
        ],
        "average_displacement_vectors": average_delta.tolist(),
        "average_displacement_vector_dot_matrix": np.inner(
            average_delta, average_delta
        ).tolist(),
        "lag_frames": lag_frames,
        "pair_count": pair_count,
        "atom_count": atom_count,
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("information_dynamics") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise InformationDynamicsError("definitions.information_dynamics must be an object")
    required = {
        "feature_source", "component_indices", "analyses", "lag_frames",
        "bin_count", "minimum_pairs", "maximum_features",
        "maximum_tensor_elements",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise InformationDynamicsError(
            "information dynamics settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    if raw["feature_source"] != "common_pca":
        raise InformationDynamicsError("feature_source must be common_pca")
    components = raw["component_indices"]
    if not isinstance(components, list) or not components or len(set(components)) != len(components):
        raise InformationDynamicsError("component_indices must be a nonempty unique array")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in components):
        raise InformationDynamicsError("component_indices must contain positive integers")
    analyses = raw["analyses"]
    allowed = {"transfer_entropy", "lagged_cross_correlation", "coskewness"}
    if not isinstance(analyses, list) or not analyses or any(value not in allowed for value in analyses):
        raise InformationDynamicsError(
            "analyses must contain transfer_entropy, lagged_cross_correlation, and/or coskewness"
        )
    if len(components) > positive_integer(raw["maximum_features"], "maximum_features"):
        raise InformationDynamicsError("maximum_features gate exceeded")
    return dict(raw)


def _pca_segments(
    report: Mapping[str, object], component_indices: Sequence[int]
) -> Tuple[List[List[List[float]]], List[Dict[str, object]]]:
    zero_based = [value - 1 for value in component_indices]
    result = []
    identities = []
    systems = report.get("systems")
    if not isinstance(systems, list):
        raise InformationDynamicsError("common_pca report contains no systems")
    for system in systems:
        for replica in system["replicas"]:
            for segment in replica["segments"]:
                member_rows: Dict[str | None, List[List[float]]] = {}
                for projection in segment["projections"]:
                    scores = projection["scores_angstrom"]
                    if max(zero_based) >= len(scores):
                        raise InformationDynamicsError(
                            "component_indices exceed common_pca output"
                        )
                    member_id = (
                        str(projection["member_id"])
                        if "member_id" in projection else None
                    )
                    member_rows.setdefault(member_id, []).append(
                        [float(scores[index]) for index in zero_based]
                    )
                for member_id, rows in sorted(
                    member_rows.items(), key=lambda item: "" if item[0] is None else item[0]
                ):
                    if not rows:
                        continue
                    result.append(rows)
                    identities.append({
                        "system_id": system["system_id"],
                        "replica_id": replica["replica_id"],
                        "segment_id": segment["segment_id"],
                        **({"member_id": member_id} if member_id is not None else {}),
                        "observation_count": len(rows),
                    })
    if not result:
        raise InformationDynamicsError("common_pca produced no segment features")
    return result, identities


def information_dynamics_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    pca = common_pca_project(source, hash_content=hash_content)
    segments, identities = _pca_segments(pca, settings["component_indices"])
    combined = [row for segment in segments for row in segment]
    analyses: Dict[str, object] = {}
    if "transfer_entropy" in settings["analyses"]:
        analyses["transfer_entropy"] = transfer_entropy_matrix(
            segments,
            positive_integer(settings["lag_frames"], "lag_frames"),
            positive_integer(settings["bin_count"], "bin_count"),
            positive_integer(settings["minimum_pairs"], "minimum_pairs"),
        )
    if "lagged_cross_correlation" in settings["analyses"]:
        analyses["lagged_cross_correlation"] = lagged_cross_correlation(
            segments,
            positive_integer(settings["lag_frames"], "lag_frames"),
            positive_integer(settings["minimum_pairs"], "minimum_pairs"),
        )
    if "coskewness" in settings["analyses"]:
        analyses["coskewness"] = coskewness_tensor(
            combined,
            positive_integer(settings["maximum_tensor_elements"], "maximum_tensor_elements"),
        )
    return {
        "module_id": "information_dynamics",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": pca["project_manifest_sha256"],
        "system_manifest_path": pca["system_manifest_path"],
        "system_manifest_sha256": pca["system_manifest_sha256"],
        "input_content_signature_sha256": pca["input_content_signature_sha256"],
        "settings": settings,
        "segment_identities": identities,
        "observation_count": len(combined),
        "feature_count": len(combined[0]),
        "analyses": analyses,
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "limitations": [
            "Transfer entropy is a finite-bin estimator and requires bin, lag, and sample-size sensitivity analysis.",
            "Lag pairs never cross a segment or replica boundary.",
            "Equivalent oligomer members are separate time series; lag pairs never cross member identities.",
            "Symmetry-expanded member observations are paired within physical frames, not independent replicas.",
            "Directional dependence does not establish causal mechanism.",
            "Lagged cross-correlation is directional in time but remains an association measure, not causal evidence.",
            "Coskewness is a third central moment and is protected by an explicit cubic resource gate.",
        ],
    }


def information_dynamics_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return information_dynamics_project(project_path, hash_content=hash_content)
    except (InformationDynamicsError, ManifestValidationError, OSError, KeyError, ValueError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "information_dynamics",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "INFORMATION_DYNAMICS_INVALID", "message": message}
                for message in messages
            ],
        }
