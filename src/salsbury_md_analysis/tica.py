"""Segment-safe time-lagged independent component analysis."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .oligomer_symmetry import OligomerSymmetryError, paired_member_score_correlations
from .pca import PCAAnalysisError, common_pca_project
from .validation import positive_integer


class TICAAnalysisError(ValueError):
    """Raised when a TICA definition or generalized eigenproblem is unsafe."""


def fit_tica(
    segment_features: Sequence[Sequence[Sequence[float]]],
    *,
    lag_frames: int,
    component_count: int,
    covariance_regularization: float = 0.0,
    covariance_eigenvalue_cutoff: float = 1.0e-10,
) -> Dict[str, object]:
    """Fit reversible TICA without joining pairs across segment boundaries.

    The instantaneous covariance is the mean of the lag-pair endpoint
    covariances. The lagged covariance is explicitly symmetrized. Eigenvectors
    are normalized in the regularized instantaneous covariance metric.
    """

    if isinstance(lag_frames, bool) or not isinstance(lag_frames, int) or lag_frames <= 0:
        raise TICAAnalysisError("lag_frames must be a positive integer")
    if isinstance(component_count, bool) or not isinstance(component_count, int) or component_count <= 0:
        raise TICAAnalysisError("component_count must be a positive integer")
    for value, label, allow_zero in (
        (covariance_regularization, "covariance_regularization", True),
        (covariance_eigenvalue_cutoff, "covariance_eigenvalue_cutoff", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            relation = "nonnegative" if allow_zero else "positive"
            raise TICAAnalysisError(f"{label} must be finite and {relation}")
    if not segment_features:
        raise TICAAnalysisError("at least one feature segment is required")

    arrays: List[np.ndarray] = []
    feature_count = None
    for index, segment in enumerate(segment_features):
        values = np.asarray(segment, dtype=float)
        if values.ndim != 2 or values.shape[0] <= lag_frames or values.shape[1] == 0:
            raise TICAAnalysisError(
                f"segment {index} must contain more than lag_frames rows and at least one feature"
            )
        if not np.isfinite(values).all():
            raise TICAAnalysisError(f"segment {index} contains a non-finite feature")
        if feature_count is None:
            feature_count = int(values.shape[1])
        elif values.shape[1] != feature_count:
            raise TICAAnalysisError("all TICA segments must have the same feature count")
        arrays.append(values)
    assert feature_count is not None
    if component_count > feature_count:
        raise TICAAnalysisError("component_count cannot exceed feature count")

    left = np.concatenate([values[:-lag_frames] for values in arrays], axis=0)
    right = np.concatenate([values[lag_frames:] for values in arrays], axis=0)
    pair_count = int(left.shape[0])
    mean = (left.sum(axis=0) + right.sum(axis=0)) / (2.0 * pair_count)
    left_centered = left - mean
    right_centered = right - mean
    covariance = (
        left_centered.T @ left_centered + right_centered.T @ right_centered
    ) / (2.0 * pair_count)
    lagged = (
        left_centered.T @ right_centered + right_centered.T @ left_centered
    ) / (2.0 * pair_count)
    covariance = 0.5 * (covariance + covariance.T)
    lagged = 0.5 * (lagged + lagged.T)
    scale = float(np.trace(covariance)) / feature_count
    if not math.isfinite(scale) or scale <= 0.0:
        raise TICAAnalysisError("instantaneous feature variance is zero or non-finite")
    regularization = float(covariance_regularization) * scale
    regularized = covariance + regularization * np.eye(feature_count)

    covariance_values, covariance_vectors = np.linalg.eigh(regularized)
    largest = float(covariance_values[-1])
    threshold = float(covariance_eigenvalue_cutoff) * largest
    retained = covariance_values > threshold
    rank = int(retained.sum())
    if rank == 0:
        raise TICAAnalysisError("no instantaneous covariance modes exceed the eigenvalue cutoff")
    if component_count > rank:
        raise TICAAnalysisError(
            f"component_count {component_count} exceeds retained covariance rank {rank}"
        )
    whitening = covariance_vectors[:, retained] / np.sqrt(covariance_values[retained])
    whitened_lagged = whitening.T @ lagged @ whitening
    whitened_lagged = 0.5 * (whitened_lagged + whitened_lagged.T)
    eigenvalues, reduced_vectors = np.linalg.eigh(whitened_lagged)
    order = sorted(range(rank), key=lambda index: (-abs(float(eigenvalues[index])), -float(eigenvalues[index]), index))
    selected = order[:component_count]
    values = eigenvalues[selected]
    vectors = whitening @ reduced_vectors[:, selected]
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        metric_norm = math.sqrt(float(vector.T @ regularized @ vector))
        if not math.isfinite(metric_norm) or metric_norm <= 0.0:
            raise TICAAnalysisError("TICA eigenvector has invalid covariance-metric norm")
        vector = vector / metric_norm
        pivot = max(range(feature_count), key=lambda index: (abs(float(vector[index])), -index))
        if vector[pivot] < 0.0:
            vector = -vector
        vectors[:, column] = vector
    residuals = []
    for column, eigenvalue in enumerate(values):
        residual = lagged @ vectors[:, column] - float(eigenvalue) * regularized @ vectors[:, column]
        residuals.append(float(np.linalg.norm(residual)))
    return {
        "pair_count": pair_count,
        "feature_count": feature_count,
        "retained_covariance_rank": rank,
        "mean": mean.tolist(),
        "instantaneous_covariance": covariance.tolist(),
        "regularized_instantaneous_covariance": regularized.tolist(),
        "symmetrized_lagged_covariance": lagged.tolist(),
        "regularization_absolute": regularization,
        "eigenvalues": [float(value) for value in values],
        "eigenvectors": vectors.T.tolist(),
        "generalized_eigen_residual_norms": residuals,
    }


def project_tica(
    features: Sequence[Sequence[float]], mean: Sequence[float], eigenvectors: Sequence[Sequence[float]]
) -> List[List[float]]:
    values = np.asarray(features, dtype=float)
    center = np.asarray(mean, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=float)
    if values.ndim != 2 or center.shape != (values.shape[1],) or vectors.ndim != 2 or vectors.shape[1] != values.shape[1]:
        raise TICAAnalysisError("TICA projection dimensions do not match")
    if not np.isfinite(values).all():
        raise TICAAnalysisError("TICA projection features contain a non-finite value")
    return ((values - center) @ vectors.T).tolist()


_positive_integer = partial(positive_integer, error_type=TICAAnalysisError)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("time_lagged_independent_component_analysis") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise TICAAnalysisError(
            "definitions.time_lagged_independent_component_analysis must be an object"
        )
    required = {
        "feature_source",
        "component_indices",
        "lag_frames",
        "component_count",
        "covariance_regularization",
        "covariance_eigenvalue_cutoff",
        "minimum_pairs_per_segment",
        "maximum_features",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing:
        raise TICAAnalysisError("TICA settings missing: " + ", ".join(missing))
    if unknown:
        raise TICAAnalysisError("TICA settings contain unknown fields: " + ", ".join(unknown))
    if raw["feature_source"] != "common_pca":
        raise TICAAnalysisError("feature_source currently supports only common_pca")
    components = raw["component_indices"]
    if (
        not isinstance(components, list)
        or not components
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in components)
        or len(set(components)) != len(components)
    ):
        raise TICAAnalysisError("component_indices must contain unique positive integers")
    maximum_features = _positive_integer(raw["maximum_features"], "maximum_features")
    if len(components) > maximum_features:
        raise TICAAnalysisError("component_indices exceed maximum_features")
    for label, allow_zero in (
        ("covariance_regularization", True),
        ("covariance_eigenvalue_cutoff", False),
    ):
        value = raw[label]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            raise TICAAnalysisError(f"{label} is outside its finite allowed range")
    component_count = _positive_integer(raw["component_count"], "component_count")
    if component_count > len(components):
        raise TICAAnalysisError("component_count cannot exceed selected feature count")
    return {
        "feature_source": "common_pca",
        "component_indices": list(components),
        "lag_frames": _positive_integer(raw["lag_frames"], "lag_frames"),
        "component_count": component_count,
        "covariance_regularization": float(raw["covariance_regularization"]),
        "covariance_eigenvalue_cutoff": float(raw["covariance_eigenvalue_cutoff"]),
        "minimum_pairs_per_segment": _positive_integer(
            raw["minimum_pairs_per_segment"], "minimum_pairs_per_segment"
        ),
        "maximum_features": maximum_features,
    }


def time_lagged_independent_component_analysis_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    if project.get("sampling_mode") != "UNBIASED_MD":
        raise TICAAnalysisError(
            "TICA kinetic interpretation currently requires sampling_mode=UNBIASED_MD"
        )
    pca_report = common_pca_project(source, hash_content=hash_content)
    if pca_report.get("technical_status") != "complete":
        raise TICAAnalysisError("common_pca feature generation did not complete")
    selected = [int(value) - 1 for value in settings["component_indices"]]
    segment_arrays: List[List[List[float]]] = []
    segment_metadata: List[Dict[str, object]] = []
    evaluated_intervals: List[float] = []
    physical_frame_identities = set()
    for system in pca_report["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            for segment in replica["segments"]:
                projections = segment["projections"]
                if not isinstance(projections, list) or not projections:
                    raise TICAAnalysisError(
                        f"{system_id}/{replica_id}/{segment['segment_id']} has no PCA projections"
                    )
                by_member: Dict[str | None, List[Mapping[str, object]]] = {}
                for projection in projections:
                    if not isinstance(projection, dict):
                        raise TICAAnalysisError("common_pca projection must be an object")
                    member_id = (
                        str(projection["member_id"])
                        if "member_id" in projection else None
                    )
                    by_member.setdefault(member_id, []).append(projection)
                    physical_frame_identities.add((
                        system_id, replica_id, str(segment["segment_id"]),
                        int(projection["source_frame_index"]),
                    ))
                for member_id, member_projections in sorted(
                    by_member.items(), key=lambda item: "" if item[0] is None else item[0]
                ):
                    member_projections = sorted(
                        member_projections,
                        key=lambda row: int(row["source_frame_index"]),
                    )
                    rows: List[List[float]] = []
                    times: List[float] = []
                    for projection in member_projections:
                        scores = projection["scores_angstrom"]
                        if max(selected) >= len(scores):
                            raise TICAAnalysisError(
                                "component_indices exceed components returned by common_pca"
                            )
                        if "time" not in projection:
                            raise TICAAnalysisError("TICA requires physical-time projections")
                        rows.append([float(scores[index]) for index in selected])
                        times.append(float(projection["time"]))
                    pair_count = len(rows) - int(settings["lag_frames"])
                    location = f"{system_id}/{replica_id}/{segment['segment_id']}"
                    if member_id is not None:
                        location += f"/{member_id}"
                    if pair_count < int(settings["minimum_pairs_per_segment"]):
                        raise TICAAnalysisError(
                            f"{location} has {pair_count} lag pairs; minimum is "
                            f"{settings['minimum_pairs_per_segment']}"
                        )
                    intervals = [right - left for left, right in zip(times, times[1:])]
                    if not intervals or any(value <= 0.0 for value in intervals):
                        raise TICAAnalysisError("TICA projection times must be strictly increasing")
                    interval = intervals[0]
                    if any(abs(value - interval) > 1.0e-9 * max(1.0, abs(interval)) for value in intervals[1:]):
                        raise TICAAnalysisError("TICA requires a constant evaluated frame interval")
                    evaluated_intervals.append(interval)
                    segment_arrays.append(rows)
                    segment_metadata.append({
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "segment_id": str(segment["segment_id"]),
                        **({"member_id": member_id} if member_id is not None else {}),
                        "time_unit": str(member_projections[0]["time_unit"]),
                        "source_frame_indices": [
                            int(row["source_frame_index"]) for row in member_projections
                        ],
                        "times": times,
                        "feature_rows": rows,
                        "lag_pair_count": pair_count,
                    })
    interval = evaluated_intervals[0]
    if any(abs(value - interval) > 1.0e-9 * max(1.0, abs(interval)) for value in evaluated_intervals[1:]):
        raise TICAAnalysisError("all TICA segments must share one evaluated physical-time interval")
    time_units = {str(segment["time_unit"]) for segment in segment_metadata}
    if len(time_units) != 1:
        raise TICAAnalysisError("all TICA segments must share one time unit")
    model = fit_tica(
        segment_arrays,
        lag_frames=int(settings["lag_frames"]),
        component_count=int(settings["component_count"]),
        covariance_regularization=float(settings["covariance_regularization"]),
        covariance_eigenvalue_cutoff=float(settings["covariance_eigenvalue_cutoff"]),
    )
    lag_time = interval * int(settings["lag_frames"])
    component_rows = []
    issues = [issue for issue in pca_report.get("issues", []) if isinstance(issue, dict)]
    for index, (eigenvalue, vector, residual) in enumerate(
        zip(model["eigenvalues"], model["eigenvectors"], model["generalized_eigen_residual_norms"]),
        start=1,
    ):
        magnitude = abs(float(eigenvalue))
        timescale = None
        if 0.0 < magnitude < 1.0:
            timescale = -lag_time / math.log(magnitude)
        else:
            issues.append({
                "severity": "warning",
                "code": "TICA_EIGENVALUE_OUTSIDE_TIMESCALE_DOMAIN",
                "location": f"component-{index}",
                "message": f"eigenvalue {eigenvalue} does not define a positive finite implied timescale",
            })
        component_rows.append({
            "component_index": index,
            "eigenvalue": eigenvalue,
            "implied_timescale": timescale,
            "time_unit": next(iter(time_units)),
            "generalized_eigen_residual_norm": residual,
            "loadings": [
                {"source_pca_component": source_index, "loading": loading}
                for source_index, loading in zip(settings["component_indices"], vector)
            ],
        })
    output_segments = []
    for metadata, features in zip(segment_metadata, segment_arrays):
        scores = project_tica(features, model["mean"], model["eigenvectors"])
        output_segments.append({
            "system_id": metadata["system_id"],
            "replica_id": metadata["replica_id"],
            "segment_id": metadata["segment_id"],
            **(
                {"member_id": metadata["member_id"]}
                if "member_id" in metadata else {}
            ),
            "lag_pair_count": metadata["lag_pair_count"],
            "projections": [
                {
                    "source_frame_index": frame_index,
                    **(
                        {"member_id": metadata["member_id"]}
                        if "member_id" in metadata else {}
                    ),
                    "time": time,
                    "time_unit": metadata["time_unit"],
                    "scores": row,
                }
                for frame_index, time, row in zip(
                    metadata["source_frame_indices"], metadata["times"], scores
                )
            ],
        })
    paired_records = [
        {
            "system_id": segment["system_id"],
            "replica_id": segment["replica_id"],
            "segment_id": segment["segment_id"],
            "source_frame_index": projection["source_frame_index"],
            "member_id": projection["member_id"],
            "scores_angstrom": projection["scores"],
        }
        for segment in output_segments if "member_id" in segment
        for projection in segment["projections"]
    ]
    return {
        "module_id": "time_lagged_independent_component_analysis",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": pca_report["project_manifest_sha256"],
        "system_manifest_path": pca_report["system_manifest_path"],
        "system_manifest_sha256": pca_report["system_manifest_sha256"],
        "contract_signature_sha256": pca_report["contract_signature_sha256"],
        "input_content_signature_sha256": pca_report["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "feature_lineage": {
            "module_id": "common_pca",
            "component_indices": settings["component_indices"],
            "common_pca_settings": pca_report["settings"],
        },
        "lag_time": lag_time,
        "time_unit": next(iter(time_units)),
        "pair_count": model["pair_count"],
        "observation_accounting": {
            "source_physical_frame_count": len(physical_frame_identities),
            "symmetry_expanded_observation_count": sum(len(values) for values in segment_arrays),
            "kinetic_trajectory_count": len(segment_arrays),
            "member_observations_are_independent_replicas": False,
        },
        "paired_member_score_correlations": (
            paired_member_score_correlations(paired_records)
            if paired_records else None
        ),
        "retained_covariance_rank": model["retained_covariance_rank"],
        "mean": model["mean"],
        "instantaneous_covariance": model["instantaneous_covariance"],
        "symmetrized_lagged_covariance": model["symmetrized_lagged_covariance"],
        "regularization_absolute": model["regularization_absolute"],
        "components": component_rows,
        "segments": output_segments,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Lag pairs are formed within declared trajectory segments only; segment boundaries are never joined.",
            "Equivalent oligomer members are separate time series; no lag pair ever joins two member identities.",
            "Symmetry-expanded member observations are paired within physical frames and do not increase the independent-replica count.",
            "The estimator uses reversible symmetrized endpoint and lagged covariances on declared common-PCA features.",
            "Current kinetic interpretation is restricted to unbiased MD with one common evaluated physical-time interval.",
            "TICA components and implied timescales require lag, feature, stationarity, convergence, and state-model sensitivity analysis.",
            "Technical completion does not establish metastability, Markovianity, mechanism, or scientific validity.",
        ],
    }


def time_lagged_independent_component_analysis_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return time_lagged_independent_component_analysis_project(
            project_path, hash_content=hash_content
        )
    except (
        ManifestValidationError, PCAAnalysisError, OligomerSymmetryError,
        TICAAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "time_lagged_independent_component_analysis",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "TICA_INVALID", "message": message}
                for message in messages
            ],
        }
