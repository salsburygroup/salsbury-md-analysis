"""Experimental, frame-identity-safe trajectory reweighting diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .manifests import (
    ManifestValidationError, load_json, resolve_manifest_path, sha256_file,
)
from .pca import common_pca_project


class ReweightingError(ValueError):
    """Raised when frame weights or their alignment contract are invalid."""


FrameIdentity = Tuple[str, str, str, str | None, int]


def normalize_log_weights(
    log_weights: Sequence[float],
    *,
    minimum_kish_effective_sample_size: float = 1.0,
    minimum_kish_ratio: float = 0.0,
    maximum_single_frame_weight: float = 1.0,
) -> Dict[str, object]:
    """Normalize log weights stably and report independent reliability metrics."""

    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ReweightingError("log_weights must be a nonempty finite vector")
    gates = (
        (minimum_kish_effective_sample_size, "minimum_kish_effective_sample_size", 1.0, math.inf),
        (minimum_kish_ratio, "minimum_kish_ratio", 0.0, 1.0),
        (maximum_single_frame_weight, "maximum_single_frame_weight", 0.0, 1.0),
    )
    for value, name, lower, upper in gates:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < lower
            or float(value) > upper
        ):
            raise ReweightingError(f"{name} is outside its valid range")
    shifted = values - float(np.max(values))
    unnormalized = np.exp(shifted)
    weights = unnormalized / float(unnormalized.sum())
    kish = 1.0 / float(np.dot(weights, weights))
    entropy = -float(np.sum(weights * np.log(weights)))
    entropy_effective = math.exp(entropy)
    count = len(weights)
    maximum = float(np.max(weights))
    ordered = np.sort(weights)[::-1]
    top_count = max(1, math.ceil(0.01 * count))
    gate_results = {
        "minimum_kish_effective_sample_size": {
            "observed": kish,
            "threshold": float(minimum_kish_effective_sample_size),
            "passed": kish >= float(minimum_kish_effective_sample_size),
        },
        "minimum_kish_ratio": {
            "observed": kish / count,
            "threshold": float(minimum_kish_ratio),
            "passed": kish / count >= float(minimum_kish_ratio),
        },
        "maximum_single_frame_weight": {
            "observed": maximum,
            "threshold": float(maximum_single_frame_weight),
            "passed": maximum <= float(maximum_single_frame_weight),
        },
    }
    valid = all(bool(row["passed"]) for row in gate_results.values())
    return {
        "normalized_weights": weights.tolist(),
        "normalization": "log-sum-exp after subtracting the maximum log weight",
        "weight_sum": float(weights.sum()),
        "frame_count": count,
        "kish_effective_sample_size": kish,
        "kish_ratio": kish / count,
        "entropy_effective_sample_size": entropy_effective,
        "entropy_effective_sample_ratio": entropy_effective / count,
        "relative_entropy_from_uniform_nats": math.log(count) - entropy,
        "maximum_single_frame_weight": maximum,
        "top_one_percent_weight_mass": float(ordered[:top_count].sum()),
        "log_weight_range": float(np.max(values) - np.min(values)),
        "gate_results": gate_results,
        "reweighting_validity_status": "passed" if valid else "failed",
    }


def weighted_moments(
    observations: Sequence[Sequence[float]], normalized_weights: Sequence[float]
) -> Dict[str, object]:
    """Return explicitly population-normalized weighted vector moments."""

    values = np.asarray(observations, dtype=float)
    weights = np.asarray(normalized_weights, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ReweightingError("observations must be a nonempty 2D matrix")
    if not np.isfinite(values).all():
        raise ReweightingError("observations contain non-finite values")
    if (
        weights.ndim != 1
        or len(weights) != len(values)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    ):
        raise ReweightingError(
            "normalized_weights must be nonnegative, aligned, and sum to one"
        )
    mean = weights @ values
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    return {
        "weighted_mean": mean.tolist(),
        "weighted_population_covariance": covariance.tolist(),
        "weighted_population_sd": np.sqrt(np.clip(np.diag(covariance), 0.0, None)).tolist(),
        "unweighted_mean": values.mean(axis=0).tolist(),
        "unweighted_population_sd": values.std(axis=0).tolist(),
        "covariance_denominator": "normalized weights summing to one",
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("trajectory_reweighting") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise ReweightingError("definitions.trajectory_reweighting must be an object")
    required = {
        "observable_source", "weights_path", "normalization_scope",
        "minimum_kish_effective_sample_size", "minimum_kish_ratio",
        "maximum_single_frame_weight",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise ReweightingError(
            "trajectory-reweighting settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    if raw["observable_source"] != "common_pca":
        raise ReweightingError("observable_source must be common_pca")
    if not isinstance(raw["weights_path"], str) or not raw["weights_path"].strip():
        raise ReweightingError("weights_path must be a nonempty path")
    if raw["normalization_scope"] != "per_system":
        raise ReweightingError("normalization_scope must be per_system")
    # Reuse the public kernel for exact range validation of all three gates.
    normalize_log_weights(
        [0.0],
        minimum_kish_effective_sample_size=float(
            raw["minimum_kish_effective_sample_size"]
        ),
        minimum_kish_ratio=float(raw["minimum_kish_ratio"]),
        maximum_single_frame_weight=float(raw["maximum_single_frame_weight"]),
    )
    return dict(raw)


def _identity(row: Mapping[str, object], location: str) -> FrameIdentity:
    fields = ("system_id", "replica_id", "segment_id")
    if any(not isinstance(row.get(field), str) or not str(row[field]).strip() for field in fields):
        raise ReweightingError(f"{location} has an invalid system/replica/segment identity")
    frame = row.get("source_frame_index")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ReweightingError(f"{location}.source_frame_index must be nonnegative")
    member = row.get("member_id")
    if member is not None and (not isinstance(member, str) or not member.strip()):
        raise ReweightingError(f"{location}.member_id must be null or nonempty")
    return (
        str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"]),
        str(member) if member is not None else None, frame,
    )


def _load_weight_rows(path: Path) -> Dict[FrameIdentity, float]:
    payload = load_json(path)
    if set(payload) != {"weight_schema", "weight_semantics", "rows"}:
        raise ReweightingError(
            "weight file must contain exactly weight_schema, weight_semantics, and rows"
        )
    if payload["weight_schema"] != "salsbury-frame-log-weights-v1":
        raise ReweightingError("weight_schema is unsupported")
    if payload["weight_semantics"] != "log_unnormalized_target_over_source_probability":
        raise ReweightingError("weight_semantics is unsupported")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ReweightingError("weight rows must be a nonempty array")
    result: Dict[FrameIdentity, float] = {}
    allowed = {
        "system_id", "replica_id", "segment_id", "member_id",
        "source_frame_index", "log_weight",
    }
    required = allowed.difference({"member_id"})
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row) or set(row).difference(allowed):
            raise ReweightingError(f"weights.rows[{index}] fields are invalid")
        identity = _identity(row, f"weights.rows[{index}]")
        value = row["log_weight"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ReweightingError(f"weights.rows[{index}].log_weight must be finite")
        if identity in result:
            raise ReweightingError(f"duplicate weight identity: {identity}")
        result[identity] = float(value)
    return result


def _projection_rows(pca: Mapping[str, object]) -> List[Tuple[FrameIdentity, List[float]]]:
    output: List[Tuple[FrameIdentity, List[float]]] = []
    systems = pca.get("systems")
    if not isinstance(systems, list):
        raise ReweightingError("common PCA report contains no systems")
    seen = set()
    for system in systems:
        if not isinstance(system, dict):
            raise ReweightingError("common PCA system entry is invalid")
        system_id = str(system["system_id"])
        replicas = system.get("replicas")
        if not isinstance(replicas, list):
            raise ReweightingError("common PCA system contains no replicas")
        for replica in replicas:
            replica_id = str(replica["replica_id"])
            segments = replica.get("segments")
            if not isinstance(segments, list):
                raise ReweightingError("common PCA replica contains no segments")
            for segment in segments:
                segment_id = str(segment["segment_id"])
                projections = segment.get("projections")
                if not isinstance(projections, list):
                    raise ReweightingError("common PCA segment contains no projections")
                for projection in projections:
                    row = {
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "segment_id": segment_id,
                        "member_id": projection.get("member_id"),
                        "source_frame_index": projection.get("source_frame_index"),
                    }
                    identity = _identity(row, "common PCA projection")
                    scores = projection.get("scores_angstrom")
                    if not isinstance(scores, list) or not scores:
                        raise ReweightingError("common PCA projection lacks scores")
                    values = [float(value) for value in scores]
                    if not all(math.isfinite(value) for value in values):
                        raise ReweightingError("common PCA projection scores are non-finite")
                    if identity in seen:
                        raise ReweightingError(f"duplicate common PCA identity: {identity}")
                    seen.add(identity)
                    output.append((identity, values))
    if not output:
        raise ReweightingError("common PCA report contains no projections")
    return output


def trajectory_reweighting_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    weights_path = resolve_manifest_path(str(settings["weights_path"]), source)
    weight_by_identity = _load_weight_rows(weights_path)
    pca = common_pca_project(source, hash_content=hash_content)
    if pca.get("technical_status") != "complete":
        raise ReweightingError("common PCA report is not technically complete")
    projections = _projection_rows(pca)
    projection_identities = {identity for identity, _ in projections}
    weight_identities = set(weight_by_identity)
    missing = projection_identities.difference(weight_identities)
    extra = weight_identities.difference(projection_identities)
    if missing or extra:
        raise ReweightingError(
            "weight/projection identity mismatch: "
            f"missing_weights={len(missing)}; extra_weights={len(extra)}"
        )
    by_system: Dict[str, List[Tuple[FrameIdentity, List[float]]]] = {}
    for identity, scores in projections:
        by_system.setdefault(identity[0], []).append((identity, scores))
    reports = []
    failed_systems = []
    for system_id, rows in by_system.items():
        diagnostics = normalize_log_weights(
            [weight_by_identity[identity] for identity, _ in rows],
            minimum_kish_effective_sample_size=float(
                settings["minimum_kish_effective_sample_size"]
            ),
            minimum_kish_ratio=float(settings["minimum_kish_ratio"]),
            maximum_single_frame_weight=float(settings["maximum_single_frame_weight"]),
        )
        moments = weighted_moments(
            [scores for _, scores in rows], diagnostics["normalized_weights"]
        )
        if diagnostics["reweighting_validity_status"] != "passed":
            failed_systems.append(system_id)
        reports.append({
            "system_id": system_id,
            "diagnostics": diagnostics,
            "common_pca_score_moments": moments,
            "frame_weights": [
                {
                    "system_id": identity[0],
                    "replica_id": identity[1],
                    "segment_id": identity[2],
                    **({"member_id": identity[3]} if identity[3] is not None else {}),
                    "source_frame_index": identity[4],
                    "log_weight": weight_by_identity[identity],
                    "normalized_weight": diagnostics["normalized_weights"][index],
                }
                for index, (identity, _) in enumerate(rows)
            ],
        })
    issues = [
        {
            "severity": "warning",
            "code": "REWEIGHTING_RELIABILITY_GATE_FAILED",
            "message": (
                "one or more declared ESS/concentration gates failed for systems: "
                + ", ".join(failed_systems)
                + "; downstream weighted thermodynamic interpretation is blocked"
            ),
        }
    ] if failed_systems else []
    pca_accounting = pca.get("observation_accounting")
    projection_selection = pca.get("projection_frame_selection")
    physical_frames = (
        pca_accounting.get("selected_physical_frame_count")
        if isinstance(pca_accounting, dict)
        else projection_selection.get("selected_frame_count")
        if isinstance(projection_selection, dict)
        else None
    )
    if isinstance(physical_frames, bool) or not isinstance(physical_frames, int):
        raise ReweightingError(
            "common PCA report lacks exact projection-frame accounting"
        )
    return {
        "module_id": "trajectory_reweighting",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "reweighting_validity_status": "failed" if failed_systems else "passed",
        "weighted_thermodynamic_interpretation_allowed": not failed_systems,
        "project_manifest_path": str(source),
        "project_manifest_sha256": pca["project_manifest_sha256"],
        "system_manifest_path": pca["system_manifest_path"],
        "system_manifest_sha256": pca["system_manifest_sha256"],
        "input_content_signature_sha256": pca["input_content_signature_sha256"],
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "settings": settings,
        "observation_accounting": {
            "selected_physical_frame_count": physical_frames,
            "symmetry_expanded_observation_count": len(projections),
            "accounting_basis": "exact matched common-PCA projection and weight identities",
        },
        "systems": reports,
        "error_count": 0,
        "warning_count": len(issues),
        "issues": issues,
        "limitations": [
            "This module consumes externally derived log weights; it does not infer bias potentials, MBAR free energies, or maximum-entropy multipliers.",
            "Every weight is joined to one exact system/replica/segment/member/frame identity; positional row-order matching is prohibited.",
            "Weights are normalized independently within each system and are not comparable probabilities across systems.",
            "Kish and entropy effective sample sizes diagnose weight concentration but do not prove phase-space overlap, equilibration, convergence, or model correctness.",
            "Weighted common-PCA moments are descriptive. This module does not silently authorize biased or enhanced-sampling FES conversion.",
            "Frame correlation is not removed by the weight-only effective sample-size diagnostics.",
            "Technical completion and passing reliability gates do not establish scientific validity.",
        ],
    }


def trajectory_reweighting_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return trajectory_reweighting_project(project_path, hash_content=hash_content)
    except (
        ReweightingError, ManifestValidationError, OSError, KeyError,
        TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "trajectory_reweighting",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "reweighting_validity_status": "not evaluated",
            "weighted_thermodynamic_interpretation_allowed": False,
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "TRAJECTORY_REWEIGHTING_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
