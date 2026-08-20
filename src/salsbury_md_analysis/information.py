"""Nonlinear dependence matrices from explicit trajectory feature tables."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .manifests import ManifestValidationError, load_json
from .pca import PCAAnalysisError, common_pca_project
from .validation import integer_at_least


class InformationAnalysisError(ValueError):
    """Raised when an information estimator is undefined or under-specified."""


def _quantile_assignments(values: np.ndarray, bin_count: int) -> Tuple[np.ndarray, int]:
    if np.all(values == values[0]):
        return np.zeros(len(values), dtype=int), 1
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles, method="linear"))
    assignments = np.searchsorted(edges, values, side="right")
    return assignments, len(edges) + 1


def _entropy(assignments: np.ndarray, bin_count: int) -> float:
    counts = np.bincount(assignments, minlength=bin_count).astype(float)
    probabilities = counts[counts > 0.0] / len(assignments)
    return float(-np.sum(probabilities * np.log(probabilities)))


def mutual_information_matrices(
    features: Sequence[Sequence[float]],
    *,
    bin_count: int,
    minimum_observations: int,
) -> Dict[str, object]:
    """Return empirical quantile-histogram MI and derived dependence matrices.

    Mutual information is in natural-log units. Normalized MI is
    ``I/sqrt(Hx*Hy)``. The scalar generalized-correlation transform is
    ``sqrt(1-exp(-2I))``. Undefined normalized values caused by constant
    features are represented as ``None``, never as zero.
    """

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise InformationAnalysisError("features must be an N by D array with D at least 2")
    if not np.isfinite(values).all():
        raise InformationAnalysisError("features contain a non-finite value")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 2:
        raise InformationAnalysisError("bin_count must be an integer of at least 2")
    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, int)
        or minimum_observations <= 0
    ):
        raise InformationAnalysisError("minimum_observations must be a positive integer")
    if values.shape[0] < minimum_observations:
        raise InformationAnalysisError(
            f"feature table has {values.shape[0]} observations; minimum is {minimum_observations}"
        )
    if bin_count > values.shape[0] // 2:
        raise InformationAnalysisError(
            "bin_count must leave at least two observations per requested bin"
        )

    assignments = []
    actual_bins = []
    entropies = []
    for feature_index in range(values.shape[1]):
        assigned, actual = _quantile_assignments(values[:, feature_index], bin_count)
        assignments.append(assigned)
        actual_bins.append(actual)
        entropies.append(_entropy(assigned, actual))
    size = values.shape[1]
    mutual_information = [[0.0] * size for _ in range(size)]
    normalized: List[List[float | None]] = [[None] * size for _ in range(size)]
    generalized: List[List[float | None]] = [[None] * size for _ in range(size)]
    for left in range(size):
        mutual_information[left][left] = entropies[left]
        if entropies[left] > 0.0:
            normalized[left][left] = 1.0
            generalized[left][left] = 1.0
        for right in range(left + 1, size):
            joint = np.zeros((actual_bins[left], actual_bins[right]), dtype=float)
            np.add.at(joint, (assignments[left], assignments[right]), 1.0)
            joint /= values.shape[0]
            left_probabilities = joint.sum(axis=1)
            right_probabilities = joint.sum(axis=0)
            information = 0.0
            for left_bin, right_bin in zip(*np.nonzero(joint)):
                probability = joint[left_bin, right_bin]
                information += float(
                    probability
                    * math.log(
                        probability
                        / (left_probabilities[left_bin] * right_probabilities[right_bin])
                    )
                )
            information = max(0.0, information)
            mutual_information[left][right] = mutual_information[right][left] = information
            denominator = math.sqrt(entropies[left] * entropies[right])
            if denominator > 0.0:
                nmi = min(1.0, max(0.0, information / denominator))
                correlation = math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * information)))
                normalized[left][right] = normalized[right][left] = nmi
                generalized[left][right] = generalized[right][left] = correlation
    return {
        "observation_count": int(values.shape[0]),
        "feature_count": int(values.shape[1]),
        "requested_bin_count": bin_count,
        "actual_bin_counts": actual_bins,
        "marginal_entropies_nats": entropies,
        "mutual_information_nats": mutual_information,
        "normalized_mutual_information": normalized,
        "generalized_correlation": generalized,
        "estimator": "empirical quantile-histogram plug-in estimator",
    }


_positive_integer = partial(integer_at_least, error_type=InformationAnalysisError)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("generalized_correlation_and_information") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise InformationAnalysisError(
            "definitions.generalized_correlation_and_information must be an object"
        )
    required = {
        "feature_source",
        "component_indices",
        "bin_count",
        "minimum_observations_per_replica",
        "maximum_features",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing:
        raise InformationAnalysisError("information settings missing: " + ", ".join(missing))
    if unknown:
        raise InformationAnalysisError("information settings contain unknown fields: " + ", ".join(unknown))
    if raw["feature_source"] != "common_pca":
        raise InformationAnalysisError("feature_source currently supports only common_pca")
    components = raw["component_indices"]
    if (
        not isinstance(components, list)
        or len(components) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in components)
        or len(set(components)) != len(components)
    ):
        raise InformationAnalysisError(
            "component_indices must contain at least two unique positive integers"
        )
    maximum = _positive_integer(raw["maximum_features"], "maximum_features", 2)
    if len(components) > maximum:
        raise InformationAnalysisError("component_indices exceed maximum_features")
    return {
        "feature_source": "common_pca",
        "component_indices": list(components),
        "bin_count": _positive_integer(raw["bin_count"], "bin_count", 2),
        "minimum_observations_per_replica": _positive_integer(
            raw["minimum_observations_per_replica"],
            "minimum_observations_per_replica",
        ),
        "maximum_features": maximum,
    }


def _feature_tables(
    pca_report: Mapping[str, object], component_indices: Sequence[int]
) -> Tuple[List[Dict[str, object]], Dict[str, List[List[float]]]]:
    zero_based = [value - 1 for value in component_indices]
    replicas = []
    systems: Dict[str, List[List[float]]] = {}
    for system in pca_report["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            rows = []
            for segment in replica["segments"]:
                for projection in segment["projections"]:
                    scores = projection["scores_angstrom"]
                    if max(zero_based) >= len(scores):
                        raise InformationAnalysisError(
                            "component_indices exceed components returned by common_pca"
                        )
                    rows.append([float(scores[index]) for index in zero_based])
            replicas.append({
                "system_id": system_id,
                "replica_id": str(replica["replica_id"]),
                "features": rows,
            })
            systems.setdefault(system_id, []).extend(rows)
    return replicas, systems


def generalized_correlation_and_information_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    pca_report = common_pca_project(source, hash_content=hash_content)
    if pca_report.get("technical_status") != "complete":
        raise InformationAnalysisError("common_pca feature generation did not complete")
    replica_tables, system_tables = _feature_tables(
        pca_report, [int(value) for value in settings["component_indices"]]
    )
    minimum_observations = int(settings["minimum_observations_per_replica"])
    issues = [issue for issue in pca_report.get("issues", []) if isinstance(issue, dict)]
    replica_reports = []
    for table in replica_tables:
        observation_count = len(table["features"])
        if observation_count < minimum_observations:
            replica_reports.append({
                "system_id": table["system_id"],
                "replica_id": table["replica_id"],
                "technical_status": "skipped",
                "scientific_status": "not evaluated",
                "observation_count": observation_count,
                "minimum_required_observation_count": minimum_observations,
                "reason": "insufficient observations for a replica-level estimate",
            })
            issues.append({
                "severity": "warning",
                "code": "REPLICA_INFORMATION_ESTIMATE_SKIPPED",
                "location": f"{table['system_id']}/{table['replica_id']}",
                "message": (
                    f"replica-level information estimate skipped: {observation_count} "
                    f"observations; minimum is {minimum_observations}; the replica "
                    "remains included in its pooled system estimate"
                ),
            })
            continue
        matrix = mutual_information_matrices(
            table["features"],
            bin_count=int(settings["bin_count"]),
            minimum_observations=minimum_observations,
        )
        replica_reports.append({
            "system_id": table["system_id"],
            "replica_id": table["replica_id"],
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            **matrix,
        })
    system_reports = []
    for system_id, features in sorted(system_tables.items()):
        observation_count = len(features)
        if observation_count < minimum_observations:
            system_reports.append({
                "system_id": system_id,
                "technical_status": "skipped",
                "scientific_status": "not evaluated",
                "observation_count": observation_count,
                "minimum_required_observation_count": minimum_observations,
                "reason": "insufficient pooled observations for a system-level estimate",
            })
            issues.append({
                "severity": "warning",
                "code": "SYSTEM_INFORMATION_ESTIMATE_SKIPPED",
                "location": system_id,
                "message": (
                    f"system-level information estimate skipped: {observation_count} "
                    f"pooled observations; minimum is {minimum_observations}"
                ),
            })
            continue
        system_reports.append({
            "system_id": system_id,
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            **mutual_information_matrices(
                features,
                bin_count=int(settings["bin_count"]),
                minimum_observations=minimum_observations,
            ),
        })
    if any(
        any(value == 1 for value in report["actual_bin_counts"])
        for report in replica_reports if report.get("technical_status") == "complete"
    ):
        issues.append({
            "severity": "warning",
            "code": "CONSTANT_INFORMATION_FEATURE",
            "location": str(source),
            "message": "at least one feature is constant; its normalized dependence entries are null",
        })
    return {
        "module_id": "generalized_correlation_and_information",
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
        "replicas": replica_reports,
        "systems": system_reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Mutual information is estimated from declared quantile histograms and must be tested across bin counts and sample sizes.",
            "The current feature provider uses scalar common-PCA projections; residue-vector and distance-feature providers remain explicit extensions.",
            "Normalized MI and generalized correlation are symmetric dependence measures and do not establish direction, causality, or mechanism.",
            "Frames are correlated observations; plug-in MI uncertainty is not inferred from frame count.",
        ],
    }


def generalized_correlation_and_information_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return generalized_correlation_and_information_project(
            project_path, hash_content=hash_content
        )
    except (
        ManifestValidationError,
        PCAAnalysisError,
        InformationAnalysisError,
        OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "generalized_correlation_and_information",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "INFORMATION_INVALID", "message": message}
                for message in messages
            ],
        }
