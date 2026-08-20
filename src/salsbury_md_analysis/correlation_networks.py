"""Thresholded correlation-network analysis without graph-library dependencies."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .dccm import dccm_project
from .manifests import ManifestValidationError, load_json
from .validation import positive_integer


class CorrelationNetworkError(ValueError):
    """Raised when a matrix cannot support the declared network analysis."""


def correlation_network(
    matrix: Sequence[Sequence[object]],
    absolute_threshold: float,
    include_negative: bool = True,
    maximum_nodes: int = 2000,
) -> Dict[str, object]:
    """Build an undirected threshold network and basic topology diagnostics."""

    size = len(matrix)
    maximum_nodes = positive_integer(maximum_nodes, "maximum_nodes")
    if size < 1 or size > maximum_nodes or any(len(row) != size for row in matrix):
        raise CorrelationNetworkError("matrix must be square and within maximum_nodes")
    if not math.isfinite(absolute_threshold) or not 0.0 <= absolute_threshold <= 1.0:
        raise CorrelationNetworkError("absolute_threshold must be between 0 and 1")
    adjacency = [set() for _ in range(size)]
    strengths = [0.0] * size
    edges = []
    undefined = 0
    for left in range(size - 1):
        for right in range(left + 1, size):
            value = matrix[left][right]
            if value is None:
                undefined += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CorrelationNetworkError("matrix entries must be finite numbers or null")
            weight = float(value)
            if not math.isfinite(weight):
                raise CorrelationNetworkError("matrix contains a non-finite value")
            if abs(weight) < absolute_threshold or (weight < 0.0 and not include_negative):
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
            strengths[left] += abs(weight)
            strengths[right] += abs(weight)
            edges.append({
                "node_i": left,
                "node_j": right,
                "weight": weight,
                "absolute_weight": abs(weight),
                "sign": "positive" if weight >= 0.0 else "negative",
            })
    components: List[List[int]] = []
    unseen = set(range(size))
    while unseen:
        start = min(unseen)
        stack = [start]
        component = []
        unseen.remove(start)
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = sorted(adjacency[node] & unseen, reverse=True)
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values))
    local_clustering = []
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            local_clustering.append(0.0)
            continue
        links = sum(
            right in adjacency[left]
            for left in neighbors for right in neighbors if left < right
        )
        local_clustering.append(2.0 * links / (degree * (degree - 1)))
    possible = size * (size - 1) / 2
    return {
        "node_count": size,
        "edge_count": len(edges),
        "edge_density": len(edges) / possible if possible else 0.0,
        "undefined_pair_count": undefined,
        "absolute_threshold": float(absolute_threshold),
        "include_negative": bool(include_negative),
        "edges": edges,
        "node_degrees": [len(neighbors) for neighbors in adjacency],
        "node_absolute_strengths": strengths,
        "local_clustering_coefficients": local_clustering,
        "connected_components": components,
        "largest_component_size": len(components[0]),
    }


def correlation_profile_clustering(
    matrix: Sequence[Sequence[object]],
    minimum_cluster_size: int,
    minimum_samples: int,
    input_mode: str = "profiles",
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    maximum_nodes: int = 2000,
) -> Dict[str, object]:
    """Cluster correlation profiles with the optional reference HDBSCAN package."""

    size = len(matrix)
    maximum_nodes = positive_integer(maximum_nodes, "maximum_nodes")
    if size < 2 or size > maximum_nodes or any(len(row) != size for row in matrix):
        raise CorrelationNetworkError("matrix must be square and within maximum_nodes")
    values = np.asarray(matrix, dtype=object)
    if any(value is None or isinstance(value, bool) for value in values.flat):
        raise CorrelationNetworkError(
            "correlation-profile clustering requires a complete numeric matrix"
        )
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise CorrelationNetworkError("correlation matrix entries must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise CorrelationNetworkError("correlation matrix contains non-finite values")
    minimum_cluster_size = positive_integer(
        minimum_cluster_size, "minimum_cluster_size"
    )
    minimum_samples = positive_integer(minimum_samples, "minimum_samples")
    if input_mode not in {"profiles", "absolute_similarity"}:
        raise CorrelationNetworkError(
            "input_mode must be profiles or absolute_similarity"
        )
    if cluster_selection_method not in {"eom", "leaf"}:
        raise CorrelationNetworkError("cluster_selection_method must be eom or leaf")
    try:
        package = importlib.import_module("hdbscan")
    except ImportError as exc:
        raise CorrelationNetworkError(
            "optional dependency hdbscan is unavailable; install the hdbscan extra"
        ) from exc
    if not hasattr(package, "HDBSCAN"):
        raise CorrelationNetworkError("imported hdbscan package does not expose HDBSCAN")
    if input_mode == "absolute_similarity":
        model_input = 1.0 - np.clip(np.abs(numeric), 0.0, 1.0)
        np.fill_diagonal(model_input, 0.0)
        metric = "precomputed"
    else:
        model_input = numeric
        metric = "euclidean"
    model = package.HDBSCAN(
        min_cluster_size=minimum_cluster_size,
        min_samples=minimum_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=bool(allow_single_cluster),
        core_dist_n_jobs=1,
    )
    raw_labels = [int(value) for value in model.fit_predict(model_input)]
    if len(raw_labels) != size:
        raise CorrelationNetworkError("HDBSCAN returned an assignment count mismatch")
    members = {
        label: [index for index, assigned in enumerate(raw_labels) if assigned == label]
        for label in sorted(set(raw_labels))
        if label >= 0
    }
    ordered = sorted(members, key=lambda label: members[label])
    remap = {label: index for index, label in enumerate(ordered)}
    labels = [remap[label] if label >= 0 else -1 for label in raw_labels]
    return {
        "labels": labels,
        "cluster_count": len(ordered),
        "cluster_sizes": [labels.count(label) for label in range(len(ordered))],
        "noise_count": labels.count(-1),
        "retained_fraction": sum(label >= 0 for label in labels) / len(labels),
        "input_mode": input_mode,
        "distance_contract": (
            "Euclidean distance between complete signed correlation profiles"
            if input_mode == "profiles"
            else "one minus absolute correlation similarity"
        ),
        "minimum_cluster_size": minimum_cluster_size,
        "minimum_samples": minimum_samples,
        "cluster_selection_method": cluster_selection_method,
        "allow_single_cluster": bool(allow_single_cluster),
        "dependency": "hdbscan.HDBSCAN",
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("correlation_networks") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise CorrelationNetworkError("definitions.correlation_networks must be an object")
    required = {"matrix_kinds", "absolute_threshold", "include_negative", "maximum_nodes"}
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | {"profile_clustering"}))
    if missing or unknown:
        raise CorrelationNetworkError(
            "correlation-network settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    kinds = raw["matrix_kinds"]
    allowed = {"frame_pooled_dccm", "difference_from_reference_dccm"}
    if not isinstance(kinds, list) or not kinds or any(value not in allowed for value in kinds):
        raise CorrelationNetworkError("matrix_kinds contains an unsupported matrix")
    profile = raw.get("profile_clustering")
    if profile is not None:
        expected = {
            "input_mode", "minimum_cluster_size", "minimum_samples",
            "cluster_selection_method", "allow_single_cluster",
        }
        if not isinstance(profile, dict) or set(profile) != expected:
            raise CorrelationNetworkError(
                "profile_clustering fields do not match the required contract"
            )
        if profile["input_mode"] not in {"profiles", "absolute_similarity"}:
            raise CorrelationNetworkError(
                "profile_clustering.input_mode is unsupported"
            )
        for field in ("minimum_cluster_size", "minimum_samples"):
            positive_integer(profile[field], f"profile_clustering.{field}")
        if profile["cluster_selection_method"] not in {"eom", "leaf"}:
            raise CorrelationNetworkError(
                "profile_clustering.cluster_selection_method must be eom or leaf"
            )
        if not isinstance(profile["allow_single_cluster"], bool):
            raise CorrelationNetworkError(
                "profile_clustering.allow_single_cluster must be boolean"
            )
    return dict(raw)


def _dccm_observation_accounting(
    dccm: Mapping[str, object],
) -> Dict[str, object]:
    """Recover the exact physical-frame workload inherited from DCCM.

    Correlation networks transform already pooled DCCM matrices and therefore
    do not create member-expanded trajectory observations of their own.  The
    DCCM frame-selection contract is authoritative, while the independently
    summed replica/segment counts protect the execution sidecar from accepting
    a partially evaluated or internally inconsistent upstream report.
    """

    frame_selection = dccm.get("frame_selection")
    systems = dccm.get("systems")
    if not isinstance(frame_selection, dict) or not isinstance(systems, list):
        raise CorrelationNetworkError(
            "DCCM report lacks exact frame-selection or system accounting"
        )
    source_count = frame_selection.get("source_frame_count")
    selected_count = frame_selection.get("selected_frame_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (source_count, selected_count)
    ):
        raise CorrelationNetworkError(
            "DCCM frame-selection counts must be positive integers"
        )
    evaluated_count = 0
    segment_count = 0
    for system in systems:
        replicas = system.get("replicas") if isinstance(system, dict) else None
        if not isinstance(replicas, list):
            raise CorrelationNetworkError(
                "DCCM system report lacks replica-level frame accounting"
            )
        for replica in replicas:
            segments = replica.get("segments") if isinstance(replica, dict) else None
            if not isinstance(segments, list):
                raise CorrelationNetworkError(
                    "DCCM replica report lacks segment-level frame accounting"
                )
            for segment in segments:
                count = (
                    segment.get("evaluated_frame_count")
                    if isinstance(segment, dict) else None
                )
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise CorrelationNetworkError(
                        "DCCM segment evaluated-frame count is invalid"
                    )
                evaluated_count += count
                segment_count += 1
    if segment_count < 1 or evaluated_count != selected_count:
        raise CorrelationNetworkError(
            "DCCM selected/evaluated frame accounting mismatch: "
            f"selected={selected_count}, evaluated={evaluated_count}"
        )
    return {
        "source_physical_frame_count": source_count,
        "selected_physical_frame_count": selected_count,
        "symmetry_expanded_observation_count": selected_count,
        "subsampling_triggered": selected_count < source_count,
        "accounting_basis": (
            "exact DCCM replica/segment evaluated-frame sum; the network "
            "transforms pooled matrices and does not multiply trajectory observations"
        ),
    }


def correlation_networks_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    dccm = dccm_project(source, hash_content=hash_content)
    observation_accounting = _dccm_observation_accounting(dccm)
    reports = []
    for system in dccm["systems"]:
        matrices = []
        for kind in settings["matrix_kinds"]:
            payload = system.get(kind)
            if not isinstance(payload, dict) or not isinstance(payload.get("matrix"), list):
                continue
            matrix_report = {
                "matrix_kind": kind,
                "network": correlation_network(
                    payload["matrix"],
                    float(settings["absolute_threshold"]),
                    bool(settings["include_negative"]),
                    positive_integer(settings["maximum_nodes"], "maximum_nodes"),
                ),
            }
            profile = settings.get("profile_clustering")
            if isinstance(profile, dict):
                matrix_report["profile_clustering"] = correlation_profile_clustering(
                    payload["matrix"],
                    positive_integer(profile["minimum_cluster_size"], "minimum_cluster_size"),
                    positive_integer(profile["minimum_samples"], "minimum_samples"),
                    str(profile["input_mode"]),
                    str(profile["cluster_selection_method"]),
                    bool(profile["allow_single_cluster"]),
                    positive_integer(settings["maximum_nodes"], "maximum_nodes"),
                )
            matrices.append(matrix_report)
        reports.append({"system_id": system["system_id"], "matrices": matrices})
    return {
        "module_id": "correlation_networks",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": dccm["project_manifest_sha256"],
        "system_manifest_path": dccm["system_manifest_path"],
        "system_manifest_sha256": dccm["system_manifest_sha256"],
        "input_content_signature_sha256": dccm["input_content_signature_sha256"],
        "analysis_atoms": dccm["analysis_atoms"],
        "observation_accounting": observation_accounting,
        "settings": settings,
        "systems": reports,
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "limitations": [
            "Network topology depends directly on the declared absolute threshold.",
            "Edges encode correlation or correlation change, not direct physical interactions or causality.",
            "Threshold and atom-selection sensitivity must be reported before interpretation.",
            "Correlation-profile HDBSCAN is a noise-aware sensitivity partition and does not establish domains, pathways, or mechanism.",
        ],
    }


def correlation_networks_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return correlation_networks_project(project_path, hash_content=hash_content)
    except (CorrelationNetworkError, ManifestValidationError, OSError, KeyError, ValueError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "correlation_networks",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "CORRELATION_NETWORK_INVALID", "message": message}
                for message in messages
            ],
        }
