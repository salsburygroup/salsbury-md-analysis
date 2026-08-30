"""Partitioned Local Depth community analysis on bounded trajectory samples."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .alternative_clustering import (
    AlternativeClusteringError,
    partitioned_local_depths,
)
from .clustering import ClusteringAnalysisError, _standardize
from .feature_matrix import load_feature_matrix, parse_feature_selection
from .frame_sampling import integer_stride_indices
from .manifests import ManifestValidationError, load_json
from .provenance import stable_json_sha256
from .trajectory_features import TrajectoryFeatureError
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class PaLDCommunityError(ValueError):
    """Raised when a PaLD community-analysis contract is invalid."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("pald_community_analysis")
        if isinstance(definitions, dict) else None
    )
    if not isinstance(raw, dict):
        raise PaLDCommunityError(
            "definitions.pald_community_analysis must be an object"
        )
    required = {
        "feature_source", "standardize_features", "maximum_observations",
        "community_msm_enabled", "maximum_reported_intercommunity_ties",
    }
    missing = sorted(required.difference(raw))
    allowed = required | {"component_indices", "trajectory_feature_columns"}
    unknown = sorted(set(raw).difference(allowed))
    if missing:
        raise PaLDCommunityError(
            "definitions.pald_community_analysis is missing required fields: "
            + ", ".join(missing)
        )
    if unknown:
        raise PaLDCommunityError(
            "definitions.pald_community_analysis contains unknown fields: "
            + ", ".join(unknown)
        )
    feature = parse_feature_selection(raw, PaLDCommunityError)
    if not isinstance(raw["standardize_features"], bool):
        raise PaLDCommunityError("standardize_features must be boolean")
    if not isinstance(raw["community_msm_enabled"], bool):
        raise PaLDCommunityError("community_msm_enabled must be boolean")
    return {
        **feature,
        "standardize_features": raw["standardize_features"],
        "maximum_observations": positive_integer(
            raw["maximum_observations"], "maximum_observations",
            error_type=PaLDCommunityError,
        ),
        "community_msm_enabled": raw["community_msm_enabled"],
        "maximum_reported_intercommunity_ties": positive_integer(
            raw["maximum_reported_intercommunity_ties"],
            "maximum_reported_intercommunity_ties",
            error_type=PaLDCommunityError,
        ),
    }


def _trajectory_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    return (
        str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"]),
        *((str(row["member_id"]),) if "member_id" in row else ()),
    )


def regular_strided_sample(
    metadata: Sequence[Mapping[str, object]], maximum_observations: int,
) -> Tuple[List[int], Dict[str, object]]:
    """Select one common integer stride within every replica-member segment."""

    maximum = positive_integer(
        maximum_observations, "maximum_observations", error_type=PaLDCommunityError
    )
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(_trajectory_key(row), []).append(index)
    if not groups:
        raise PaLDCommunityError("PaLD feature matrix has no trajectory groups")
    for key, indices in groups.items():
        indices.sort(key=lambda index: int(metadata[index]["source_frame_index"]))
        frames = [int(metadata[index]["source_frame_index"]) for index in indices]
        if len(frames) != len(set(frames)):
            raise PaLDCommunityError(
                "duplicate PaLD source frame in " + "/".join(key)
            )
    stride = max(1, math.ceil(len(metadata) / maximum))
    while True:
        selected = sorted(
            index
            for indices in groups.values()
            for index in (
                indices[position]
                for position in sorted(integer_stride_indices(len(indices), stride))
            )
        )
        if len(selected) <= maximum:
            break
        stride += 1
    if len(selected) < 2:
        raise PaLDCommunityError(
            "maximum_observations retains fewer than two PaLD observations"
        )
    selected_set = set(selected)
    rows = []
    for key in sorted(groups):
        chosen = [index for index in groups[key] if index in selected_set]
        rows.append({
            "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
            **({"member_id": key[3]} if len(key) == 4 else {}),
            "source_observation_count": len(groups[key]),
            "selected_observation_count": len(chosen),
        })
    return selected, {
        "mode": "common_regular_stride_per_replica_member_segment_v1",
        "source_observation_count": len(metadata),
        "maximum_observations": maximum,
        "selected_observation_count": len(selected),
        "source_frame_stride": stride,
        "selected_source_matrix_indices": selected,
        "trajectory_groups": rows,
        "pooling_contract": (
            "replicas and equivalent oligomer members contribute pooled community "
            "observations while replica, member, and segment boundaries remain explicit"
        ),
    }


def _connected_components(strong: np.ndarray) -> List[List[int]]:
    unseen = set(range(strong.shape[0]))
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = {
                int(value) for value in np.flatnonzero(strong[node] > 0.0)
            }
            for neighbor in sorted(neighbors & unseen, reverse=True):
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (values[0], len(values)))


def _community_msm(
    project: Mapping[str, object], rows: Sequence[Mapping[str, object]],
    coverage: float,
) -> Dict[str, object]:
    from .msm import _evaluate_state_definition, _settings as msm_settings

    definitions = project.get("definitions")
    if not isinstance(definitions, dict) or not isinstance(
        definitions.get("markov_state_models"), dict
    ):
        return {
            "status": "not run",
            "reason": "the project has no markov_state_models definition",
        }
    settings = msm_settings(project)
    return {
        "status": "complete",
        "model": _evaluate_state_definition(
            rows, settings, candidate_id="pald_strong_tie_communities",
            family="pald_community", geometric_score=None,
            geometric_coverage=coverage,
        ),
    }


def pald_community_analysis_project(
    project_path: Path, hash_content: bool = False,
) -> Dict[str, object]:
    """Calculate sampled PaLD cohesion, depth, strong ties, and communities."""

    source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "pald_community_analysis", source, hash_content=hash_content,
        error_type=PaLDCommunityError,
    )
    if cached is not None:
        return cached
    project = load_json(source)
    if not isinstance(project, dict):
        raise PaLDCommunityError("project manifest must be an object")
    settings = _settings(project)
    feature_report, metadata, raw_vectors, feature_contract = load_feature_matrix(
        source, settings, hash_content=hash_content,
        error_type=PaLDCommunityError,
    )
    vectors, means, scales = _standardize(
        raw_vectors, bool(settings["standardize_features"])
    )
    selected, sampling = regular_strided_sample(
        metadata, int(settings["maximum_observations"])
    )
    sampled_vectors = [vectors[index] for index in selected]
    sampled_metadata = [metadata[index] for index in selected]
    raw = partitioned_local_depths(
        sampled_vectors, int(settings["maximum_observations"])
    )
    cohesion = np.asarray(raw["cohesion_matrix"], dtype=float)
    if cohesion.shape != (len(selected), len(selected)):
        raise PaLDCommunityError("PaLD cohesion matrix shape is invalid")
    local_depth = cohesion.sum(axis=1)
    threshold = float(np.mean(np.diag(cohesion)) / 2.0)
    mutual = np.minimum(cohesion, cohesion.T)
    np.fill_diagonal(mutual, 0.0)
    strong = np.where(mutual >= threshold, mutual, 0.0)
    components = _connected_components(strong)
    labels = [0] * len(selected)
    for community_id, component in enumerate(components, start=1):
        for index in component:
            labels[index] = community_id

    observation_rows = []
    for index, (record, community_id) in enumerate(zip(sampled_metadata, labels)):
        same = np.asarray(labels, dtype=int) == community_id
        within = float(mutual[index, same].sum())
        cross = float(mutual[index, ~same].sum())
        total = within + cross
        observation_rows.append({
            **record,
            "pald_sample_index": index,
            "community_id": community_id,
            "cluster_id": community_id,
            "local_depth": float(local_depth[index]),
            "within_community_mutual_cohesion": within,
            "cross_community_mutual_cohesion": cross,
            "boundary_cohesion_fraction": cross / total if total > 0.0 else 0.0,
        })
    communities = []
    for community_id, component in enumerate(components, start=1):
        core = max(component, key=lambda index: (local_depth[index], -index))
        communities.append({
            "community_id": community_id,
            "sampled_population": len(component),
            "sampled_population_fraction": len(component) / len(selected),
            "core_observation": observation_rows[core],
            "mean_local_depth": float(np.mean(local_depth[component])),
        })
    intercommunity = []
    for left in range(len(selected) - 1):
        for right in range(left + 1, len(selected)):
            if labels[left] == labels[right] or mutual[left, right] <= 0.0:
                continue
            intercommunity.append({
                "left_sample_index": left,
                "right_sample_index": right,
                "left_community_id": labels[left],
                "right_community_id": labels[right],
                "mutual_cohesion": float(mutual[left, right]),
            })
    intercommunity.sort(
        key=lambda row: (-float(row["mutual_cohesion"]), row["left_sample_index"], row["right_sample_index"])
    )
    community_msm: Dict[str, object] = {
        "status": "not run", "reason": "disabled by configuration",
    }
    if bool(settings["community_msm_enabled"]):
        if len(components) < 2:
            community_msm = {
                "status": "not run",
                "reason": "PaLD strong-tie graph contains fewer than two communities",
            }
        else:
            community_msm = _community_msm(
                project, observation_rows, len(selected) / len(vectors)
            )

    source_physical_frames = {
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        for row in metadata
    }
    sampled_physical_frames = {
        (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        for row in sampled_metadata
    }

    return {
        "module_id": "pald_community_analysis",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": feature_report["project_manifest_sha256"],
        "system_manifest_path": feature_report["system_manifest_path"],
        "system_manifest_sha256": feature_report["system_manifest_sha256"],
        "input_content_signature_sha256": feature_report[
            "input_content_signature_sha256"
        ],
        "content_hashes_included": hash_content,
        "settings": settings,
        "feature_contract": feature_contract,
        "feature_standardization": {"means": means, "scales": scales},
        "source_observation_count": len(vectors),
        "sampled_observation_count": len(selected),
        "sampled_coverage_fraction": len(selected) / len(vectors),
        "observation_accounting": {
            "source_physical_frame_count": len(source_physical_frames),
            "source_member_observation_count": len(vectors),
            "selected_physical_frame_count": len(sampled_physical_frames),
            "symmetry_expanded_observation_count": len(selected),
            "member_observations_are_independent_replicas": False,
        },
        "sampling": sampling,
        "strong_tie_threshold": threshold,
        "strong_tie_threshold_definition": "half_mean_cohesion_diagonal_v1",
        "local_depth_mean": float(np.mean(local_depth)),
        "local_depths": [float(value) for value in local_depth],
        "cohesion_matrix": cohesion.tolist(),
        "mutual_cohesion_matrix": mutual.tolist(),
        "strong_tie_matrix": strong.tolist(),
        "strong_tie_count": int(np.count_nonzero(np.triu(strong, 1))),
        "community_count": len(components),
        "communities": communities,
        "sampled_observations": observation_rows,
        "strongest_intercommunity_ties": intercommunity[
            : int(settings["maximum_reported_intercommunity_ties"])
        ],
        "community_msm": community_msm,
        "workload_signature_sha256": stable_json_sha256({
            "module_id": "pald_community_analysis",
            "settings": settings,
            "feature_contract": feature_contract,
            "sampled_observation_count": len(selected),
        }),
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
            "PaLD is a sampled geometric community analysis, not a conventional all-frame clustering method.",
            "Cohesion, local depth, and strong ties do not establish kinetics, pathways, metastability, causality, or convergence.",
            "The direct cohesion calculation is cubic in sampled observation count and is protected by a separate planner gate.",
            "Equivalent oligomer members contribute separate observations but are not independent simulation replicas.",
            "Any community MSM uses only regularly strided sampled labels and remains separate from best-clustering selection.",
        ],
    }


def pald_community_analysis_project_safe(
    project_path: Path, hash_content: bool = False,
) -> Dict[str, object]:
    try:
        return pald_community_analysis_project(
            project_path, hash_content=hash_content
        )
    except (
        PaLDCommunityError, AlternativeClusteringError, ClusteringAnalysisError,
        TrajectoryFeatureError, ManifestValidationError, ImportError, OSError,
        KeyError, TypeError, ValueError,
    ) as exc:
        messages = (
            list(exc.issues) if isinstance(exc, ManifestValidationError)
            else [str(exc)]
        )
        return {
            "module_id": "pald_community_analysis",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "PALD_COMMUNITY_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
