"""Deterministic representative-frame selection for clusters and PCA basins."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .clustering import ClusteringAnalysisError, clustering_kmeans_project
from .manifests import ManifestValidationError, load_json
from .pca import PCAAnalysisError
from .pca_fes import PCAFESAnalysisError, pca_fes_basins_project
from .validation import positive_integer


class RepresentativeFrameError(ValueError):
    """Raised when representative frames cannot be selected unambiguously."""


_REQUIRED_SETTINGS = {
    "source",
    "representatives_per_state",
    "maximum_states",
    "maximum_candidates",
}
_OPTIONAL_SETTINGS = {"fes_smoothing_sigma_bins"}
_SOURCES = {"clustering_kmeans", "pca_fes_basins"}
_IDENTITY_FIELDS = (
    "system_id",
    "replica_id",
    "segment_id",
    "source_frame_index",
)


_positive_integer = partial(positive_integer, error_type=RepresentativeFrameError)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise RepresentativeFrameError(
            "project definitions.representative_frames is required"
        )
    raw = definitions.get("representative_frames")
    if not isinstance(raw, dict):
        raise RepresentativeFrameError(
            "project definitions.representative_frames must be an object"
        )
    unknown = sorted(set(raw).difference(_REQUIRED_SETTINGS | _OPTIONAL_SETTINGS))
    if unknown:
        raise RepresentativeFrameError(
            "definitions.representative_frames contains unknown fields: "
            + ", ".join(unknown)
        )
    missing = sorted(_REQUIRED_SETTINGS.difference(raw))
    if missing:
        raise RepresentativeFrameError(
            "definitions.representative_frames is missing required fields: "
            + ", ".join(missing)
        )
    source = raw["source"]
    if source not in _SOURCES:
        raise RepresentativeFrameError(
            "representative_frames source must be clustering_kmeans or pca_fes_basins"
        )
    normalized = {
        "source": str(source),
        "representatives_per_state": _positive_integer(
            raw["representatives_per_state"], "representatives_per_state"
        ),
        "maximum_states": _positive_integer(raw["maximum_states"], "maximum_states"),
        "maximum_candidates": _positive_integer(
            raw["maximum_candidates"], "maximum_candidates"
        ),
    }
    if "fes_smoothing_sigma_bins" in raw:
        value = raw["fes_smoothing_sigma_bins"]
        if (
            source != "pca_fes_basins"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RepresentativeFrameError(
                "fes_smoothing_sigma_bins must be a finite nonnegative number and is valid only for pca_fes_basins"
            )
        normalized["fes_smoothing_sigma_bins"] = float(value)
    return normalized


def select_state_representatives(
    candidates: Sequence[Mapping[str, object]],
    state_field: str,
    distance_field: str,
    representatives_per_state: int,
    maximum_states: int,
    maximum_candidates: int,
) -> List[Dict[str, object]]:
    """Select nearest observed frames per state with a stable identity tie break."""

    representatives_per_state = _positive_integer(
        representatives_per_state, "representatives_per_state"
    )
    maximum_states = _positive_integer(maximum_states, "maximum_states")
    maximum_candidates = _positive_integer(maximum_candidates, "maximum_candidates")
    if not candidates:
        raise RepresentativeFrameError("representative-frame candidates must not be empty")
    if len(candidates) > maximum_candidates:
        raise RepresentativeFrameError(
            f"candidate count {len(candidates)} exceeds maximum_candidates {maximum_candidates}"
        )

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row_index, candidate in enumerate(candidates):
        missing = [field for field in (*_IDENTITY_FIELDS, state_field, distance_field) if field not in candidate]
        if missing:
            raise RepresentativeFrameError(
                f"candidate {row_index} is missing fields: {', '.join(missing)}"
            )
        raw_state = candidate[state_field]
        if raw_state is None:
            continue
        if isinstance(raw_state, bool) or not isinstance(raw_state, int) or raw_state <= 0:
            raise RepresentativeFrameError(
                f"candidate {row_index} {state_field} must be a positive integer or null"
            )
        raw_distance = candidate[distance_field]
        if (
            isinstance(raw_distance, bool)
            or not isinstance(raw_distance, (int, float))
            or not math.isfinite(float(raw_distance))
            or float(raw_distance) < 0.0
        ):
            raise RepresentativeFrameError(
                f"candidate {row_index} {distance_field} must be finite and nonnegative"
            )
        frame_index = candidate["source_frame_index"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise RepresentativeFrameError(
                f"candidate {row_index} source_frame_index must be a nonnegative integer"
            )
        normalized = dict(candidate)
        normalized[distance_field] = float(raw_distance)
        grouped.setdefault(raw_state, []).append(normalized)

    if not grouped:
        raise RepresentativeFrameError("no assigned states have representative candidates")
    if len(grouped) > maximum_states:
        raise RepresentativeFrameError(
            f"state count {len(grouped)} exceeds maximum_states {maximum_states}"
        )

    selected: List[Dict[str, object]] = []
    for state_id in sorted(grouped):
        ordered = sorted(
            grouped[state_id],
            key=lambda row: (
                float(row[distance_field]),
                str(row["system_id"]),
                str(row["replica_id"]),
                str(row["segment_id"]),
                int(row["source_frame_index"]),
                str(row.get("member_id", "")),
            ),
        )
        for rank, row in enumerate(ordered[:representatives_per_state], start=1):
            selected.append({
                "state_id": state_id,
                "representative_rank": rank,
                "distance_to_state_center_squared": float(row[distance_field]),
                **{field: row[field] for field in _IDENTITY_FIELDS},
                **(
                    {"member_id": row["member_id"]}
                    if "member_id" in row else {}
                ),
                **(
                    {"sample_index": row["sample_index"]}
                    if "sample_index" in row
                    else {
                        "time": row.get("time"),
                        "time_unit": row.get("time_unit"),
                    }
                ),
            })
    return selected


def _basin_candidates(
    report: Mapping[str, object], requested_sigma_bins: object = None
) -> List[Dict[str, object]]:
    selected: Mapping[str, object] = report
    if requested_sigma_bins is not None:
        alternatives = report.get("smoothing_landscapes")
        if not isinstance(alternatives, list):
            raise RepresentativeFrameError(
                "PCA-FES report does not contain smoothing sensitivity landscapes"
            )
        matches = [
            row for row in alternatives
            if isinstance(row, dict)
            and row.get("smoothing_sigma_bins") == requested_sigma_bins
        ]
        if len(matches) != 1:
            raise RepresentativeFrameError(
                f"PCA-FES report does not contain smoothing sigma {requested_sigma_bins}"
            )
        selected = matches[0]
    landscape = selected.get("landscape")
    if not isinstance(landscape, dict) or not isinstance(landscape.get("basins"), list):
        raise RepresentativeFrameError("PCA-FES report does not contain basin definitions")
    centers: Dict[int, tuple[float, float]] = {}
    for basin in landscape["basins"]:
        if not isinstance(basin, dict):
            raise RepresentativeFrameError("PCA-FES basin definition must be an object")
        basin_id = basin.get("basin_id")
        if isinstance(basin_id, bool) or not isinstance(basin_id, int) or basin_id <= 0:
            raise RepresentativeFrameError("PCA-FES basin_id must be a positive integer")
        centers[basin_id] = (
            float(basin["root_x_center_angstrom"]),
            float(basin["root_y_center_angstrom"]),
        )
    assignments = selected.get("frame_assignments")
    if not isinstance(assignments, list):
        raise RepresentativeFrameError("PCA-FES report does not contain frame assignments")
    candidates = []
    for row in assignments:
        if not isinstance(row, dict):
            raise RepresentativeFrameError("PCA-FES frame assignment must be an object")
        basin_id = row.get("basin_id")
        if basin_id is None:
            candidates.append({**row, "distance_to_basin_root_squared": 0.0})
            continue
        if basin_id not in centers:
            raise RepresentativeFrameError(
                f"PCA-FES frame references unknown basin_id {basin_id!r}"
            )
        center = centers[basin_id]
        distance = (
            (float(row["pc_x_angstrom"]) - center[0]) ** 2
            + (float(row["pc_y_angstrom"]) - center[1]) ** 2
        )
        candidates.append({**row, "distance_to_basin_root_squared": distance})
    return candidates


def representative_frames_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Select declared representatives without writing or modifying coordinates."""

    source_path = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source_path)
    settings = _settings(project)
    source = str(settings["source"])
    if source == "clustering_kmeans":
        upstream = clustering_kmeans_project(source_path, hash_content=hash_content)
        candidates = upstream.get("assignments")
        if not isinstance(candidates, list):
            raise RepresentativeFrameError("KMeans report does not contain assignments")
        state_field = "cluster_id"
        distance_field = "squared_distance_in_clustering_space"
    else:
        upstream = pca_fes_basins_project(source_path, hash_content=hash_content)
        candidates = _basin_candidates(
            upstream, settings.get("fes_smoothing_sigma_bins")
        )
        state_field = "basin_id"
        distance_field = "distance_to_basin_root_squared"

    if upstream.get("technical_status") != "complete":
        raise RepresentativeFrameError(
            f"source module {source} must complete before representatives are selected"
        )

    representatives = select_state_representatives(
        candidates,
        state_field=state_field,
        distance_field=distance_field,
        representatives_per_state=int(settings["representatives_per_state"]),
        maximum_states=int(settings["maximum_states"]),
        maximum_candidates=int(settings["maximum_candidates"]),
    )
    physical_identities = {
        (
            str(row["system_id"]),
            str(row["replica_id"]),
            str(row["segment_id"]),
            int(row["source_frame_index"]),
        )
        for row in candidates
    }
    representative_physical_identities = {
        (
            str(row["system_id"]),
            str(row["replica_id"]),
            str(row["segment_id"]),
            int(row["source_frame_index"]),
        )
        for row in representatives
    }
    issues = [issue for issue in upstream.get("issues", []) if isinstance(issue, dict)]
    return {
        "module_id": "representative_frames",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source_path),
        "project_manifest_sha256": upstream.get("project_manifest_sha256"),
        "system_manifest_path": upstream.get("system_manifest_path"),
        "system_manifest_sha256": upstream.get("system_manifest_sha256"),
        "input_content_signature_sha256": upstream.get(
            "input_content_signature_sha256"
        ),
        "content_hashes_included": hash_content,
        "settings": settings,
        "source_module_id": source,
        "selection_rule": (
            "minimum squared distance to the declared fitted cluster center or PCA-basin root; "
            "ties resolve by system, replica, segment, and source frame identity"
        ),
        "representative_count": len(representatives),
        "state_count": len({row["state_id"] for row in representatives}),
        "observation_accounting": {
            "selected_physical_frame_count": len(physical_identities),
            "symmetry_expanded_observation_count": len(candidates),
            "representative_physical_frame_count": len(
                representative_physical_identities
            ),
            "representative_observation_count": len(representatives),
            "member_observations_are_independent_replicas": False
            if any("member_id" in row for row in candidates) else None,
        },
        "representatives": representatives,
        "coordinate_files_written": 0,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Representatives are observed frames, not arithmetic average structures.",
            "Nearest-center selection inherits every feature, scaling, clustering, and basin-definition choice from the source module.",
            "This module reports immutable frame locators and does not write PDB, trajectory, figure, or publication-specific files.",
            "A representative frame is descriptive and does not establish metastability, convergence, mechanism, or scientific validity.",
        ],
    }


def representative_frames_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Return a machine-readable failure rather than an uncaught exception."""

    try:
        return representative_frames_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        PCAAnalysisError,
        PCAFESAnalysisError,
        ClusteringAnalysisError,
        RepresentativeFrameError,
        OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "representative_frames",
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
                    "code": "REPRESENTATIVE_FRAMES_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
