"""Reusable Cartesian, distance, center-of-mass, and principal-axis features."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, read_topology_atoms
from .columnar_artifacts import (
    AtomicColumnarBundle,
    ColumnarArtifactError,
    iter_columnar_records,
)
from .context import compile_project_context_file
from .coordinates import CellVectors, CoordinateReadError, iter_coordinate_frames
from .frame_sampling import frame_selected, plan_frame_selection, reader_frame_indices
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .observables import minimum_group_distance
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    minimum_image_displacement,
)
from .rmsd_rg import RMSDRGError, atomic_masses
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)
from .upstream_cache import project_module_contract_sha256
from .validation import positive_integer


class TrajectoryFeatureError(ValueError):
    """Raised when a trajectory-feature contract is invalid."""


def iter_feature_records(
    feature_report: Mapping[str, object],
) -> Iterator[Mapping[str, object]]:
    """Return an iterator over legacy inline or hash-bound columnar records."""

    records = feature_report.get("records")
    if isinstance(records, list):
        return iter(records)
    reference = feature_report.get("columnar_artifact")
    if isinstance(reference, dict):
        return iter_columnar_records(reference)
    raise TrajectoryFeatureError(
        "trajectory feature has neither inline records nor a columnar artifact"
    )


def flatten_coordinates(
    coordinates: Sequence[Sequence[float]], atom_indices: Sequence[int]
) -> List[float]:
    if not atom_indices:
        raise TrajectoryFeatureError("atom_indices must be nonempty")
    try:
        return [float(value) for index in atom_indices for value in coordinates[index]]
    except IndexError as exc:
        raise TrajectoryFeatureError("atom index exceeds coordinate count") from exc


def center_of_mass(
    coordinates: Sequence[Sequence[float]],
    masses: Sequence[float],
    atom_indices: Sequence[int],
) -> Tuple[float, float, float]:
    if not atom_indices:
        raise TrajectoryFeatureError("center-of-mass atom_indices must be nonempty")
    selected_masses = [float(masses[index]) for index in atom_indices]
    total = sum(selected_masses)
    if not math.isfinite(total) or total <= 0.0:
        raise TrajectoryFeatureError("selected masses must sum to a positive value")
    return tuple(
        sum(float(coordinates[index][axis]) * mass for index, mass in zip(atom_indices, selected_masses)) / total
        for axis in range(3)
    )  # type: ignore[return-value]


def center_of_geometry(
    coordinates: Sequence[Sequence[float]], atom_indices: Sequence[int]
) -> Tuple[float, float, float]:
    if not atom_indices:
        raise TrajectoryFeatureError("center-of-geometry atom_indices must be nonempty")
    try:
        return tuple(
            sum(float(coordinates[index][axis]) for index in atom_indices)
            / len(atom_indices)
            for axis in range(3)
        )  # type: ignore[return-value]
    except IndexError as exc:
        raise TrajectoryFeatureError("atom index exceeds coordinate count") from exc


def _distance(
    first: Sequence[float],
    second: Sequence[float],
    cell: CellVectors | None = None,
) -> float:
    displacement = tuple(float(second[axis]) - float(first[axis]) for axis in range(3))
    if cell is not None:
        displacement = minimum_image_displacement(displacement, cell)
    return math.sqrt(sum(value * value for value in displacement))


def group_distance_statistics(
    coordinates: Sequence[Sequence[float]],
    group_a: Sequence[int],
    group_b: Sequence[int],
    cell: CellVectors | None = None,
) -> Dict[str, object]:
    """Return explicit min/mean/max statistics over all cross-group atom pairs."""

    if not group_a or not group_b:
        raise TrajectoryFeatureError("distance groups must be nonempty")
    try:
        pairs = [
            (_distance(coordinates[left], coordinates[right], cell), left, right)
            for left in group_a
            for right in group_b
        ]
    except IndexError as exc:
        raise TrajectoryFeatureError("atom index exceeds coordinate count") from exc
    closest = min(pairs)
    farthest = max(pairs)
    return {
        "minimum_distance_angstrom": closest[0],
        "mean_distance_angstrom": sum(row[0] for row in pairs) / len(pairs),
        "maximum_distance_angstrom": farthest[0],
        "closest_atom_indices": [closest[1], closest[2]],
        "farthest_atom_indices": [farthest[1], farthest[2]],
        "pair_count": len(pairs),
    }


def minimum_mean_group_distance(
    coordinates: Sequence[Sequence[float]],
    reference_group: Sequence[int],
    candidate_group: Sequence[int],
    cell: CellVectors | None = None,
) -> Dict[str, object]:
    """Find the candidate atom with the smallest mean distance to a reference group."""

    if not reference_group or not candidate_group:
        raise TrajectoryFeatureError("distance groups must be nonempty")
    try:
        candidates = [
            (
                sum(
                    _distance(coordinates[left], coordinates[right], cell)
                    for left in reference_group
                )
                / len(reference_group),
                right,
            )
            for right in candidate_group
        ]
    except IndexError as exc:
        raise TrajectoryFeatureError("atom index exceeds coordinate count") from exc
    distance, index = min(candidates)
    return {
        "minimum_mean_distance_angstrom": distance,
        "selected_candidate_atom_index": index,
        "reference_atom_count": len(reference_group),
    }


def principal_axes(
    coordinates: Sequence[Sequence[float]],
    masses: Sequence[float],
    atom_indices: Sequence[int],
    mass_weighted: bool = True,
) -> Dict[str, object]:
    """Return descending principal moments and orthonormal inertia axes."""

    if len(atom_indices) < 3:
        raise TrajectoryFeatureError("principal axes require at least three atoms")
    xyz = np.asarray([coordinates[index] for index in atom_indices], dtype=float)
    weights = np.asarray(
        [masses[index] if mass_weighted else 1.0 for index in atom_indices], dtype=float
    )
    if not np.isfinite(xyz).all() or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise TrajectoryFeatureError("principal-axis coordinates or masses are invalid")
    centroid = np.average(xyz, axis=0, weights=weights)
    centered = xyz - centroid
    radius2 = np.sum(centered * centered, axis=1)
    tensor = np.eye(3) * float(np.sum(weights * radius2))
    tensor -= np.einsum("n,ni,nj->ij", weights, centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return {
        "centroid_angstrom": centroid.tolist(),
        "principal_moments": eigenvalues[order].tolist(),
        "principal_axes": axes.T.tolist(),
        "mass_weighted": bool(mass_weighted),
    }


_KINDS = {
    "cartesian", "fluctuation", "center_of_mass", "pair_distance",
    "group_minimum_distance", "group_distance_statistics",
    "group_minimum_mean_distance", "center_of_mass_distance",
    "center_of_mass_minimum_distance", "center_of_geometry_minimum_distance",
    "principal_axes",
}


_VALUE_LABELS = {
    "pair_distance": ["distance_angstrom"],
    "group_minimum_distance": ["minimum_distance_angstrom"],
    "group_distance_statistics": [
        "minimum_distance_angstrom", "mean_distance_angstrom", "maximum_distance_angstrom"
    ],
    "group_minimum_mean_distance": ["minimum_mean_distance_angstrom"],
    "center_of_mass_distance": ["center_of_mass_distance_angstrom"],
    "center_of_mass_minimum_distance": ["center_of_mass_minimum_distance_angstrom"],
    "center_of_geometry_minimum_distance": ["center_of_geometry_minimum_distance_angstrom"],
}


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("trajectory_features") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise TrajectoryFeatureError("definitions.trajectory_features must be an object")
    required = {"frame_stride", "maximum_feature_values", "features"}
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise TrajectoryFeatureError(
            "trajectory-feature settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    features = raw["features"]
    if not isinstance(features, list) or not features:
        raise TrajectoryFeatureError("features must be a nonempty array")
    normalized = []
    identifiers = set()
    for feature in features:
        if not isinstance(feature, dict):
            raise TrajectoryFeatureError("each trajectory feature must be an object")
        feature_id = str(feature.get("feature_id", "")).strip()
        kind = feature.get("kind")
        if not feature_id or feature_id in identifiers or kind not in _KINDS:
            raise TrajectoryFeatureError("feature IDs must be unique and kinds supported")
        expected = {"feature_id", "kind"}
        if kind in {"cartesian", "fluctuation", "center_of_mass", "principal_axes"}:
            expected.add("atom_indices")
        elif kind == "pair_distance":
            expected.add("atom_indices")
        elif kind in {
            "group_minimum_distance", "group_distance_statistics",
            "group_minimum_mean_distance", "center_of_mass_distance",
            "center_of_mass_minimum_distance", "center_of_geometry_minimum_distance",
        }:
            expected.update({"group_a_atom_indices", "group_b_atom_indices"})
        else:
            raise AssertionError(kind)
        if kind == "principal_axes":
            expected.add("mass_weighted")
        if set(feature) != expected:
            raise TrajectoryFeatureError(
                f"feature {feature_id} fields do not match the {kind} contract"
            )
        groups = []
        if "atom_indices" in feature:
            groups.append(feature["atom_indices"])
        else:
            groups.extend([feature["group_a_atom_indices"], feature["group_b_atom_indices"]])
        for group in groups:
            if not isinstance(group, list) or not group or len(set(group)) != len(group):
                raise TrajectoryFeatureError("feature atom-index groups must be nonempty and unique")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in group):
                raise TrajectoryFeatureError("feature atom indices must be nonnegative integers")
        if kind == "pair_distance" and len(feature["atom_indices"]) != 2:
            raise TrajectoryFeatureError("pair_distance requires exactly two atom indices")
        identifiers.add(feature_id)
        normalized.append(dict(feature))
    return {
        "frame_stride": positive_integer(raw["frame_stride"], "frame_stride"),
        "maximum_feature_values": positive_integer(
            raw["maximum_feature_values"], "maximum_feature_values"
        ),
        "features": normalized,
    }


def trajectory_features_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    selection_plan, _ = plan_frame_selection(
        system,
        system_path,
        coordinate_unit,
        {
            "mode": "integer_stride_per_replica_v1",
            "stride": int(settings["frame_stride"]),
        },
        frame_stride=1,
        error_type=TrajectoryFeatureError,
    )
    output_time_unit = project.get("time_unit")
    artifact_root_text = os.environ.get(
        "SALSBURY_MD_ANALYSIS_COLUMNAR_ARTIFACT_ROOT"
    )
    artifact_bundle = (
        AtomicColumnarBundle(Path(artifact_root_text))
        if artifact_root_text else None
    )
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    segment_reports = []
    stored_values = 0
    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            masses = atomic_masses(atoms)
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            reconstruction_atom_indices = tuple(sorted({
                int(index)
                for feature in settings["features"]
                for group in (
                    [feature["atom_indices"]]
                    if "atom_indices" in feature
                    else [
                        feature["group_a_atom_indices"],
                        feature["group_b_atom_indices"],
                    ]
                )
                for index in group
            }))
            for feature in settings["features"]:
                groups = (
                    [feature["atom_indices"]]
                    if "atom_indices" in feature
                    else [feature["group_a_atom_indices"], feature["group_b_atom_indices"]]
                )
                if max(value for group in groups for value in group) >= len(atoms):
                    raise TrajectoryFeatureError(
                        f"feature {feature['feature_id']} exceeds topology atom count"
                    )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = selection_plan[(system_id, replica_id, segment_id)]
                axis = normalize_segment_axis(
                    segment, str(output_time_unit) if output_time_unit else None
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                frame_indices: List[int] = []
                feature_states = [
                    {
                        "raw_vectors": [],
                        "closest_pairs": [],
                        "auxiliary_records": [],
                        "reference_axes": None,
                    }
                    for _ in settings["features"]
                ]
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, str(project["periodic_coordinate_policy"])),
                ):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if not frame_selected(frame.frame_index, selected_indices, 1):
                        continue
                    frame_indices.append(frame.frame_index)
                    coordinates = frame.coordinates_angstrom
                    for feature, state in zip(settings["features"], feature_states):
                        kind = str(feature["kind"])
                        closest_pair = None
                        auxiliary = None
                        if kind in {"cartesian", "fluctuation"}:
                            vector = flatten_coordinates(coordinates, feature["atom_indices"])
                        elif kind == "center_of_mass":
                            vector = list(center_of_mass(coordinates, masses, feature["atom_indices"]))
                        elif kind == "pair_distance":
                            distance, closest_pair = minimum_group_distance(
                                coordinates,
                                [feature["atom_indices"][0]],
                                [feature["atom_indices"][1]],
                                frame.cell_vectors_angstrom,
                            )
                            vector = [distance]
                        elif kind == "group_minimum_distance":
                            distance, closest_pair = minimum_group_distance(
                                coordinates,
                                feature["group_a_atom_indices"],
                                feature["group_b_atom_indices"],
                                frame.cell_vectors_angstrom,
                            )
                            vector = [distance]
                        elif kind == "group_distance_statistics":
                            payload = group_distance_statistics(
                                coordinates,
                                feature["group_a_atom_indices"],
                                feature["group_b_atom_indices"],
                                frame.cell_vectors_angstrom,
                            )
                            vector = [
                                float(payload["minimum_distance_angstrom"]),
                                float(payload["mean_distance_angstrom"]),
                                float(payload["maximum_distance_angstrom"]),
                            ]
                            auxiliary = {
                                "closest_atom_indices": payload["closest_atom_indices"],
                                "farthest_atom_indices": payload["farthest_atom_indices"],
                                "pair_count": payload["pair_count"],
                            }
                        elif kind == "group_minimum_mean_distance":
                            payload = minimum_mean_group_distance(
                                coordinates,
                                feature["group_a_atom_indices"],
                                feature["group_b_atom_indices"],
                                frame.cell_vectors_angstrom,
                            )
                            vector = [float(payload["minimum_mean_distance_angstrom"])]
                            auxiliary = {
                                "selected_candidate_atom_index": payload["selected_candidate_atom_index"],
                                "reference_atom_count": payload["reference_atom_count"],
                            }
                        elif kind == "center_of_mass_distance":
                            first = center_of_mass(
                                coordinates, masses, feature["group_a_atom_indices"]
                            )
                            second = center_of_mass(
                                coordinates, masses, feature["group_b_atom_indices"]
                            )
                            vector = [_distance(first, second, frame.cell_vectors_angstrom)]
                        elif kind in {
                            "center_of_mass_minimum_distance",
                            "center_of_geometry_minimum_distance",
                        }:
                            center = (
                                center_of_mass(
                                    coordinates, masses, feature["group_a_atom_indices"]
                                )
                                if kind == "center_of_mass_minimum_distance"
                                else center_of_geometry(
                                    coordinates, feature["group_a_atom_indices"]
                                )
                            )
                            candidates = [
                                (
                                    _distance(
                                        center, coordinates[index], frame.cell_vectors_angstrom
                                    ),
                                    index,
                                )
                                for index in feature["group_b_atom_indices"]
                            ]
                            distance, index = min(candidates)
                            vector = [distance]
                            auxiliary = {"closest_group_b_atom_index": index}
                        else:
                            payload = principal_axes(
                                coordinates, masses, feature["atom_indices"],
                                bool(feature["mass_weighted"]),
                            )
                            axes = np.asarray(payload["principal_axes"], dtype=float)
                            if state["reference_axes"] is None:
                                state["reference_axes"] = axes
                            vector = list(payload["principal_moments"])
                            vector.extend(
                                np.abs(axes @ state["reference_axes"].T).reshape(-1).tolist()
                            )
                        state["raw_vectors"].append(vector)
                        if closest_pair is not None:
                            state["closest_pairs"].append(list(closest_pair))
                        if auxiliary is not None:
                            state["auxiliary_records"].append(auxiliary)
                if not frame_indices:
                    raise TrajectoryFeatureError("trajectory segment produced no evaluated frames")
                feature_reports = []
                for feature, state in zip(settings["features"], feature_states):
                    kind = str(feature["kind"])
                    values = np.asarray(state["raw_vectors"], dtype=float)
                    if kind == "fluctuation":
                        values -= values.mean(axis=0)
                    stored_values += int(values.size)
                    if stored_values > settings["maximum_feature_values"]:
                        raise TrajectoryFeatureError("maximum_feature_values gate exceeded")
                    axis_values = [
                        frame_axis_value(axis, frame_index)
                        for frame_index in frame_indices
                    ]
                    feature_report = {
                        "feature_id": feature["feature_id"],
                        "kind": kind,
                        "dimension": int(values.shape[1]),
                    }
                    if artifact_bundle is None:
                        rows = []
                        for index, (frame_index, values_row) in enumerate(
                            zip(frame_indices, values)
                        ):
                            row = {
                                "source_frame_index": frame_index,
                                "axis_kind": axis["kind"],
                                "axis_value": axis_values[index],
                                "values": values_row.tolist(),
                            }
                            if state["closest_pairs"]:
                                row["closest_atom_indices"] = (
                                    state["closest_pairs"][index]
                                )
                            if state["auxiliary_records"]:
                                row.update(state["auxiliary_records"][index])
                            rows.append(row)
                        feature_report["records"] = rows
                        feature_report["record_storage"] = "inline_json"
                    else:
                        columns: Dict[str, object] = {
                            "source_frame_index": np.asarray(
                                frame_indices, dtype=np.int64
                            ),
                            "axis_value": np.asarray(axis_values, dtype=np.float64),
                            "values": values,
                        }
                        if state["closest_pairs"]:
                            columns["closest_atom_indices"] = np.asarray(
                                state["closest_pairs"], dtype=np.int64
                            )
                        if state["auxiliary_records"]:
                            auxiliary_keys = sorted(
                                state["auxiliary_records"][0]
                            )
                            if any(
                                sorted(record) != auxiliary_keys
                                for record in state["auxiliary_records"]
                            ):
                                raise TrajectoryFeatureError(
                                    "auxiliary feature columns are inconsistent"
                                )
                            for key in auxiliary_keys:
                                columns[key] = np.asarray([
                                    record[key]
                                    for record in state["auxiliary_records"]
                                ])
                        segment_ordinal = len(segment_reports)
                        feature_ordinal = len(feature_reports)
                        feature_report["columnar_artifact"] = (
                            artifact_bundle.write_table(
                                f"segment-{segment_ordinal:05d}/"
                                f"feature-{feature_ordinal:04d}",
                                columns,
                                constants={"axis_kind": axis["kind"]},
                                provenance={
                                    "module_id": "trajectory_features",
                                    "project_manifest_sha256": context[
                                        "project_manifest_sha256"
                                    ],
                                    "system_manifest_sha256": context[
                                        "system_manifest_sha256"
                                    ],
                                    "input_content_signature_sha256": context[
                                        "input_content_signature_sha256"
                                    ],
                                    "system_id": system_id,
                                    "replica_id": replica_id,
                                    "segment_id": segment_id,
                                    "feature_id": feature["feature_id"],
                                },
                            )
                        )
                        feature_report["records"] = None
                        feature_report["record_storage"] = (
                            "hash_bound_numpy_columnar"
                        )
                    if kind in _VALUE_LABELS:
                        feature_report["value_labels"] = _VALUE_LABELS[kind]
                    feature_reports.append(feature_report)
                segment_reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "segment_id": segment_id,
                    "evaluated_frame_count": len(frame_indices),
                    "features": feature_reports,
                })
    if artifact_bundle is not None:
        artifact_bundle.publish()
    return {
        "module_id": "trajectory_features",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "module_contract_sha256": project_module_contract_sha256(
            "trajectory_features", source
        ),
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "stored_feature_value_count": stored_values,
        "columnar_artifact_root": (
            str(artifact_bundle.output_root)
            if artifact_bundle is not None else None
        ),
        "segments": segment_reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Feature definitions and atom indices must be frozen before comparative analysis.",
            "Group distance statistics distinguish pairwise minimum, mean, maximum, center-of-mass, and center-of-geometry definitions; they are not interchangeable.",
            "Principal-axis orientations are sign-insensitive and reported relative to the first evaluated frame.",
            "Project-specific distance meanings remain in project manifests rather than package source.",
        ],
    }


def trajectory_features_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return trajectory_features_project(project_path, hash_content=hash_content)
    except (
        TrajectoryFeatureError, ColumnarArtifactError,
        ManifestValidationError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError,
        RMSDRGError, OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "trajectory_features",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "TRAJECTORY_FEATURE_INVALID", "message": message}
                for message in messages
            ],
        }
