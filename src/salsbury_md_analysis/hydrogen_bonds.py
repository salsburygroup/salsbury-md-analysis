"""Explicit, auditable hydrogen-bond occupancy features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, read_topology_atoms
from .clustering import ClusteringAnalysisError, clustering_kmeans_project
from .context import compile_project_context_file
from .coordinates import CellVectors, CoordinateReadError, iter_coordinate_frames
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    minimum_image_displacement,
)
from .pca import PCAAnalysisError
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)


class HydrogenBondAnalysisError(ValueError):
    """Raised when a declared hydrogen-bond feature is unsafe or incomplete."""


def _displacement(
    first: Sequence[float], second: Sequence[float], cell: CellVectors | None
) -> Tuple[float, float, float]:
    vector = tuple(second[index] - first[index] for index in range(3))
    return minimum_image_displacement(vector, cell) if cell is not None else vector


def distance_angstrom(
    first: Sequence[float], second: Sequence[float], cell: CellVectors | None = None
) -> float:
    return math.sqrt(sum(value * value for value in _displacement(first, second, cell)))


def angle_degrees(
    first: Sequence[float], vertex: Sequence[float], third: Sequence[float],
    cell: CellVectors | None = None,
) -> float:
    left = _displacement(vertex, first, cell)
    right = _displacement(vertex, third, cell)
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if min(left_norm, right_norm) <= 1.0e-15:
        raise HydrogenBondAnalysisError("hydrogen-bond angle contains a zero-length vector")
    cosine = sum(left[index] * right[index] for index in range(3)) / (left_norm * right_norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def hydrogen_bond_present(
    donor: Sequence[float], hydrogen: Sequence[float], acceptor: Sequence[float],
    maximum_distance: float, minimum_angle: float, cell: CellVectors | None = None,
) -> Tuple[bool, float, float]:
    distance = distance_angstrom(donor, acceptor, cell)
    angle = angle_degrees(donor, hydrogen, acceptor, cell)
    return distance <= maximum_distance and angle >= minimum_angle, distance, angle


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("hydrogen_bonds") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise HydrogenBondAnalysisError("definitions.hydrogen_bonds must be an object")
    required = {
        "features", "frame_stride", "maximum_donor_acceptor_distance_angstrom",
        "minimum_donor_hydrogen_acceptor_angle_degrees",
        "maximum_reference_donor_hydrogen_bond_angstrom", "water_policy",
        "condition_source", "maximum_feature_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing:
        raise HydrogenBondAnalysisError("hydrogen-bond settings missing: " + ", ".join(missing))
    if unknown:
        raise HydrogenBondAnalysisError("hydrogen-bond settings contain unknown fields: " + ", ".join(unknown))
    features = raw["features"]
    if not isinstance(features, list) or not features:
        raise HydrogenBondAnalysisError("features must be a nonempty array")
    normalized = []
    feature_ids = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or set(feature) != {
            "feature_id", "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
        }:
            raise HydrogenBondAnalysisError(
                f"features[{index}] must declare feature_id and three zero-based atom indices"
            )
        feature_id = str(feature["feature_id"]).strip()
        if not feature_id or feature_id in feature_ids:
            raise HydrogenBondAnalysisError("feature IDs must be nonempty and unique")
        indices = [feature[name] for name in (
            "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
        )]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices):
            raise HydrogenBondAnalysisError("hydrogen-bond atom indices must be nonnegative integers")
        if len(set(indices)) != 3:
            raise HydrogenBondAnalysisError("donor, hydrogen, and acceptor indices must be distinct")
        feature_ids.add(feature_id)
        normalized.append(dict(feature))
    if raw["water_policy"] != "exclude":
        raise HydrogenBondAnalysisError(
            "water_policy currently supports only exclude; water-mediated definitions require a separate validated contract"
        )
    if raw["condition_source"] not in {"none", "clustering_kmeans"}:
        raise HydrogenBondAnalysisError("condition_source must be none or clustering_kmeans")
    for label in ("frame_stride", "maximum_feature_observations"):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise HydrogenBondAnalysisError(f"{label} must be a positive integer")
    for label in (
        "maximum_donor_acceptor_distance_angstrom",
        "minimum_donor_hydrogen_acceptor_angle_degrees",
        "maximum_reference_donor_hydrogen_bond_angstrom",
    ):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise HydrogenBondAnalysisError(f"{label} must be finite and positive")
    angle = float(raw["minimum_donor_hydrogen_acceptor_angle_degrees"])
    if angle >= 180.0:
        raise HydrogenBondAnalysisError("minimum hydrogen-bond angle must be below 180 degrees")
    return {
        "features": normalized,
        "frame_stride": raw["frame_stride"],
        "maximum_donor_acceptor_distance_angstrom": float(raw["maximum_donor_acceptor_distance_angstrom"]),
        "minimum_donor_hydrogen_acceptor_angle_degrees": angle,
        "maximum_reference_donor_hydrogen_bond_angstrom": float(raw["maximum_reference_donor_hydrogen_bond_angstrom"]),
        "water_policy": "exclude", "condition_source": raw["condition_source"],
        "maximum_feature_observations": raw["maximum_feature_observations"],
    }


def hydrogen_bonds_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    conditions: Dict[Tuple[str, str, str, int], int] = {}
    condition_model = None
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    if settings["condition_source"] == "clustering_kmeans":
        condition_model = clustering_kmeans_project(source, hash_content=hash_content)
        for row in condition_model["assignments"]:
            key = (
                str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"]),
                int(row["source_frame_index"]),
            )
            if key in conditions:
                raise HydrogenBondAnalysisError("condition source contains duplicate frame identities")
            conditions[key] = int(row["cluster_id"])
        issues.extend(issue for issue in condition_model.get("issues", []) if isinstance(issue, dict))
    totals: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    conditioned: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    observation_count = 0
    segment_reports = []
    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            reconstruction_atom_indices = tuple(sorted({
                int(feature[field])
                for feature in settings["features"]
                for field in (
                    "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
                )
            }))
            raw_reference = next(iter_coordinate_frames(topology_path, coordinate_unit))
            reference = processor.process(
                raw_reference, str(topology_path), reconstruction_atom_indices
            )
            for feature in settings["features"]:
                donor, hydrogen, acceptor = (
                    int(feature[name]) for name in (
                        "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
                    )
                )
                if max(donor, hydrogen, acceptor) >= len(atoms):
                    raise HydrogenBondAnalysisError(
                        f"feature {feature['feature_id']} atom index exceeds topology atom count"
                    )
                if atoms[hydrogen].element.upper() != "H":
                    raise HydrogenBondAnalysisError(
                        f"feature {feature['feature_id']} hydrogen_atom_index is not hydrogen in the topology"
                    )
                reference_bond = distance_angstrom(
                    reference.coordinates_angstrom[donor], reference.coordinates_angstrom[hydrogen],
                    reference.cell_vectors_angstrom,
                )
                if reference_bond > float(settings["maximum_reference_donor_hydrogen_bond_angstrom"]):
                    raise HydrogenBondAnalysisError(
                        f"feature {feature['feature_id']} reference donor-hydrogen distance {reference_bond:.3f} exceeds gate"
                    )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                evaluated_frames = 0
                periodic_frames = 0
                processor.begin_segment(
                    bool(segment.get("continuous_with_previous", False))
                )
                for raw_frame in iter_coordinate_frames(trajectory_path, coordinate_unit):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if frame.atom_count != len(atoms):
                        raise HydrogenBondAnalysisError("trajectory/topology atom count mismatch")
                    periodic_frames += int(frame.periodic_cell_present)
                    if frame.frame_index % int(settings["frame_stride"]):
                        continue
                    evaluated_frames += 1
                    frame_axis_value(axis, frame.frame_index)
                    condition = conditions.get((system_id, replica_id, segment_id, frame.frame_index))
                    if settings["condition_source"] != "none" and condition is None:
                        raise HydrogenBondAnalysisError("condition source lacks an evaluated trajectory frame")
                    for feature in settings["features"]:
                        donor, hydrogen, acceptor = (
                            int(feature[name]) for name in (
                                "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
                            )
                        )
                        present, distance, angle = hydrogen_bond_present(
                            frame.coordinates_angstrom[donor], frame.coordinates_angstrom[hydrogen],
                            frame.coordinates_angstrom[acceptor],
                            float(settings["maximum_donor_acceptor_distance_angstrom"]),
                            float(settings["minimum_donor_hydrogen_acceptor_angle_degrees"]),
                            frame.cell_vectors_angstrom,
                        )
                        key = (system_id, replica_id, str(feature["feature_id"]))
                        accumulator = totals.setdefault(key, {"frames": 0.0, "present": 0.0, "distance_sum": 0.0, "angle_sum": 0.0})
                        accumulator["frames"] += 1
                        accumulator["present"] += int(present)
                        accumulator["distance_sum"] += distance
                        accumulator["angle_sum"] += angle
                        if condition is not None:
                            conditioned_key = (*key, condition)
                            condition_accumulator = conditioned.setdefault(conditioned_key, {"frames": 0.0, "present": 0.0})
                            condition_accumulator["frames"] += 1
                            condition_accumulator["present"] += int(present)
                        observation_count += 1
                        if observation_count > int(settings["maximum_feature_observations"]):
                            raise HydrogenBondAnalysisError("maximum_feature_observations gate exceeded")
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append({
                        "severity": "warning", "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        "location": f"{system_id}/{replica_id}/{segment_id}",
                        "message": f"{periodic_frames} periodic frames used local minimum-image hydrogen-bond geometry without connectivity-aware reconstruction",
                    })
                segment_reports.append({
                    "system_id": system_id, "replica_id": replica_id,
                    "segment_id": segment_id, "evaluated_frame_count": evaluated_frames,
                    "periodic_cell_frame_count": periodic_frames,
                })
    occupancies = [{
        "system_id": key[0], "replica_id": key[1], "feature_id": key[2],
        "evaluated_frame_count": int(values["frames"]),
        "present_frame_count": int(values["present"]),
        "occupancy_fraction": values["present"] / values["frames"],
        "mean_donor_acceptor_distance_angstrom": values["distance_sum"] / values["frames"],
        "mean_donor_hydrogen_acceptor_angle_degrees": values["angle_sum"] / values["frames"],
    } for key, values in sorted(totals.items())]
    conditioned_occupancies = [{
        "system_id": key[0], "replica_id": key[1], "feature_id": key[2],
        "cluster_id": key[3], "evaluated_frame_count": int(values["frames"]),
        "present_frame_count": int(values["present"]),
        "occupancy_fraction": values["present"] / values["frames"],
    } for key, values in sorted(conditioned.items())]
    return {
        "module_id": "hydrogen_bonds", "technical_status": "complete",
        "scientific_status": "not evaluated", "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "geometry_contract": {
            "distance": "donor-acceptor distance with minimum-image displacement when a periodic cell is present",
            "angle": "donor-hydrogen-acceptor angle at hydrogen",
            "periodic_minimum_image": True,
            "coordinate_reconstruction": periodic_policy,
            "water_policy": "exclude",
        },
        "condition_model": condition_model["selected_model"] if condition_model else None,
        "segment_reports": segment_reports, "feature_observation_count": observation_count,
        "occupancies": occupancies, "conditioned_occupancies": conditioned_occupancies,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Every donor, hydrogen, and acceptor is explicitly indexed; the module does not guess bond topology.",
            "Water-mediated bonds are excluded until a separately validated water-network contract exists.",
            "allow_wrapped_diagnostic results remain diagnostic; periodic production geometry requires the connectivity-aware make_whole or unwrap_continuous project policy.",
            "Occupancy is definition-dependent and does not establish energetic or mechanistic importance.",
        ],
    }


def hydrogen_bonds_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return hydrogen_bonds_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, HydrogenBondAnalysisError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError, ClusteringAnalysisError,
        PCAAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "hydrogen_bonds", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "HYDROGEN_BOND_INVALID", "message": message} for message in messages],
        }
