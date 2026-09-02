"""Question-driven explicit distance and contact observables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CellVectors, CoordinateReadError, iter_coordinate_frames
from .frame_sampling import frame_selected, plan_frame_selection, reader_frame_indices
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    minimum_image_displacement,
)
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)


class ObservableAnalysisError(ValueError):
    """Raised when an observable specification is unsupported or incomplete."""


def _distance(
    first: Sequence[float],
    second: Sequence[float],
    cell: CellVectors | None = None,
) -> float:
    displacement = tuple(second[index] - first[index] for index in range(3))
    if cell is not None:
        displacement = minimum_image_displacement(displacement, cell)
    return math.sqrt(sum(value * value for value in displacement))


def minimum_group_distance(
    coordinates: Sequence[Sequence[float]],
    group_a: Sequence[int], group_b: Sequence[int],
    cell: CellVectors | None = None,
) -> Tuple[float, Tuple[int, int]]:
    if not group_a or not group_b:
        raise ObservableAnalysisError("distance groups must be nonempty")
    return min(
        (
            _distance(coordinates[left], coordinates[right], cell),
            (left, right),
        )
        for left in group_a for right in group_b
    )


def native_contact_pairs(
    reference_coordinates: Sequence[Sequence[float]],
    atom_indices: Sequence[int],
    reference_cutoff_angstrom: float,
    minimum_atom_index_separation: int,
    cell: CellVectors | None = None,
) -> List[Tuple[int, int]]:
    """Return deterministic reference-defined native pairs for explicit atoms."""

    if max(atom_indices, default=-1) >= len(reference_coordinates):
        raise ObservableAnalysisError("native-contact atom index exceeds reference atom count")
    pairs = []
    for position, left in enumerate(atom_indices[:-1]):
        for right in atom_indices[position + 1 :]:
            if abs(right - left) < minimum_atom_index_separation:
                continue
            if _distance(
                reference_coordinates[left], reference_coordinates[right], cell
            ) <= reference_cutoff_angstrom:
                pairs.append((min(left, right), max(left, right)))
    pairs = sorted(set(pairs))
    if not pairs:
        raise ObservableAnalysisError(
            "native-contact reference definition produces no atom pairs"
        )
    return pairs


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("optional_observables") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise ObservableAnalysisError("definitions.optional_observables must be an object")
    required = {"frame_stride", "features", "maximum_observations"}
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing:
        raise ObservableAnalysisError("observable settings missing: " + ", ".join(missing))
    if unknown:
        raise ObservableAnalysisError("observable settings contain unknown fields: " + ", ".join(unknown))
    for label in ("frame_stride", "maximum_observations"):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ObservableAnalysisError(f"{label} must be a positive integer")
    features = raw["features"]
    if not isinstance(features, list) or not features:
        raise ObservableAnalysisError("features must be a nonempty array")
    normalized = []
    feature_ids = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ObservableAnalysisError(f"features[{index}] must be an object")
        feature_id = str(feature.get("feature_id", "")).strip()
        question = str(feature.get("question", "")).strip()
        kind = feature.get("kind")
        if not feature_id or feature_id in feature_ids or not question:
            raise ObservableAnalysisError("observable feature IDs and questions must be nonempty; IDs must be unique")
        if kind not in {
            "distance", "contact", "group_minimum_distance", "group_contact",
            "native_contact_fraction",
        }:
            raise ObservableAnalysisError(
                "observable kind must be distance, contact, group_minimum_distance, group_contact, or native_contact_fraction"
            )
        expected = {"feature_id", "question", "kind"}
        if kind in {"distance", "contact", "native_contact_fraction"}:
            expected.add("atom_indices")
        else:
            expected.update({"group_a_atom_indices", "group_b_atom_indices"})
        if kind in {"contact", "group_contact"}:
            expected.add("threshold_angstrom")
        if kind == "native_contact_fraction":
            expected.update({
                "reference_cutoff_angstrom", "observation_cutoff_angstrom",
                "minimum_atom_index_separation",
            })
        if set(feature) != expected:
            raise ObservableAnalysisError(
                f"feature {feature_id} fields do not match the {kind} contract"
            )
        index_lists = []
        if kind in {"distance", "contact"}:
            atoms = feature["atom_indices"]
            if not isinstance(atoms, list) or len(atoms) != 2:
                raise ObservableAnalysisError(f"feature {feature_id} atom_indices must contain two indices")
            index_lists.append(atoms)
        elif kind == "native_contact_fraction":
            atoms = feature["atom_indices"]
            if not isinstance(atoms, list) or len(atoms) < 2:
                raise ObservableAnalysisError(
                    f"feature {feature_id} atom_indices must contain at least two indices"
                )
            index_lists.append(atoms)
        else:
            for label in ("group_a_atom_indices", "group_b_atom_indices"):
                values = feature[label]
                if not isinstance(values, list) or not values:
                    raise ObservableAnalysisError(f"feature {feature_id} {label} must be nonempty")
                index_lists.append(values)
        for values in index_lists:
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ObservableAnalysisError("observable atom indices must be nonnegative integers")
            if len(set(values)) != len(values):
                raise ObservableAnalysisError("observable atom index groups cannot contain duplicates")
        if kind in {"contact", "group_contact"}:
            threshold = feature["threshold_angstrom"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
                raise ObservableAnalysisError("contact threshold must be finite and positive")
        if kind == "native_contact_fraction":
            for field in ("reference_cutoff_angstrom", "observation_cutoff_angstrom"):
                value = feature[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    raise ObservableAnalysisError(
                        f"native-contact {field} must be finite and positive"
                    )
            separation = feature["minimum_atom_index_separation"]
            if isinstance(separation, bool) or not isinstance(separation, int) or separation < 1:
                raise ObservableAnalysisError(
                    "minimum_atom_index_separation must be a positive integer"
                )
        feature_ids.add(feature_id)
        normalized.append(dict(feature))
    return {
        "frame_stride": raw["frame_stride"], "features": normalized,
        "maximum_observations": raw["maximum_observations"],
    }


def optional_observables_project(
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
        error_type=ObservableAnalysisError,
    )
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    native_features = [
        feature for feature in settings["features"]
        if feature["kind"] == "native_contact_fraction"
    ]
    reference_atoms = None
    native_pairs_by_feature: Dict[str, List[Tuple[int, int]]] = {}
    if native_features:
        reference_value = project.get("reference_structure")
        if not isinstance(reference_value, str):
            raise ObservableAnalysisError(
                "reference_structure is required for native_contact_fraction"
            )
        reference_path = resolve_manifest_path(reference_value, source)
        _, reference_atoms = read_topology_atoms(reference_path)
        try:
            raw_reference = next(iter_coordinate_frames(reference_path, coordinate_unit))
        except StopIteration as exc:
            raise ObservableAnalysisError("reference_structure has no coordinate frame") from exc
        reference_processor = PeriodicFrameProcessor.from_reference(
            project, source, len(reference_atoms)
        )
        reference_frame = reference_processor.process(
            raw_reference, str(reference_path)
        )
        for feature in native_features:
            native_pairs_by_feature[str(feature["feature_id"])] = native_contact_pairs(
                reference_frame.coordinates_angstrom,
                feature["atom_indices"],
                float(feature["reference_cutoff_angstrom"]),
                int(feature["minimum_atom_index_separation"]),
                reference_frame.cell_vectors_angstrom,
            )
    series: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    segment_reports = []
    observation_count = 0
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
                int(index)
                for feature in settings["features"]
                for index in (
                    list(feature["atom_indices"])
                    if feature["kind"] in {
                        "distance", "contact", "native_contact_fraction"
                    }
                    else list(feature["group_a_atom_indices"])
                    + list(feature["group_b_atom_indices"])
                )
            }))
            for feature in settings["features"]:
                indices = (
                    list(feature["atom_indices"])
                    if feature["kind"] in {
                        "distance", "contact", "native_contact_fraction"
                    }
                    else list(feature["group_a_atom_indices"]) + list(feature["group_b_atom_indices"])
                )
                if max(indices) >= len(atoms):
                    raise ObservableAnalysisError(f"feature {feature['feature_id']} exceeds topology atom count")
                if feature["kind"] == "native_contact_fraction":
                    assert reference_atoms is not None
                    policy = str(project.get("common_atom_policy", "strict"))
                    if any(
                        atoms[index].match_key(policy)
                        != reference_atoms[index].match_key(policy)
                        for index in indices
                    ):
                        raise ObservableAnalysisError(
                            f"feature {feature['feature_id']} native atom identities do not match the reference under {policy} policy"
                        )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = selection_plan[(system_id, replica_id, segment_id)]
                axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                evaluated_frames = 0
                periodic_frames = 0
                processor.begin_segment(
                    bool(segment.get("continuous_with_previous", False))
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, periodic_policy),
                ):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if frame.atom_count != len(atoms):
                        raise ObservableAnalysisError("trajectory/topology atom count mismatch")
                    periodic_frames += int(frame.periodic_cell_present)
                    if not frame_selected(frame.frame_index, selected_indices, 1):
                        continue
                    evaluated_frames += 1
                    axis_value = frame_axis_value(axis, frame.frame_index)
                    for feature in settings["features"]:
                        kind = str(feature["kind"])
                        if kind == "native_contact_fraction":
                            native_pairs = native_pairs_by_feature[str(feature["feature_id"])]
                            distances = [
                                _distance(
                                    frame.coordinates_angstrom[left],
                                    frame.coordinates_angstrom[right],
                                    frame.cell_vectors_angstrom,
                                )
                                for left, right in native_pairs
                            ]
                            contacts = [
                                distance <= float(feature["observation_cutoff_angstrom"])
                                for distance in distances
                            ]
                            row = {
                                "source_frame_index": frame.frame_index,
                                "axis_kind": axis["kind"],
                                "axis_value": axis_value,
                                "native_contact_count": sum(contacts),
                                "native_contact_fraction": sum(contacts) / len(contacts),
                                "native_pair_distances_angstrom": distances,
                                "native_pair_contacts": contacts,
                            }
                            series.setdefault(
                                (system_id, replica_id, str(feature["feature_id"])), []
                            ).append(row)
                            observation_count += 1
                            if observation_count > int(settings["maximum_observations"]):
                                raise ObservableAnalysisError(
                                    "maximum_observations gate exceeded"
                                )
                            continue
                        if kind in {"distance", "contact"}:
                            left, right = feature["atom_indices"]
                            distance = _distance(
                                frame.coordinates_angstrom[left],
                                frame.coordinates_angstrom[right],
                                frame.cell_vectors_angstrom,
                            )
                            pair = (left, right)
                        else:
                            distance, pair = minimum_group_distance(
                                frame.coordinates_angstrom,
                                feature["group_a_atom_indices"], feature["group_b_atom_indices"],
                                frame.cell_vectors_angstrom,
                            )
                        row = {
                            "source_frame_index": frame.frame_index,
                            "axis_kind": axis["kind"],
                            "axis_value": axis_value,
                            "distance_angstrom": distance,
                            "closest_atom_indices": list(pair),
                        }
                        if kind in {"contact", "group_contact"}:
                            row["contact_present"] = distance <= float(feature["threshold_angstrom"])
                        series.setdefault((system_id, replica_id, str(feature["feature_id"])), []).append(row)
                        observation_count += 1
                        if observation_count > int(settings["maximum_observations"]):
                            raise ObservableAnalysisError("maximum_observations gate exceeded")
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append({
                        "severity": "warning", "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        "location": f"{system_id}/{replica_id}/{segment_id}",
                        "message": f"{periodic_frames} periodic frames used wrapped coordinates; pair distances used minimum-image vectors but molecular reconstruction was not performed",
                    })
                segment_reports.append({
                    "system_id": system_id, "replica_id": replica_id,
                    "segment_id": segment_id, "evaluated_frame_count": evaluated_frames,
                    "periodic_cell_frame_count": periodic_frames,
                })
    feature_by_id = {str(feature["feature_id"]): feature for feature in settings["features"]}
    reports = []
    for key in sorted(series):
        rows = series[key]
        feature = feature_by_id[key[2]]
        report = {
            "system_id": key[0], "replica_id": key[1], "feature_id": key[2],
            "question": feature["question"], "kind": feature["kind"],
            "timeseries": rows,
        }
        if feature["kind"] == "native_contact_fraction":
            pairs = native_pairs_by_feature[key[2]]
            report.update({
                "reference_cutoff_angstrom": feature["reference_cutoff_angstrom"],
                "observation_cutoff_angstrom": feature["observation_cutoff_angstrom"],
                "minimum_atom_index_separation": feature["minimum_atom_index_separation"],
                "native_pair_count": len(pairs),
                "native_contact_fraction_summary": sample_summary([
                    float(row["native_contact_fraction"]) for row in rows
                ]),
                "native_pair_occupancies": [
                    {
                        "atom_indices": list(pair),
                        "contact_frame_count": sum(
                            bool(row["native_pair_contacts"][pair_index])
                            for row in rows
                        ),
                        "contact_occupancy_fraction": sum(
                            bool(row["native_pair_contacts"][pair_index])
                            for row in rows
                        ) / len(rows),
                    }
                    for pair_index, pair in enumerate(pairs)
                ],
            })
        else:
            distances = [float(row["distance_angstrom"]) for row in rows]
            report["distance_summary_angstrom"] = sample_summary(distances)
        if feature["kind"] in {"contact", "group_contact"}:
            count = sum(bool(row["contact_present"]) for row in rows)
            report.update({
                "threshold_angstrom": feature["threshold_angstrom"],
                "contact_frame_count": count,
                "contact_occupancy_fraction": count / len(rows),
            })
        reports.append(report)
    return {
        "module_id": "optional_observables", "technical_status": "complete",
        "scientific_status": "not evaluated", "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "distance_contract": "Cartesian distance after the declared periodic-coordinate preprocessing policy",
        "segment_reports": segment_reports, "observation_count": observation_count,
        "feature_reports": reports, "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Every feature carries a declared question and explicit atom indices.",
            "This experimental slice supports pair distances/contacts, group-minimum distances/contacts, and reference-defined native-contact fractions; SASA and domain-specific plugins remain separate extensions.",
            "Wrapped results remain diagnostic; production periodic distances require connectivity-aware make_whole or unwrap_continuous preprocessing.",
            "Contact thresholds are definitions, not evidence of energetic or functional importance.",
            "Native-contact atom indices must retain the declared reference identity; native-contact fractions depend on both reference and observation cutoffs.",
        ],
    }


def optional_observables_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return optional_observables_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, ObservableAnalysisError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "optional_observables", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "OBSERVABLE_INVALID", "message": message} for message in messages],
        }
