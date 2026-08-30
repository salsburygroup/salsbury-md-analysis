"""Streaming, common-basis dynamic cross-correlation matrices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import ManifestValidationError, load_json, resolve_manifest_path, sha256_file
from .moments import DisplacementCovariance, MomentError
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    merge_frame_selection_reports,
    restore_source_provenance,
    unique_issues,
)
from .reporting import atom_identity_record, issue_record
from .selections import AtomCorrespondence, build_common_correspondences
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_time,
    normalize_segment_timing,
    require_periodic_policy,
)
from .upstream_cache import load_cached_project_report


_REQUIRED_SETTINGS = {
    "alignment_selection",
    "analysis_selection",
    "minimum_reference_coverage",
    "frame_stride",
    "maximum_atoms",
    "minimum_evaluated_frames_per_replica",
    "minimum_variance_angstrom2",
}


class DCCMError(ValueError):
    """Raised when a DCCM contract or input cannot be interpreted safely."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise DCCMError("project definitions.dccm is required")
    raw = definitions.get("dccm")
    if not isinstance(raw, dict):
        raise DCCMError("project definitions.dccm must be an object")
    unknown = sorted(set(raw).difference(_REQUIRED_SETTINGS | {"frame_selection"}))
    if unknown:
        raise DCCMError("definitions.dccm contains unknown fields: " + ", ".join(unknown))
    missing = sorted(_REQUIRED_SETTINGS.difference(raw))
    if missing:
        raise DCCMError("definitions.dccm is missing required fields: " + ", ".join(missing))
    result: Dict[str, object] = {}
    for field in ("alignment_selection", "analysis_selection"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise DCCMError(f"{field} must be a nonempty selection name")
        result[field] = value
    coverage = raw["minimum_reference_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise DCCMError("minimum_reference_coverage must be between 0 and 1")
    result["minimum_reference_coverage"] = float(coverage)
    for field in (
        "frame_stride",
        "maximum_atoms",
        "minimum_evaluated_frames_per_replica",
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DCCMError(f"{field} must be a positive integer")
        result[field] = value
    minimum_variance = raw["minimum_variance_angstrom2"]
    if (
        isinstance(minimum_variance, bool)
        or not isinstance(minimum_variance, (int, float))
        or not math.isfinite(float(minimum_variance))
        or float(minimum_variance) <= 0.0
    ):
        raise DCCMError("minimum_variance_angstrom2 must be finite and positive")
    result["minimum_variance_angstrom2"] = float(minimum_variance)
    result["frame_selection"] = normalize_frame_selection(
        raw.get("frame_selection"), int(result["frame_stride"]),
        error_type=DCCMError,
    )
    return result


def _coordinates_at(
    coordinates: Sequence[Tuple[float, float, float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(coordinates[index] for index in indices)
    except IndexError as exc:
        raise DCCMError("atom correspondence index exceeds coordinate atom count") from exc


def _mapping_sets(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    selections: Mapping[str, object],
    settings: Mapping[str, object],
    policy: str,
) -> Tuple[Dict[str, AtomCorrespondence], ...]:
    results: List[Dict[str, AtomCorrespondence]] = [{} for _ in target_atom_sets]
    for role, field in (
        ("alignment", "alignment_selection"),
        ("analysis", "analysis_selection"),
    ):
        selection_id = str(settings[field])
        definition = selections.get(selection_id)
        if not isinstance(definition, dict):
            raise DCCMError(f"{field} names undefined selection {selection_id!r}")
        mappings = build_common_correspondences(
            reference_atoms,
            target_atom_sets,
            definition,
            selection_id,
            policy,
            float(settings["minimum_reference_coverage"]),
        )
        for result, mapping in zip(results, mappings):
            result[role] = mapping
    return tuple(results)


def _matrix_payload(
    state: DisplacementCovariance, minimum_variance: float
) -> Dict[str, object]:
    matrix = state.correlation_matrix(minimum_variance)
    undefined = sum(
        matrix[left][right] is None
        for left in range(len(matrix))
        for right in range(left, len(matrix))
    )
    off_diagonal = [
        float(matrix[left][right])
        for left in range(len(matrix))
        for right in range(left + 1, len(matrix))
        if matrix[left][right] is not None
    ]
    return {
        "evaluated_frame_count": state.count,
        "undefined_upper_triangle_entry_count": undefined,
        "valid_off_diagonal_summary": {
            "count": len(off_diagonal),
            "minimum": min(off_diagonal) if off_diagonal else None,
            "maximum": max(off_diagonal) if off_diagonal else None,
            "mean": sum(off_diagonal) / len(off_diagonal) if off_diagonal else None,
        },
        "matrix": [list(row) for row in matrix],
    }


def _difference_matrix(
    matrix: Sequence[Sequence[object]], reference: Sequence[Sequence[object]]
) -> Tuple[List[List[object]], int]:
    difference: List[List[object]] = []
    undefined = 0
    for row, reference_row in zip(matrix, reference):
        output_row: List[object] = []
        for value, reference_value in zip(row, reference_row):
            if value is None or reference_value is None:
                output_row.append(None)
                undefined += 1
            else:
                output_row.append(float(value) - float(reference_value))
        difference.append(output_row)
    return difference, undefined


def _dccm_project_serial(
    project_path: Path,
    hash_content: bool = False,
    *,
    allow_incomplete_pooled_reference: bool = False,
) -> Dict[str, object]:
    """Calculate per-replica and system-pooled DCCMs on one global atom basis."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "dccm",
        project_source,
        hash_content=hash_content,
        error_type=DCCMError,
    )
    if cached is not None:
        return cached
    project = load_json(project_source)
    settings = _settings(project)
    reference_value = project.get("reference_structure")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise DCCMError("reference_structure is required for dccm")
    policy_value = project.get("common_atom_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise DCCMError("common_atom_policy is required for dccm")
    policy = policy_value.strip()

    context = compile_project_context_file(project_source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    reference_system = str(contract["reference_system"])
    assert isinstance(selections, dict)
    assert isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    time_unit = str(units["time"])
    periodic_policy = require_periodic_policy(
        contract.get("periodic_coordinate_policy")
    )
    reference_path = resolve_manifest_path(reference_value, project_source)
    reference_format, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference_frame = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise DCCMError("reference_structure contains no coordinate frame") from exc
    reference_processor = PeriodicFrameProcessor.from_reference(
        project, project_source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(
        raw_reference_frame, str(reference_path)
    )
    if reference_frame.atom_count != len(reference_atoms):
        raise DCCMError(
            f"reference coordinate count {reference_frame.atom_count} does not match "
            f"reference topology count {len(reference_atoms)}"
        )

    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system_manifest, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=DCCMError,
    )
    inventory = context["input_inventory"]
    assert isinstance(inventory, dict)
    entries = inventory["entries"]
    assert isinstance(entries, list)
    inventory_by_path = {
        str(entry["resolved_path"]): entry
        for entry in entries
        if isinstance(entry, dict)
    }

    topology_records: List[Dict[str, object]] = []
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            topology_format, target_atoms = read_topology_atoms(topology_path)
            topology_records.append({
                "key": (system_id, replica_id),
                "path": topology_path,
                "format": topology_format,
                "atoms": target_atoms,
            })
    mappings = _mapping_sets(
        reference_atoms,
        [record["atoms"] for record in topology_records],
        selections,
        settings,
        policy,
    )
    topology_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record, mapping in zip(topology_records, mappings):
        record["mappings"] = mapping
        key = record["key"]
        assert isinstance(key, tuple)
        topology_by_key[key] = record
    first_analysis = mappings[0]["analysis"]
    analysis_atoms = first_analysis.reference_atoms
    atom_count = len(analysis_atoms)
    if atom_count > int(settings["maximum_atoms"]):
        raise DCCMError(
            f"common analysis selection contains {atom_count} atoms; maximum_atoms is "
            f"{settings['maximum_atoms']} because DCCM scales quadratically"
        )

    issues: List[Dict[str, object]] = list(context["issues"])
    system_reports: List[Dict[str, object]] = []
    system_states: Dict[str, DisplacementCovariance] = {}
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        system_state = DisplacementCovariance(atom_count)
        replica_reports: List[Dict[str, object]] = []
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            location = f"{system_id}/{replica_id}"
            topology = topology_by_key[(system_id, replica_id)]
            topology_path = topology["path"]
            topology_format = topology["format"]
            target_atoms = topology["atoms"]
            mapping = topology["mappings"]
            assert isinstance(topology_path, Path)
            assert isinstance(topology_format, str)
            assert isinstance(target_atoms, list)
            assert isinstance(mapping, dict)
            alignment = mapping["alignment"]
            analysis = mapping["analysis"]
            assert isinstance(alignment, AtomCorrespondence)
            assert isinstance(analysis, AtomCorrespondence)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(target_atoms)
            )
            reconstruction_atom_indices = tuple(sorted(
                set(alignment.target_indices) | set(analysis.target_indices)
            ))
            if policy == "position":
                for role, correspondence in mapping.items():
                    if correspondence.residue_name_mismatch_count:
                        issues.append(issue_record(
                            "warning",
                            "RESIDUE_NAME_MISMATCH",
                            f"{location}/{role}",
                            f"{correspondence.residue_name_mismatch_count} mapped atoms differ in residue name "
                            "because the position policy ignores residue substitutions",
                        ))
            replica_state = DisplacementCovariance(atom_count)
            segment_reports: List[Dict[str, object]] = []
            replica_error_start = sum(issue["severity"] == "error" for issue in issues)
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                segment_location = f"{location}/{segment_id}"
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                observed_frames = 0
                evaluated_frames = 0
                periodic_frames = 0
                first_evaluated_time = None
                last_evaluated_time = None
                timing = normalize_segment_timing(segment, time_unit)
                try:
                    processor.begin_segment(
                        bool(segment.get("continuous_with_previous", False))
                    )
                    reader_indices = reader_frame_indices(
                        selected_indices, processor.policy
                    )
                    for raw_frame in iter_coordinate_frames(
                        trajectory_path, coordinate_unit, reader_indices
                    ):
                        selected = frame_selected(
                            raw_frame.frame_index, selected_indices,
                            int(settings["frame_stride"]),
                        )
                        if not selected and processor.policy != "unwrap_continuous":
                            continue
                        frame = processor.process(
                            raw_frame,
                            f"{segment_location}/frame-{raw_frame.frame_index}",
                            reconstruction_atom_indices,
                        )
                        observed_frames += 1
                        periodic_frames += int(frame.periodic_cell_present)
                        if frame.atom_count != len(target_atoms):
                            raise DCCMError(
                                f"frame {frame.frame_index} has {frame.atom_count} atoms; "
                                f"topology has {len(target_atoms)}"
                            )
                        if not all(
                            math.isfinite(value)
                            for coordinate in frame.coordinates_angstrom
                            for value in coordinate
                        ):
                            raise DCCMError(
                                f"frame {frame.frame_index} contains non-finite coordinates"
                            )
                        if not selected:
                            continue
                        evaluated_frames += 1
                        current_time = frame_time(timing, frame.frame_index)
                        if first_evaluated_time is None:
                            first_evaluated_time = current_time
                        last_evaluated_time = current_time
                        transform = best_fit_transform(
                            _coordinates_at(
                                frame.coordinates_angstrom, alignment.target_indices
                            ),
                            _coordinates_at(
                                reference_frame.coordinates_angstrom,
                                alignment.reference_indices,
                            ),
                        )
                        replica_state.update(apply_transform(
                            _coordinates_at(
                                frame.coordinates_angstrom, analysis.target_indices
                            ),
                            transform,
                        ))
                    if observed_frames == 0:
                        raise DCCMError("trajectory contains no readable coordinate frames")
                except (
                    CoordinateReadError, GeometryError, MomentError,
                    PeriodicReconstructionError, DCCMError, OSError,
                ) as exc:
                    issues.append(issue_record(
                        "error", "TRAJECTORY_ANALYSIS_FAILED", segment_location, str(exc)
                    ))
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append(issue_record(
                        "warning",
                        "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        segment_location,
                        f"{periodic_frames} frames declare a periodic cell; this module does not unwrap molecules",
                    ))
                trajectory_record = inventory_by_path.get(str(trajectory_path), {})
                segment_reports.append({
                    "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": trajectory_record.get("sha256"),
                    "observed_frame_count": observed_frames,
                    "evaluated_frame_count": evaluated_frames,
                    "periodic_cell_frame_count": periodic_frames,
                    "timing": timing,
                    "evaluated_time_range": (
                        {
                            "start": first_evaluated_time,
                            "end": last_evaluated_time,
                            "unit": time_unit,
                        }
                        if first_evaluated_time is not None
                        else None
                    ),
                })
            minimum_frames = int(settings["minimum_evaluated_frames_per_replica"])
            if replica_state.count < minimum_frames:
                issues.append(issue_record(
                    "warning",
                    "INSUFFICIENT_REPLICA_FRAMES_FOR_REPLICA_DCCM",
                    location,
                    f"replica has {replica_state.count} evaluated frames; its local DCCM "
                    f"requires {minimum_frames}, so only the system-pooled DCCM uses these frames",
                ))
            if replica_state.count:
                system_state.merge(replica_state)
                if replica_state.count >= minimum_frames:
                    matrix_payload = _matrix_payload(
                        replica_state, float(settings["minimum_variance_angstrom2"])
                    )
                    if matrix_payload["undefined_upper_triangle_entry_count"]:
                        issues.append(issue_record(
                            "warning",
                            "UNDEFINED_ZERO_VARIANCE_CORRELATIONS",
                            location,
                            f"{matrix_payload['undefined_upper_triangle_entry_count']} upper-triangle entries are null because at least one atom is below the variance gate",
                        ))
                else:
                    matrix_payload = None
            else:
                matrix_payload = None
            topology_inventory = inventory_by_path.get(str(topology_path), {})
            replica_reports.append({
                "replica_id": replica_id,
                "technical_status": (
                    "failed"
                    if sum(issue["severity"] == "error" for issue in issues)
                    > replica_error_start
                    else "complete"
                ),
                "topology_path": str(topology_path),
                "topology_format": topology_format,
                "topology_sha256": topology_inventory.get("sha256"),
                "topology_atom_count": len(target_atoms),
                "periodic_reconstruction": processor.report(),
                "mappings": {
                    role: correspondence.as_dict()
                    for role, correspondence in mapping.items()
                },
                "dccm": matrix_payload,
                "mergeable_displacement_covariance": replica_state.to_state(),
                "segments": segment_reports,
            })
        if system_state.count >= int(settings["minimum_evaluated_frames_per_replica"]):
            pooled_payload = _matrix_payload(
                system_state, float(settings["minimum_variance_angstrom2"])
            )
            if pooled_payload["undefined_upper_triangle_entry_count"]:
                issues.append(issue_record(
                    "warning",
                    "UNDEFINED_ZERO_VARIANCE_CORRELATIONS",
                    system_id,
                    f"{pooled_payload['undefined_upper_triangle_entry_count']} pooled upper-triangle entries are null because at least one atom is below the variance gate",
                ))
        else:
            pooled_payload = None
            issues.append(issue_record(
                "error", "INSUFFICIENT_SYSTEM_FRAMES", system_id,
                f"system produced {system_state.count} pooled evaluated frames; at least "
                f"{settings['minimum_evaluated_frames_per_replica']} are required",
            ))
        system_states[system_id] = system_state
        system_reports.append({
            "system_id": system_id,
            "frame_pooled_dccm": pooled_payload,
            "replicas": replica_reports,
        })

    reference_report = next(
        report for report in system_reports if report["system_id"] == reference_system
    )
    reference_payload = reference_report["frame_pooled_dccm"]
    if not isinstance(reference_payload, dict) and not allow_incomplete_pooled_reference:
        raise DCCMError("reference system produced no pooled DCCM")
    reference_matrix = (
        reference_payload["matrix"] if isinstance(reference_payload, dict) else None
    )
    if reference_matrix is not None:
        assert isinstance(reference_matrix, list)
    for report in system_reports:
        payload = report["frame_pooled_dccm"]
        if not isinstance(payload, dict) or reference_matrix is None:
            report["difference_from_reference_dccm"] = None
            continue
        matrix = payload["matrix"]
        assert isinstance(matrix, list)
        difference, undefined = _difference_matrix(matrix, reference_matrix)
        report["difference_from_reference_dccm"] = {
            "reference_system_id": reference_system,
            "undefined_entry_count": undefined,
            "matrix": difference,
        }

    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            "DCCM evaluated "
            f"{frame_selection_report['selected_frame_count']} of "
            f"{frame_selection_report['source_frame_count']} source frames under "
            f"{frame_selection_report['mode']}",
        ))
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "module_id": "dccm",
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(project_source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "reference": {
            "path": str(reference_path),
            "format": reference_format,
            "sha256": sha256_file(reference_path) if hash_content else None,
            "atom_count": len(reference_atoms),
            "frame_index": reference_frame.frame_index,
        },
        "reference_system_id": reference_system,
        "settings": settings,
        "frame_selection": frame_selection_report,
        "common_atom_policy": policy,
        "periodic_coordinate_policy": periodic_policy,
        "time_unit": time_unit,
        "analysis_atoms": [
            atom_identity_record(atom, index) for index, atom in enumerate(analysis_atoms)
        ],
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "systems": system_reports,
        "limitations": [
            "DCCM is the normalized dot-product covariance of fitted Cartesian displacement vectors.",
            "Entries are null when either atom falls below the declared positional-variance gate; zero correlation is not invented.",
            "The maximum_atoms gate is mandatory because time and memory scale quadratically with the analysis selection.",
            "System-pooled matrices are frame weighted; per-replica matrices remain available and pooled frames are not uncertainty units.",
            "A replica with fewer than the declared local-matrix minimum contributes its frames to the system-pooled DCCM but has no standalone replica DCCM.",
            "Difference matrices subtract the declared reference-system DCCM only where both entries are defined.",
            "Generalized correlation, mutual information, causality, directionality, and statistical significance are not computed.",
            "Periodic production analysis requires make_whole or unwrap_continuous with explicit connectivity; allow_wrapped_diagnostic remains diagnostic only.",
            "A technically complete DCCM does not establish equilibration, convergence, adequate sampling, mechanism, or scientific validity.",
            "No real-project regression fixture has yet been approved; status remains experimental.",
        ],
    }


def _reduce_dccm_replica_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in (
            "module_id", "reference", "settings", "common_atom_policy",
            "periodic_coordinate_policy", "time_unit", "analysis_atoms",
        ):
            if report.get(key) != first.get(key):
                raise DCCMError(f"replica DCCM reports disagree on {key}")
    first["frame_selection"] = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for report in reports:
        for system in report.get("systems", []):
            if isinstance(system, dict):
                grouped.setdefault(str(system["system_id"]), []).extend(
                    row for row in system.get("replicas", []) if isinstance(row, dict)
                )
    issues = [
        issue for issue in unique_issues(reports)
        if issue.get("code") not in {
            "INSUFFICIENT_SYSTEM_FRAMES", "UNDEFINED_ZERO_VARIANCE_CORRELATIONS",
        }
    ]
    minimum = int(first["settings"]["minimum_evaluated_frames_per_replica"])  # type: ignore[index]
    minimum_variance = float(first["settings"]["minimum_variance_angstrom2"])  # type: ignore[index]
    system_reports = []
    for system_id, replica_rows in grouped.items():
        states = [
            DisplacementCovariance.from_state(row["mergeable_displacement_covariance"])
            for row in replica_rows
        ]
        pooled = DisplacementCovariance(states[0].atom_count)
        for state in states:
            pooled.merge(state)
        for replica_row, state in zip(replica_rows, states):
            matrix = (
                _matrix_payload(state, minimum_variance)
                if state.count >= minimum else None
            )
            replica_row["dccm"] = matrix
            if matrix and matrix["undefined_upper_triangle_entry_count"]:
                issues.append(issue_record(
                    "warning", "UNDEFINED_ZERO_VARIANCE_CORRELATIONS",
                    f"{system_id}/{replica_row['replica_id']}",
                    f"{matrix['undefined_upper_triangle_entry_count']} upper-triangle entries are null because at least one atom is below the variance gate",
                ))
        if pooled.count >= minimum:
            pooled_payload = _matrix_payload(pooled, minimum_variance)
            if pooled_payload["undefined_upper_triangle_entry_count"]:
                issues.append(issue_record(
                    "warning", "UNDEFINED_ZERO_VARIANCE_CORRELATIONS", system_id,
                    f"{pooled_payload['undefined_upper_triangle_entry_count']} pooled upper-triangle entries are null because at least one atom is below the variance gate",
                ))
        else:
            pooled_payload = None
            issues.append(issue_record(
                "error", "INSUFFICIENT_SYSTEM_FRAMES", system_id,
                f"system produced {pooled.count} pooled evaluated frames; at least {minimum} are required",
            ))
        system_reports.append({
            "system_id": system_id,
            "frame_pooled_dccm": pooled_payload,
            "replicas": replica_rows,
        })
    contract = source_context.get("contract")
    if not isinstance(contract, dict):
        raise DCCMError("source context contract is missing")
    reference_system = str(contract["reference_system"])
    reference_report = next(
        (row for row in system_reports if row["system_id"] == reference_system), None
    )
    if reference_report is None or not isinstance(
        reference_report.get("frame_pooled_dccm"), dict
    ):
        raise DCCMError("reference system produced no pooled DCCM")
    reference_matrix = reference_report["frame_pooled_dccm"]["matrix"]
    for system in system_reports:
        payload = system["frame_pooled_dccm"]
        if not isinstance(payload, dict):
            system["difference_from_reference_dccm"] = None
            continue
        difference, undefined = _difference_matrix(payload["matrix"], reference_matrix)
        system["difference_from_reference_dccm"] = {
            "reference_system_id": reference_system,
            "undefined_entry_count": undefined,
            "matrix": difference,
        }
    first["reference_system_id"] = reference_system
    first["systems"] = system_reports
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(issue.get("severity") == "warning" for issue in issues)
    first["technical_status"] = "failed" if first["error_count"] else "complete"
    restore_source_provenance(first, source_context)
    return first


def dccm_project(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Scan replicas independently and merge exact displacement covariance."""

    project = load_json(Path(project_path).expanduser().resolve(strict=False))
    settings = _settings(project)
    selection = settings.get("frame_selection")
    if isinstance(selection, dict) and selection.get("mode") == "auto_resource_budget_v1":
        return _dccm_project_serial(project_path, hash_content=hash_content)
    return execute_replica_final_module(
        project_path,
        runner_id="dccm",
        hash_content=hash_content,
        reducer=_reduce_dccm_replica_reports,
    )


def dccm_project_safe(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Convert configuration, mapping, and input failures into a JSON report."""

    try:
        return dccm_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        DCCMError,
        AtomMappingError,
        CoordinateReadError,
        GeometryError,
        MomentError,
        PeriodicReconstructionError,
        StopIteration,
        TrajectoryContractError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "dccm",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "DCCM_INVALID", "message": message}
                for message in messages
            ],
        }
