"""Replica-aware, streaming atomic root-mean-square fluctuation analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import frame_selected, reader_frame_indices, source_frame_count
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
    sha256_file,
)
from .moments import CoordinateMoments, MomentError, sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .reporting import atom_identity_record, issue_record
from .selections import AtomCorrespondence, build_common_correspondences
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_time,
    normalize_segment_timing,
    require_periodic_policy,
)


_REQUIRED_SETTINGS = {
    "alignment_selection",
    "analysis_selection",
    "minimum_reference_coverage",
    "frame_stride",
    "time_block_size_frames",
    "include_partial_final_block",
    "minimum_replicas_for_uncertainty",
}


class RMSFError(ValueError):
    """Raised when RMSF configuration or inputs cannot be interpreted safely."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise RMSFError("project definitions.pooled_rmsf is required")
    raw = definitions.get("pooled_rmsf")
    if not isinstance(raw, dict):
        raise RMSFError("project definitions.pooled_rmsf must be an object")
    unknown = sorted(set(raw).difference(_REQUIRED_SETTINGS))
    if unknown:
        raise RMSFError(
            "definitions.pooled_rmsf contains unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_REQUIRED_SETTINGS.difference(raw))
    if missing:
        raise RMSFError(
            "definitions.pooled_rmsf is missing required fields: " + ", ".join(missing)
        )
    result: Dict[str, object] = {}
    for field in ("alignment_selection", "analysis_selection"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise RMSFError(f"{field} must be a nonempty selection name")
        result[field] = value
    coverage = raw["minimum_reference_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise RMSFError("minimum_reference_coverage must be between 0 and 1")
    result["minimum_reference_coverage"] = float(coverage)
    for field in (
        "frame_stride",
        "time_block_size_frames",
        "minimum_replicas_for_uncertainty",
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RMSFError(f"{field} must be a positive integer")
        result[field] = value
    include_partial = raw["include_partial_final_block"]
    if not isinstance(include_partial, bool):
        raise RMSFError("include_partial_final_block must be true or false")
    result["include_partial_final_block"] = include_partial
    return result


def _coordinates_at(
    coordinates: Sequence[Tuple[float, float, float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(coordinates[index] for index in indices)
    except IndexError as exc:
        raise RMSFError("atom correspondence index exceeds coordinate atom count") from exc


def _moment_rows(
    moments: CoordinateMoments, atoms: Sequence[AtomRecord]
) -> List[Dict[str, object]]:
    rows = []
    for common_index, atom in enumerate(atoms):
        mean = moments.mean_coordinate(common_index)
        rows.append({
            "common_atom_index": common_index,
            **atom_identity_record(atom),
            "mean_x_angstrom": mean[0],
            "mean_y_angstrom": mean[1],
            "mean_z_angstrom": mean[2],
            "rmsf_angstrom": moments.rmsf(common_index),
        })
    return rows


def _finish_block(
    block: CoordinateMoments,
    block_index: int,
    start_frame_index: int,
    end_frame_index: int,
    start_time: float,
    end_time: float,
    time_unit: str,
    complete: bool,
) -> Dict[str, object]:
    return {
        "block_index": block_index,
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "start_time": start_time,
        "end_time": end_time,
        "time_unit": time_unit,
        "evaluated_frame_count": block.count,
        "complete": complete,
        "rmsf_angstrom_by_common_atom_index": list(block.rmsf_values()),
    }


def _mapping_sets(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    selections: Mapping[str, object],
    settings: Mapping[str, object],
    policy: str,
) -> Tuple[Dict[str, AtomCorrespondence], ...]:
    results: List[Dict[str, AtomCorrespondence]] = [
        {} for _ in target_atom_sets
    ]
    for role, field in (
        ("alignment", "alignment_selection"),
        ("analysis", "analysis_selection"),
    ):
        selection_id = str(settings[field])
        definition = selections.get(selection_id)
        if not isinstance(definition, dict):
            raise RMSFError(f"{field} names undefined selection {selection_id!r}")
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


def pooled_rmsf_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Fit and stream every declared replica into explicit RMSF estimators."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(project_source)
    settings = _settings(project)
    reference_value = project.get("reference_structure")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise RMSFError("reference_structure is required for pooled_rmsf")
    policy_value = project.get("common_atom_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise RMSFError("common_atom_policy is required for pooled_rmsf")
    policy = policy_value.strip()

    context = compile_project_context_file(project_source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
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
        raise RMSFError("reference_structure contains no coordinate frame") from exc
    reference_processor = PeriodicFrameProcessor.from_reference(
        project, project_source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(
        raw_reference_frame, str(reference_path)
    )
    if reference_frame.atom_count != len(reference_atoms):
        raise RMSFError(
            f"reference coordinate count {reference_frame.atom_count} does not match "
            f"reference topology count {len(reference_atoms)}"
        )

    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
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

    issues: List[Dict[str, object]] = list(context["issues"])
    system_reports: List[Dict[str, object]] = []
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        first_key = (system_id, str(replicas[0]["replica_id"]))
        first_mapping = topology_by_key[first_key]["mappings"]
        assert isinstance(first_mapping, dict)
        analysis_atoms = first_mapping["analysis"].reference_atoms
        atom_count = len(analysis_atoms)
        pooled_moments = CoordinateMoments(atom_count)
        replica_moments: List[CoordinateMoments] = []
        included_block_rmsf: List[Tuple[float, ...]] = []
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
            replica_state = CoordinateMoments(atom_count)
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
                source_frames = source_frame_count(
                    trajectory_path, coordinate_unit, error_type=RMSFError
                )
                selected_indices = (
                    set(range(0, source_frames, int(settings["frame_stride"])))
                    if int(settings["frame_stride"]) > 1 else None
                )
                decoded_frames = 0
                evaluated_frames = 0
                periodic_frames = 0
                blocks: List[Dict[str, object]] = []
                block_state = CoordinateMoments(atom_count)
                block_start = -1
                block_end = -1
                block_start_time = 0.0
                block_end_time = 0.0
                first_evaluated_time = None
                last_evaluated_time = None
                discarded_partial_frames = 0
                timing = normalize_segment_timing(segment, time_unit)
                try:
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
                            f"{segment_location}/frame-{raw_frame.frame_index}",
                            reconstruction_atom_indices,
                        )
                        decoded_frames += 1
                        periodic_frames += int(frame.periodic_cell_present)
                        if frame.atom_count != len(target_atoms):
                            raise RMSFError(
                                f"frame {frame.frame_index} has {frame.atom_count} atoms; "
                                f"topology has {len(target_atoms)}"
                            )
                        if not all(
                            math.isfinite(value)
                            for coordinate in frame.coordinates_angstrom
                            for value in coordinate
                        ):
                            raise RMSFError(
                                f"frame {frame.frame_index} contains non-finite coordinates"
                            )
                        if not frame_selected(
                            frame.frame_index,
                            selected_indices,
                            int(settings["frame_stride"]),
                        ):
                            continue
                        evaluated_frames += 1
                        current_time = frame_time(timing, frame.frame_index)
                        if first_evaluated_time is None:
                            first_evaluated_time = current_time
                        last_evaluated_time = current_time
                        mobile_alignment = _coordinates_at(
                            frame.coordinates_angstrom, alignment.target_indices
                        )
                        reference_alignment = _coordinates_at(
                            reference_frame.coordinates_angstrom,
                            alignment.reference_indices,
                        )
                        transform = best_fit_transform(
                            mobile_alignment, reference_alignment
                        )
                        aligned_analysis = apply_transform(
                            _coordinates_at(
                                frame.coordinates_angstrom, analysis.target_indices
                            ),
                            transform,
                        )
                        replica_state.update(aligned_analysis)
                        if block_state.count == 0:
                            block_start = frame.frame_index
                            block_start_time = current_time
                        block_end = frame.frame_index
                        block_end_time = current_time
                        block_state.update(aligned_analysis)
                        if block_state.count == int(settings["time_block_size_frames"]):
                            block_record = _finish_block(
                                block_state,
                                len(blocks),
                                block_start,
                                block_end,
                                block_start_time,
                                block_end_time,
                                time_unit,
                                True,
                            )
                            blocks.append(block_record)
                            included_block_rmsf.append(block_state.rmsf_values())
                            block_state = CoordinateMoments(atom_count)
                    if decoded_frames == 0:
                        raise RMSFError("trajectory contains no readable coordinate frames")
                    if block_state.count:
                        if bool(settings["include_partial_final_block"]):
                            block_record = _finish_block(
                                block_state,
                                len(blocks),
                                block_start,
                                block_end,
                                block_start_time,
                                block_end_time,
                                time_unit,
                                False,
                            )
                            blocks.append(block_record)
                            included_block_rmsf.append(block_state.rmsf_values())
                        else:
                            discarded_partial_frames = block_state.count
                except (
                    CoordinateReadError, GeometryError, MomentError,
                    PeriodicReconstructionError, RMSFError, OSError,
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
                    "observed_frame_count": source_frames,
                    "decoded_frame_count": decoded_frames,
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
                    "discarded_partial_block_frame_count": discarded_partial_frames,
                    "time_blocks": blocks,
                })
            if replica_state.count:
                # Merge the replica's count, mean, and centered second moments.
                # The between-replica mean correction in CoordinateMoments.merge
                # makes this identical to one pooled serial stream and permits a
                # future worker to own the replica scan without redefining RMSF.
                pooled_moments.merge(replica_state)
                replica_moments.append(replica_state)
                replica_statistics = _moment_rows(replica_state, analysis_atoms)
            else:
                replica_statistics = []
                issues.append(issue_record(
                    "error", "NO_EVALUATED_FRAMES", location,
                    "replica produced no evaluated frames",
                ))
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
                "evaluated_frame_count": replica_state.count,
                "atom_statistics": replica_statistics,
                "segments": segment_reports,
            })

        if len(replica_moments) < int(settings["minimum_replicas_for_uncertainty"]):
            issues.append(issue_record(
                "warning",
                "REPLICA_UNCERTAINTY_UNAVAILABLE",
                system_id,
                f"{len(replica_moments)} replicas produced estimates; "
                f"{settings['minimum_replicas_for_uncertainty']} were requested for uncertainty",
            ))
        system_rows: List[Dict[str, object]] = []
        if pooled_moments.count:
            replica_rmsf = [moments.rmsf_values() for moments in replica_moments]
            for atom_index, atom in enumerate(analysis_atoms):
                mean_coordinate = pooled_moments.mean_coordinate(atom_index)
                system_rows.append({
                    "common_atom_index": atom_index,
                    **atom_identity_record(atom),
                    "frame_pooled_mean_x_angstrom": mean_coordinate[0],
                    "frame_pooled_mean_y_angstrom": mean_coordinate[1],
                    "frame_pooled_mean_z_angstrom": mean_coordinate[2],
                    "frame_pooled_rmsf_angstrom": pooled_moments.rmsf(atom_index),
                    "replica_rmsf_summary_angstrom": sample_summary(
                        values[atom_index] for values in replica_rmsf
                    ),
                    "time_block_rmsf_summary_angstrom": sample_summary(
                        values[atom_index] for values in included_block_rmsf
                    ),
                })
        system_reports.append({
            "system_id": system_id,
            "frame_pooled_sample_count": pooled_moments.count,
            "replica_estimate_count": len(replica_moments),
            "included_time_block_count": len(included_block_rmsf),
            "atom_statistics": system_rows,
            "replicas": replica_reports,
        })

    if int(settings["frame_stride"]) > 1:
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            f"RMSF was evaluated every {settings['frame_stride']} frames",
        ))
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "module_id": "pooled_rmsf",
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
        "settings": settings,
        "common_atom_policy": policy,
        "periodic_coordinate_policy": periodic_policy,
        "time_unit": time_unit,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "systems": system_reports,
        "limitations": [
            "RMSF is sqrt(<|r-<r>|^2>) over fitted positions with population denominator N.",
            "Frame-pooled RMSF is frame weighted; replica RMSF summaries give each replica one estimate and are not the same estimator.",
            "Between-replica sample SD and SEM are absent when fewer than two replica estimates exist.",
            "Time blocks reset at every segment boundary and retain their explicit evaluated-frame count plus manifest-declared physical start and end times.",
            "Partial final blocks are either explicitly included and flagged or explicitly discarded according to project settings.",
            "Time-block estimates are diagnostics and are not assumed statistically independent.",
            "Periodic production analysis requires make_whole or unwrap_continuous with explicit connectivity; allow_wrapped_diagnostic remains diagnostic only.",
            "A technically complete RMSF report does not establish equilibration, convergence, adequate sampling, functional importance, or scientific validity.",
            "No real-project regression fixture has yet been approved; status remains experimental.",
        ],
    }


def pooled_rmsf_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Convert configuration, mapping, and input failures into a JSON report."""

    try:
        return pooled_rmsf_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        RMSFError,
        AtomMappingError,
        CoordinateReadError,
        GeometryError,
        MomentError,
        PeriodicReconstructionError,
        TrajectoryContractError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "pooled_rmsf",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "RMSF_INVALID", "message": message}
                for message in messages
            ],
        }
