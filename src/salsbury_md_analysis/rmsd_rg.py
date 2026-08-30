"""Replica-resolved RMSD and mass-weighted radius-of-gyration analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected,
    integer_stride_indices,
    reader_frame_indices,
    source_frame_count,
)
from .geometry import (
    GeometryError,
    apply_transform,
    best_fit_transform,
    mass_weighted_radius_of_gyration,
    rmsd,
)
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
    sha256_file,
)
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    restore_source_provenance,
    unique_issues,
)
from .reporting import issue_record
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
    "rmsd_selection",
    "rg_selection",
    "minimum_reference_coverage",
    "frame_stride",
}
_ATOMIC_MASSES = {
    "H": 1.008,
    "D": 2.014,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403,
    "NA": 22.98977,
    "MG": 24.305,
    "P": 30.973762,
    "S": 32.06,
    "CL": 35.45,
    "K": 39.0983,
    "CA": 40.078,
    "MN": 54.938,
    "FE": 55.845,
    "CO": 58.933,
    "NI": 58.693,
    "CU": 63.546,
    "ZN": 65.38,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90447,
}


class RMSDRGError(ValueError):
    """Raised when RMSD/Rg configuration or inputs are unsafe to interpret."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise RMSDRGError(
            "project definitions.replica_rmsd_rg is required for explicit analysis gates"
        )
    raw = definitions.get("replica_rmsd_rg")
    if not isinstance(raw, dict):
        raise RMSDRGError("project definitions.replica_rmsd_rg must be an object")
    unknown = sorted(set(raw).difference(_REQUIRED_SETTINGS))
    if unknown:
        raise RMSDRGError(
            "definitions.replica_rmsd_rg contains unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_REQUIRED_SETTINGS.difference(raw))
    if missing:
        raise RMSDRGError(
            "definitions.replica_rmsd_rg is missing required fields: " + ", ".join(missing)
        )
    selection_fields = ("alignment_selection", "rmsd_selection", "rg_selection")
    result: Dict[str, object] = {}
    for field in selection_fields:
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise RMSDRGError(f"{field} must be a nonempty selection name")
        result[field] = value
    coverage = raw["minimum_reference_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise RMSDRGError("minimum_reference_coverage must be between 0 and 1")
    stride = raw["frame_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise RMSDRGError("frame_stride must be a positive integer")
    result["minimum_reference_coverage"] = float(coverage)
    result["frame_stride"] = stride
    return result


def _coordinates_at(
    coordinates: Sequence[Tuple[float, float, float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(coordinates[index] for index in indices)
    except IndexError as exc:
        raise RMSDRGError("atom correspondence index exceeds coordinate atom count") from exc


def atomic_masses(atoms: Sequence[AtomRecord]) -> Tuple[float, ...]:
    """Resolve declared/inferred elements to fixed standard atomic masses."""

    masses = []
    for atom in atoms:
        element = atom.element.strip().upper()
        try:
            masses.append(_ATOMIC_MASSES[element])
        except KeyError as exc:
            raise RMSDRGError(
                f"no atomic mass is registered for element {element or '<empty>'!r} "
                f"at topology atom index {atom.atom_index} ({atom.atom_name})"
            ) from exc
    return tuple(masses)


def _summary(rows: Sequence[Mapping[str, object]], field: str) -> Dict[str, object]:
    values = [float(row[field]) for row in rows]
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _mapping_bundles(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    selections: Mapping[str, object],
    settings: Mapping[str, object],
    policy: str,
) -> Tuple[Dict[str, AtomCorrespondence], ...]:
    result: List[Dict[str, AtomCorrespondence]] = [
        {} for _ in target_atom_sets
    ]
    # Alignment and RMSD compare every topology to the declared reference and
    # therefore require one common reference-ordered atom basis.  Rg is an
    # intrinsic scalar for each topology: forcing its (often broader) solute
    # selection through the global common-atom intersection silently drops
    # system-specific ligands/ions and can fail otherwise valid comparative
    # campaigns.  Build the Rg selection independently on each topology.
    for role, field in (
        ("alignment", "alignment_selection"),
        ("rmsd", "rmsd_selection"),
    ):
        selection_id = str(settings[field])
        definition = selections.get(selection_id)
        if not isinstance(definition, dict):
            raise RMSDRGError(
                f"{field} names undefined selection {selection_id!r}"
            )
        mappings = build_common_correspondences(
            reference_atoms,
            target_atom_sets,
            definition,
            selection_id,
            policy,
            float(settings["minimum_reference_coverage"]),
        )
        for bundle, mapping in zip(result, mappings):
            bundle[role] = mapping
    rg_selection_id = str(settings["rg_selection"])
    rg_definition = selections.get(rg_selection_id)
    if not isinstance(rg_definition, dict):
        raise RMSDRGError(
            f"rg_selection names undefined selection {rg_selection_id!r}"
        )
    for bundle, target_atoms in zip(result, target_atom_sets):
        bundle["rg"] = build_common_correspondences(
            target_atoms,
            (target_atoms,),
            rg_definition,
            rg_selection_id,
            policy,
            1.0,
        )[0]
    return tuple(result)


def _replica_rmsd_rg_project_serial(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Analyze every declared replica/segment without writing analysis outputs."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "replica_rmsd_rg",
        project_source,
        hash_content=hash_content,
        error_type=RMSDRGError,
    )
    if cached is not None:
        return cached
    project = load_json(project_source)
    settings = _settings(project)
    reference_value = project.get("reference_structure")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise RMSDRGError("reference_structure is required for replica_rmsd_rg")
    policy_value = project.get("common_atom_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise RMSDRGError("common_atom_policy is required for replica_rmsd_rg")
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
    if reference_format not in {"pdb", "gro"}:
        raise RMSDRGError("reference_structure must be PDB or GRO")
    try:
        raw_reference_frame = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise RMSDRGError("reference_structure contains no coordinate frame") from exc
    reference_processor = PeriodicFrameProcessor.from_reference(
        project, project_source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(
        raw_reference_frame, str(reference_path)
    )
    if reference_frame.atom_count != len(reference_atoms):
        raise RMSDRGError(
            f"reference coordinate count {reference_frame.atom_count} does not match "
            f"reference topology count {len(reference_atoms)}"
        )

    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    inventory = context["input_inventory"]
    assert isinstance(inventory, dict)
    entries = inventory["entries"]
    assert isinstance(entries, list)
    inventory_by_path = {
        str(entry["resolved_path"]): entry
        for entry in entries
        if isinstance(entry, dict)
    }

    issues: List[Dict[str, object]] = list(context["issues"])
    system_reports: List[Dict[str, object]] = []
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
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
    mapping_bundles = _mapping_bundles(
        reference_atoms,
        [record["atoms"] for record in topology_records],
        selections,
        settings,
        policy,
    )
    topology_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record, mappings in zip(topology_records, mapping_bundles):
        record["mappings"] = mappings
        record["rg_masses"] = atomic_masses(mappings["rg"].target_atoms)
        key = record["key"]
        assert isinstance(key, tuple)
        topology_by_key[key] = record

    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replica_reports: List[Dict[str, object]] = []
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            replica_location = f"{system_id}/{replica_id}"
            topology_record = topology_by_key[(system_id, replica_id)]
            topology_path = topology_record["path"]
            topology_format = topology_record["format"]
            target_atoms = topology_record["atoms"]
            mappings = topology_record["mappings"]
            masses = topology_record["rg_masses"]
            assert isinstance(topology_path, Path)
            assert isinstance(topology_format, str)
            assert isinstance(target_atoms, list)
            assert isinstance(mappings, dict)
            assert isinstance(masses, tuple)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(target_atoms)
            )
            reconstruction_atom_indices = tuple(sorted({
                index
                for mapping in mappings.values()
                for index in mapping.target_indices
            }))

            mapping_report = {role: mapping.as_dict() for role, mapping in mappings.items()}
            for role, mapping in mappings.items():
                if policy == "position" and mapping.residue_name_mismatch_count:
                    issues.append(issue_record(
                        "warning",
                        "RESIDUE_NAME_MISMATCH",
                        f"{replica_location}/{role}",
                        f"{mapping.residue_name_mismatch_count} mapped atoms differ in residue name "
                        "because the position policy ignores residue substitutions",
                    ))

            replica_error_start = sum(
                issue["severity"] == "error" for issue in issues
            )
            segment_reports: List[Dict[str, object]] = []
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                location = f"{replica_location}/{segment_id}"
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                source_frames = source_frame_count(
                    trajectory_path, coordinate_unit, error_type=RMSDRGError
                )
                selected_indices = (
                    integer_stride_indices(
                        source_frames,
                        int(settings["frame_stride"]),
                        error_type=RMSDRGError,
                    )
                    if int(settings["frame_stride"]) > 1 else None
                )
                decoded_frames = 0
                periodic_frames = 0
                rows: List[Dict[str, object]] = []
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
                            f"{location}/frame-{raw_frame.frame_index}",
                            reconstruction_atom_indices,
                        )
                        decoded_frames += 1
                        periodic_frames += int(frame.periodic_cell_present)
                        if frame.atom_count != len(target_atoms):
                            raise RMSDRGError(
                                f"frame {frame.frame_index} has {frame.atom_count} atoms; "
                                f"topology has {len(target_atoms)}"
                            )
                        if not all(
                            math.isfinite(value)
                            for coordinate in frame.coordinates_angstrom
                            for value in coordinate
                        ):
                            raise RMSDRGError(
                                f"frame {frame.frame_index} contains non-finite coordinates"
                            )
                        if not frame_selected(
                            frame.frame_index,
                            selected_indices,
                            int(settings["frame_stride"]),
                        ):
                            continue
                        alignment = mappings["alignment"]
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
                        rmsd_mapping = mappings["rmsd"]
                        mobile_rmsd = _coordinates_at(
                            frame.coordinates_angstrom, rmsd_mapping.target_indices
                        )
                        reference_rmsd = _coordinates_at(
                            reference_frame.coordinates_angstrom,
                            rmsd_mapping.reference_indices,
                        )
                        fitted_rmsd_coordinates = apply_transform(mobile_rmsd, transform)
                        rg_mapping = mappings["rg"]
                        rg_coordinates = _coordinates_at(
                            frame.coordinates_angstrom, rg_mapping.target_indices
                        )
                        rows.append({
                            "frame_index": frame.frame_index,
                            "time": frame_time(timing, frame.frame_index),
                            "time_unit": time_unit,
                            "alignment_rmsd_angstrom": transform.fitted_rmsd_angstrom,
                            "rmsd_angstrom": rmsd(
                                fitted_rmsd_coordinates, reference_rmsd
                            ),
                            "radius_of_gyration_angstrom": mass_weighted_radius_of_gyration(
                                rg_coordinates, masses
                            ),
                        })
                    if decoded_frames == 0:
                        raise RMSDRGError("trajectory contains no readable coordinate frames")
                except (
                    CoordinateReadError,
                    GeometryError,
                    PeriodicReconstructionError,
                    RMSDRGError,
                    OSError,
                ) as exc:
                    issues.append(issue_record(
                        "error", "TRAJECTORY_ANALYSIS_FAILED", location, str(exc)
                    ))

                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append(issue_record(
                        "warning",
                        "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        location,
                        f"{periodic_frames} frames declare a periodic cell; this module does not unwrap molecules",
                    ))
                inventory_record = inventory_by_path.get(str(trajectory_path), {})
                segment_reports.append({
                    "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": inventory_record.get("sha256"),
                    "observed_frame_count": source_frames,
                    "decoded_frame_count": decoded_frames,
                    "evaluated_frame_count": len(rows),
                    "periodic_cell_frame_count": periodic_frames,
                    "timing": timing,
                    "evaluated_time_range": (
                        {
                            "start": rows[0]["time"],
                            "end": rows[-1]["time"],
                            "unit": time_unit,
                        }
                        if rows
                        else None
                    ),
                    "timeseries": rows,
                    "summary": {
                        "alignment_rmsd_angstrom": _summary(rows, "alignment_rmsd_angstrom"),
                        "rmsd_angstrom": _summary(rows, "rmsd_angstrom"),
                        "radius_of_gyration_angstrom": _summary(
                            rows, "radius_of_gyration_angstrom"
                        ),
                    },
                })
            topology_inventory_record = inventory_by_path.get(str(topology_path), {})
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
                "topology_sha256": topology_inventory_record.get("sha256"),
                "topology_atom_count": len(target_atoms),
                "periodic_reconstruction": processor.report(),
                "mappings": mapping_report,
                "segments": segment_reports,
            })
        system_reports.append({"system_id": system_id, "replicas": replica_reports})

    if int(settings["frame_stride"]) > 1:
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            f"RMSD and radius of gyration were evaluated every {settings['frame_stride']} frames",
        ))
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "module_id": "replica_rmsd_rg",
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
            "Every timeseries row reports manifest-declared physical time; embedded trajectory time metadata is not silently substituted for that contract.",
            "Alignment is unweighted; radius of gyration uses element-derived standard atomic masses on the mapped target basis.",
            "PDB and GRO topologies are supported; trajectories are limited to the coordinate-reader formats documented by this suite.",
            "Periodic production analysis requires make_whole or unwrap_continuous with explicit connectivity; allow_wrapped_diagnostic remains diagnostic only.",
            "The analysis performs finite-coordinate and atom-count checks but does not replace structural-integrity QC.",
            "A technically complete RMSD/Rg report does not establish equilibration, convergence, adequate sampling, functional importance, or scientific validity.",
            "No real-project regression fixture has yet been approved; status remains experimental.",
        ],
    }


def _reduce_rmsd_rg_replica_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in (
            "module_id", "reference", "settings", "common_atom_policy",
            "periodic_coordinate_policy", "time_unit",
        ):
            if report.get(key) != first.get(key):
                raise RMSDRGError(f"replica RMSD/Rg reports disagree on {key}")
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for report in reports:
        for system in report.get("systems", []):
            if not isinstance(system, dict):
                continue
            grouped.setdefault(str(system["system_id"]), []).extend(
                row for row in system.get("replicas", []) if isinstance(row, dict)
            )
    first["systems"] = [
        {"system_id": system_id, "replicas": grouped[system_id]}
        for system_id in grouped
    ]
    issues = unique_issues(reports)
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    first["technical_status"] = "failed" if first["error_count"] else "complete"
    restore_source_provenance(first, source_context)
    return first


def replica_rmsd_rg_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run each replica independently and retain its complete ordered series."""

    return execute_replica_final_module(
        project_path,
        runner_id="rmsd_rg",
        hash_content=hash_content,
        reducer=_reduce_rmsd_rg_replica_reports,
    )


def replica_rmsd_rg_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Convert configuration and mapping failures into a machine-readable report."""

    try:
        return replica_rmsd_rg_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        RMSDRGError,
        AtomMappingError,
        CoordinateReadError,
        GeometryError,
        PeriodicReconstructionError,
        TrajectoryContractError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "replica_rmsd_rg",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "RMSD_RG_INVALID", "message": message}
                for message in messages
            ],
        }
