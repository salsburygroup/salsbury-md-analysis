"""Immutable state-resolved trajectories and observed representative structures."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .alternative_clustering import (
    AlternativeClusteringError,
    alternative_clustering_project,
)
from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .chemical_identity import WATER_RESIDUES
from .clustering import (
    ClusteringAnalysisError,
    clustering_hdbscan_project,
    clustering_imwkmeans_project,
    clustering_kmeans_project,
)
from .context import compile_project_context_file
from .coordinates import Coordinate, CoordinateReadError, iter_coordinate_frames
from .geometry import GeometryError, apply_transform, best_fit_transform
from .frame_sampling import integer_stride_indices
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
    sha256_file,
)
from .pca import PCAAnalysisError
from .pca_fes import PCAFESAnalysisError, pca_fes_basins_project
from .oligomer_symmetry import (
    OligomerSymmetryError,
    align_member_coordinates,
    validate_member_plan,
)
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .representative_frames import (
    RepresentativeFrameError,
    _basin_candidates,
    select_state_representatives,
)
from .selections import build_common_correspondences
from .selections import select_atoms
from .validation import positive_integer


class StateCoordinateExportError(ValueError):
    """Raised when coordinate materialization would be ambiguous or destructive."""


_SOURCES = {
    "clustering_kmeans",
    "clustering_hdbscan",
    "clustering_imwkmeans",
    "alternative_clustering",
    "pca_fes_basins",
}
_EXPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_IDENTITY = ("system_id", "replica_id", "segment_id", "source_frame_index")
def _member_payload_map(
    atoms: Sequence[AtomRecord],
    plan: Mapping[str, object],
    member_id: str,
    *,
    policy: str,
) -> Tuple[Tuple[Tuple[object, ...], ...], Tuple[int, ...]]:
    """Canonicalize one complete chain-associated member molecular payload.

    PCA member maps intentionally contain only heavy feature atoms.  Exports
    instead retain every non-water atom (including hydrogens and chain-labelled
    ligands, cofactors, and ions) belonging to the member's protein and nucleic
    acid chains.  Chain labels are replaced by canonical component roles so
    equivalent members can be pooled into one valid trajectory topology.
    """

    raw_members = plan.get("members")
    if not isinstance(raw_members, list):
        raise StateCoordinateExportError("oligomer member plan has no members")
    matches = [
        row for row in raw_members
        if isinstance(row, dict) and str(row.get("member_id")) == member_id
    ]
    if len(matches) != 1:
        raise StateCoordinateExportError(
            f"oligomer member {member_id!r} has no unique payload definition"
        )
    member = matches[0]
    protein_chain = str(member.get("protein_chain_id", ""))
    raw_nucleic = member.get("nucleic_chain_ids", [])
    if not protein_chain or not isinstance(raw_nucleic, list):
        raise StateCoordinateExportError("oligomer member chain definition is invalid")
    role_by_chain = {protein_chain: "protein"}
    for position, chain_id in enumerate(raw_nucleic, start=1):
        chain = str(chain_id)
        if not chain or chain in role_by_chain:
            raise StateCoordinateExportError(
                "oligomer member contains empty or duplicate molecular chain roles"
            )
        role_by_chain[chain] = f"nucleic_{position}"

    residue_ordinals: Dict[Tuple[str, int, str, str], int] = {}
    next_ordinal: Dict[str, int] = {}
    rows: List[Tuple[Tuple[object, ...], int]] = []
    for atom in atoms:
        role = role_by_chain.get(atom.chain_id)
        if role is None or atom.residue_name.upper() in WATER_RESIDUES:
            continue
        residue_key = (
            atom.chain_id, atom.residue_number, atom.insertion_code,
            atom.residue_name,
        )
        if residue_key not in residue_ordinals:
            residue_ordinals[residue_key] = next_ordinal.get(role, 0)
            next_ordinal[role] = residue_ordinals[residue_key] + 1
        identity = (
            role,
            residue_ordinals[residue_key],
            atom.residue_name if policy == "strict" else None,
            atom.atom_name,
            atom.altloc,
            atom.element,
        )
        rows.append((identity, atom.atom_index))
    if not rows:
        raise StateCoordinateExportError(
            f"oligomer member {member_id!r} molecular payload is empty"
        )
    identities = [identity for identity, _ in rows]
    if len(set(identities)) != len(identities):
        raise StateCoordinateExportError(
            f"oligomer member {member_id!r} molecular payload identities are ambiguous"
        )
    rows.sort(key=lambda row: row[0])
    return tuple(identity for identity, _ in rows), tuple(index for _, index in rows)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("state_coordinate_exports") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise StateCoordinateExportError(
            "definitions.state_coordinate_exports must be an object"
        )
    required = {
        "source", "export_id", "trajectory_format", "representatives_per_state",
        "frame_stride_within_state", "maximum_states", "maximum_frames_per_state",
        "maximum_total_frames", "existing_output_policy",
    }
    optional = {
        "fes_smoothing_sigma_bins", "alternative_algorithm", "coordinate_selection",
        "write_trajectories",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | optional))
    if missing or unknown:
        raise StateCoordinateExportError(
            "state-coordinate export settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    source = raw["source"]
    if source not in _SOURCES:
        raise StateCoordinateExportError("state-coordinate export source is unsupported")
    export_id = raw["export_id"]
    if not isinstance(export_id, str) or not _EXPORT_ID.fullmatch(export_id):
        raise StateCoordinateExportError(
            "export_id must be 1-80 safe letters, digits, dots, underscores, or hyphens"
        )
    if raw["trajectory_format"] not in {"pdb", "xyz"}:
        raise StateCoordinateExportError("trajectory_format must be pdb or xyz")
    if raw["existing_output_policy"] != "fail":
        raise StateCoordinateExportError(
            "existing_output_policy must be fail; overwrite and merge are prohibited"
        )
    result = {
        "source": source,
        "export_id": export_id,
        "trajectory_format": raw["trajectory_format"],
        "existing_output_policy": "fail",
        "write_trajectories": raw.get("write_trajectories", True),
    }
    if not isinstance(result["write_trajectories"], bool):
        raise StateCoordinateExportError("write_trajectories must be boolean")
    if "coordinate_selection" in raw:
        selection = raw["coordinate_selection"]
        if not isinstance(selection, str) or not selection.strip():
            raise StateCoordinateExportError(
                "coordinate_selection must be a nonempty project selection name"
            )
        project_selections = project.get("selections")
        if (
            not isinstance(project_selections, dict)
            or selection.strip() not in project_selections
        ):
            raise StateCoordinateExportError(
                "coordinate_selection does not name a declared project selection"
            )
        result["coordinate_selection"] = selection.strip()
    for field in (
        "representatives_per_state", "frame_stride_within_state", "maximum_states",
        "maximum_frames_per_state", "maximum_total_frames",
    ):
        result[field] = positive_integer(
            raw[field], field, error_type=StateCoordinateExportError
        )
    if "fes_smoothing_sigma_bins" in raw:
        value = raw["fes_smoothing_sigma_bins"]
        if (
            source != "pca_fes_basins"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise StateCoordinateExportError(
                "fes_smoothing_sigma_bins is valid only for PCA-FES and must be finite and nonnegative"
            )
        result["fes_smoothing_sigma_bins"] = float(value)
    if source == "alternative_clustering":
        algorithm = raw.get("alternative_algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise StateCoordinateExportError(
                "alternative_algorithm is required for alternative_clustering exports"
            )
        result["alternative_algorithm"] = algorithm.strip()
    elif "alternative_algorithm" in raw:
        raise StateCoordinateExportError(
            "alternative_algorithm is valid only for alternative_clustering exports"
        )
    return result


def _source_candidates(
    project_path: Path, settings: Mapping[str, object], hash_content: bool
) -> Tuple[Dict[str, object], List[Dict[str, object]], str, str]:
    source = str(settings["source"])
    if source == "clustering_kmeans":
        report = clustering_kmeans_project(project_path, hash_content=hash_content)
        rows = report.get("assignments")
        state_field = "cluster_id"
        distance_field = "squared_distance_in_clustering_space"
    elif source == "clustering_imwkmeans":
        report = clustering_imwkmeans_project(project_path, hash_content=hash_content)
        rows = report.get("assignments")
        state_field = "cluster_id"
        distance_field = "weighted_minkowski_distance_power"
    elif source == "clustering_hdbscan":
        report = clustering_hdbscan_project(project_path, hash_content=hash_content)
        rows = report.get("assignments")
        state_field = "cluster_id"
        distance_field = "squared_distance_in_clustering_space"
    elif source == "pca_fes_basins":
        report = pca_fes_basins_project(project_path, hash_content=hash_content)
        rows = _basin_candidates(report, settings.get("fes_smoothing_sigma_bins"))
        state_field = "basin_id"
        distance_field = "distance_to_basin_root_squared"
    else:
        report = alternative_clustering_project(project_path, hash_content=hash_content)
        results = report.get("algorithm_results")
        matches = [
            row for row in results
            if isinstance(row, dict)
            and row.get("requested_algorithm") == settings["alternative_algorithm"]
        ] if isinstance(results, list) else []
        if len(matches) != 1:
            raise StateCoordinateExportError(
                "the requested alternative algorithm has no unique selected partition"
            )
        rows = matches[0].get("frame_assignments")
        state_field = "cluster_id"
        distance_field = "squared_distance_in_clustering_space"
    if report.get("technical_status") != "complete":
        raise StateCoordinateExportError(f"source module {source} did not complete")
    if not isinstance(rows, list) or not rows:
        raise StateCoordinateExportError("source module contains no frame assignments")
    candidates = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StateCoordinateExportError(f"source assignment {index} is not an object")
        candidates.append(dict(row))
    return report, candidates, state_field, distance_field


def _safe_slug(value: object) -> str:
    text = str(value)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.") or "unnamed"
    return normalized[:64]


def _pdb_atom_line(atom: AtomRecord, coordinate: Coordinate) -> str:
    if not 1 <= atom.serial <= 99999:
        raise StateCoordinateExportError("PDB export requires atom serials in 1..99999")
    if not -999 <= atom.residue_number <= 9999:
        raise StateCoordinateExportError(
            "PDB export requires residue numbers representable in four columns"
        )
    x, y, z = coordinate
    if max(abs(x), abs(y), abs(z)) >= 10000.0:
        raise StateCoordinateExportError("PDB coordinate exceeds fixed-column range")
    return (
        f"ATOM  {atom.serial:5d} {atom.atom_name:^4s}{atom.altloc[:1]:1s}"
        f"{atom.residue_name:>3s} {atom.chain_id[:1]:1s}{atom.residue_number:4d}"
        f"{atom.insertion_code[:1]:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {atom.element[:2]:>2s}\n"
    )


def _write_pdb(
    path: Path,
    atoms: Sequence[AtomRecord],
    frames: Sequence[Tuple[Mapping[str, object], Sequence[Coordinate]]],
    multi_model: bool,
) -> None:
    lines: List[str] = []
    for model_index, (identity, coordinates) in enumerate(frames, start=1):
        if len(coordinates) != len(atoms):
            raise StateCoordinateExportError("topology/coordinate atom-count mismatch")
        if multi_model:
            lines.append(f"MODEL     {model_index:4d}\n")
            provenance = " ".join(
                f"{field}={identity[field]}"
                for field in (*_IDENTITY, "member_id") if field in identity
            )
            lines.append(f"REMARK 950 {provenance}\n")
        lines.extend(
            _pdb_atom_line(atom, coordinate)
            for atom, coordinate in zip(atoms, coordinates)
        )
        if multi_model:
            lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="ascii")


def _write_xyz(
    path: Path,
    atoms: Sequence[AtomRecord],
    frames: Sequence[Tuple[Mapping[str, object], Sequence[Coordinate]]],
) -> None:
    lines: List[str] = []
    for identity, coordinates in frames:
        if len(coordinates) != len(atoms):
            raise StateCoordinateExportError("topology/coordinate atom-count mismatch")
        lines.extend([
            f"{len(atoms)}\n",
            " ".join(
                f"{field}={identity[field]}"
                for field in (*_IDENTITY, "member_id") if field in identity
            ) + "\n",
        ])
        for atom, (x, y, z) in zip(atoms, coordinates):
            lines.append(f"{atom.element or 'X'} {x:.8f} {y:.8f} {z:.8f}\n")
    path.write_text("".join(lines), encoding="ascii")


def _selected_state_rows(
    candidates: Sequence[Mapping[str, object]],
    state_field: str,
    settings: Mapping[str, object],
) -> List[Dict[str, object]]:
    by_state: Dict[int, List[Dict[str, object]]] = {}
    for index, row in enumerate(candidates):
        missing = [field for field in _IDENTITY if field not in row]
        if missing:
            raise StateCoordinateExportError(
                f"source assignment {index} is missing identity fields: {','.join(missing)}"
            )
        state = row.get(state_field)
        if state is None:
            continue
        if isinstance(state, bool) or not isinstance(state, int) or state <= 0:
            raise StateCoordinateExportError("assigned state IDs must be positive integers")
        by_state.setdefault(state, []).append({**row, "state_id": state})
    if not by_state:
        raise StateCoordinateExportError("source contains no assigned states")
    if len(by_state) > int(settings["maximum_states"]):
        raise StateCoordinateExportError("state count exceeds maximum_states")
    selected: List[Dict[str, object]] = []
    stride = int(settings["frame_stride_within_state"])
    for state in sorted(by_state):
        ordered_source = sorted(
            by_state[state],
            key=lambda row: (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]), int(row["source_frame_index"]),
                str(row.get("member_id", "")),
            ),
        )
        ordered = [
            ordered_source[position]
            for position in sorted(
                integer_stride_indices(len(ordered_source), stride)
            )
        ]
        if len(ordered) > int(settings["maximum_frames_per_state"]):
            raise StateCoordinateExportError(
                f"state {state} exceeds maximum_frames_per_state after declared stride"
            )
        selected.extend(ordered)
    if len(selected) > int(settings["maximum_total_frames"]):
        raise StateCoordinateExportError("selected frames exceed maximum_total_frames")
    return selected


def _capture_coordinates(
    project: Mapping[str, object],
    project_path: Path,
    system_path: Path,
    requested: Iterable[Tuple[str, str, str, int]],
) -> Tuple[
    Dict[Tuple[str, str, str, int], Tuple[Coordinate, ...]],
    Dict[Tuple[str, str], Tuple[List[AtomRecord], Path]],
    Dict[Tuple[str, str, str], int],
    List[Dict[str, object]],
]:
    requested_set = set(requested)
    requested_by_segment: Dict[Tuple[str, str, str], set[int]] = {}
    for system_id, replica_id, segment_id, frame_index in requested_set:
        requested_by_segment.setdefault(
            (system_id, replica_id, segment_id), set()
        ).add(frame_index)
    system = load_json(system_path)
    systems = system["systems"]
    assert isinstance(systems, list)
    coordinate_unit = str(project["coordinate_unit"])
    captured: Dict[Tuple[str, str, str, int], Tuple[Coordinate, ...]] = {}
    topologies: Dict[Tuple[str, str], Tuple[List[AtomRecord], Path]] = {}
    segment_order: Dict[Tuple[str, str, str], int] = {}
    reference_path = resolve_manifest_path(
        str(project["reference_structure"]), project_path
    )
    _, reference_atoms = read_topology_atoms(reference_path)
    reference_raw = next(iter_coordinate_frames(reference_path, coordinate_unit))
    reference_coordinates = reference_raw.coordinates_angstrom
    pca_definition = project.get("definitions", {}).get("common_pca", {})
    if not isinstance(pca_definition, dict):
        raise StateCoordinateExportError("common_pca definition is unavailable")
    alignment_name = pca_definition.get("alignment_selection")
    selections = project.get("selections")
    if not isinstance(alignment_name, str) or not isinstance(selections, dict):
        raise StateCoordinateExportError(
            "state exports require the common_pca alignment selection"
        )
    alignment_definition = selections.get(alignment_name)
    if not isinstance(alignment_definition, dict):
        raise StateCoordinateExportError(
            "common_pca alignment selection does not resolve to an object"
        )
    relevant_topologies: Dict[
        Tuple[str, str], Tuple[List[AtomRecord], Path]
    ] = {}
    relevant_keys: List[Tuple[str, str]] = []
    for system_row in systems:
        assert isinstance(system_row, dict)
        system_id = str(system_row["system_id"])
        replicas = system_row["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            if not any(key[:2] == (system_id, replica_id) for key in requested_set):
                continue
            topology_path = resolve_manifest_path(
                str(replica["topology"]), system_path
            )
            _, atoms = read_topology_atoms(topology_path)
            key = (system_id, replica_id)
            relevant_keys.append(key)
            relevant_topologies[key] = (atoms, topology_path)
    if not relevant_keys:
        raise StateCoordinateExportError(
            "requested state frames do not resolve to any declared replica"
        )
    minimum_coverage = pca_definition.get("minimum_reference_coverage", 0.95)
    correspondences = build_common_correspondences(
        reference_atoms,
        tuple(relevant_topologies[key][0] for key in relevant_keys),
        alignment_definition,
        alignment_name,
        str(project["common_atom_policy"]),
        float(minimum_coverage),
    )
    correspondence_by_key = dict(zip(relevant_keys, correspondences))
    reference_alignment_coordinates = tuple(
        reference_coordinates[index]
        for index in correspondences[0].reference_indices
    )
    alignment_mapping = [
        {
            "system_id": system_id,
            "replica_id": replica_id,
            **correspondence.as_dict(),
        }
        for (system_id, replica_id), correspondence in zip(
            relevant_keys, correspondences
        )
    ]
    for system_row in systems:
        assert isinstance(system_row, dict)
        system_id = str(system_row["system_id"])
        replicas = system_row["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            relevant = any(
                key[:2] == (system_id, replica_id) for key in requested_set
            )
            segments = replica["segments"]
            assert isinstance(segments, list)
            for index, segment in enumerate(segments):
                assert isinstance(segment, dict)
                segment_order[(system_id, replica_id, str(segment["segment_id"]))] = index
            if not relevant:
                continue
            atoms, topology_path = relevant_topologies[(system_id, replica_id)]
            selection_name = project.get("definitions", {}).get(
                "state_coordinate_exports", {}
            ).get("coordinate_selection")
            if selection_name is None:
                output_atoms = atoms
                output_indices = tuple(range(len(atoms)))
            else:
                selections = project.get("selections")
                assert isinstance(selections, dict)
                definition = selections.get(str(selection_name))
                if not isinstance(definition, dict):
                    raise StateCoordinateExportError(
                        "coordinate_selection does not resolve to an object"
                    )
                output_atoms = select_atoms(
                    atoms, definition, str(selection_name)
                )
                output_indices = tuple(atom.atom_index for atom in output_atoms)
            alignment_indices = correspondence_by_key[
                (system_id, replica_id)
            ].target_indices
            reconstruction_indices = tuple(sorted(set(
                output_indices + alignment_indices
            )))
            topologies[(system_id, replica_id)] = (output_atoms, topology_path)
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                wanted = requested_by_segment.get(
                    (system_id, replica_id, segment_id), set()
                )
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                processor.begin_segment(
                    bool(segment.get("continuous_with_previous", False))
                )
                # make_whole is frame-local. DCD readers can seek past all
                # unrequested coordinate records, so a 200-frame export does
                # not decode or reconstruct a 30,000-frame solvated trajectory.
                # Continuous unwrapping is the sole policy that requires every
                # intervening frame to maintain component image continuity.
                if not wanted and processor.policy != "unwrap_continuous":
                    continue
                reader_selection = (
                    None if processor.policy == "unwrap_continuous" else wanted
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit, reader_selection
                ):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_indices,
                    )
                    key = (system_id, replica_id, segment_id, frame.frame_index)
                    if key in requested_set:
                        mobile_alignment = tuple(
                            frame.coordinates_angstrom[index]
                            for index in alignment_indices
                        )
                        transform = best_fit_transform(
                            mobile_alignment, reference_alignment_coordinates
                        )
                        mobile_output = tuple(
                            frame.coordinates_angstrom[index]
                            for index in output_indices
                        )
                        captured[key] = apply_transform(mobile_output, transform)
    missing = sorted(requested_set.difference(captured))
    if missing:
        raise StateCoordinateExportError(
            f"{len(missing)} requested source frames were not found; first={missing[0]}"
        )
    return captured, topologies, segment_order, alignment_mapping


def _capture_member_coordinates(
    project: Mapping[str, object],
    project_path: Path,
    system_path: Path,
    requested: Iterable[Tuple[str, str, str, int, str]],
    plan: Mapping[str, object],
) -> Tuple[
    Dict[Tuple[str, str, str, int, str], Tuple[Coordinate, ...]],
    Dict[Tuple[str, str], Tuple[List[AtomRecord], Path]],
    Dict[Tuple[str, str, str], int],
    List[Dict[str, object]],
]:
    """Capture canonical, independently aligned equivalent-member coordinates."""

    requested_set = set(requested)
    requested_by_frame: Dict[Tuple[str, str, str, int], set[str]] = {}
    for system_id, replica_id, segment_id, frame_index, member_id in requested_set:
        requested_by_frame.setdefault(
            (system_id, replica_id, segment_id, frame_index), set()
        ).add(member_id)

    coordinate_unit = str(project["coordinate_unit"])
    reference_path = resolve_manifest_path(str(project["reference_structure"]), project_path)
    _, reference_atoms = read_topology_atoms(reference_path)
    reference_raw = next(iter_coordinate_frames(reference_path, coordinate_unit))
    reference_processor = PeriodicFrameProcessor.from_reference(
        project, project_path, len(reference_atoms)
    )
    reference_frame = reference_processor.process(reference_raw, str(reference_path))
    reference_resolved = validate_member_plan(
        reference_atoms, plan, policy=str(project["common_atom_policy"])
    )
    reference_members = reference_resolved["members"]
    if not isinstance(reference_members, list) or not reference_members:
        raise StateCoordinateExportError("canonical oligomer reference member is unavailable")
    reference_member = reference_members[0]
    if not isinstance(reference_member, dict):
        raise StateCoordinateExportError("canonical oligomer reference member is invalid")
    analysis_indices = reference_member["analysis_atom_indices"]
    if not isinstance(analysis_indices, list):
        raise StateCoordinateExportError("canonical oligomer atom indices are invalid")
    selection_name = project.get("definitions", {}).get(
        "state_coordinate_exports", {}
    ).get("coordinate_selection")
    if selection_name is None:
        output_atoms = [reference_atoms[int(index)] for index in analysis_indices]
        reference_payload_identity = None
    elif selection_name == "molecular_payload":
        reference_payload_identity, reference_output_indices = _member_payload_map(
            reference_atoms,
            plan,
            str(reference_member["member_id"]),
            policy=str(project["common_atom_policy"]),
        )
        output_atoms = [reference_atoms[index] for index in reference_output_indices]
    else:
        selections = project.get("selections")
        if not isinstance(selections, dict):
            raise StateCoordinateExportError("project selections are unavailable")
        definition = selections.get(str(selection_name))
        if not isinstance(definition, dict):
            raise StateCoordinateExportError(
                "coordinate_selection does not resolve to an object"
            )
        output_atoms = list(select_atoms(
            reference_atoms, definition, str(selection_name)
        ))
        reference_payload_identity = None
    output_identity = tuple(
        atom.match_key(str(project["common_atom_policy"])) for atom in output_atoms
    )

    system = load_json(system_path)
    systems = system["systems"]
    assert isinstance(systems, list)
    captured: Dict[Tuple[str, str, str, int, str], Tuple[Coordinate, ...]] = {}
    topologies: Dict[Tuple[str, str], Tuple[List[AtomRecord], Path]] = {}
    segment_order: Dict[Tuple[str, str, str], int] = {}
    alignment_mapping: List[Dict[str, object]] = []
    for system_row in systems:
        assert isinstance(system_row, dict)
        system_id = str(system_row["system_id"])
        replicas = system_row["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            segments = replica["segments"]
            assert isinstance(segments, list)
            for index, segment in enumerate(segments):
                assert isinstance(segment, dict)
                segment_order[(system_id, replica_id, str(segment["segment_id"]))] = index
            if not any(key[:2] == (system_id, replica_id) for key in requested_set):
                continue
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            resolved = validate_member_plan(
                atoms, plan, policy=str(project["common_atom_policy"])
            )
            members = resolved["members"]
            assert isinstance(members, list)
            by_member = {
                str(member["member_id"]): member
                for member in members if isinstance(member, dict)
            }
            reference_alignment = reference_member.get("alignment_atom_indices")
            if not isinstance(reference_alignment, list):
                raise StateCoordinateExportError(
                    "canonical oligomer alignment indices are unavailable"
                )
            requested_member_ids = sorted({
                member_id
                for requested_system, requested_replica, _, _, member_id in requested_set
                if (requested_system, requested_replica) == (system_id, replica_id)
            })
            for member_id in requested_member_ids:
                member = by_member.get(member_id)
                if member is None:
                    raise StateCoordinateExportError(
                        f"requested oligomer member {member_id!r} is absent"
                    )
                member_alignment = member.get("alignment_atom_indices")
                if not isinstance(member_alignment, list) or len(
                    member_alignment
                ) != len(reference_alignment):
                    raise StateCoordinateExportError(
                        f"member {member_id!r} alignment is not exactly compatible "
                        "with the canonical oligomer member"
                    )
                mapping_payload = {
                    "reference_indices": [int(index) for index in reference_alignment],
                    "target_indices": [int(index) for index in member_alignment],
                    "canonical_reference_member_id": str(reference_member["member_id"]),
                    "member_id": member_id,
                }
                alignment_mapping.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "selection_id": "oligomer_member_alignment",
                    "policy": str(project["common_atom_policy"]),
                    "mapped_atom_count": len(reference_alignment),
                    "reference_selected_atom_count": len(reference_alignment),
                    "target_selected_atom_count": len(member_alignment),
                    "reference_coverage": 1.0,
                    **mapping_payload,
                    "residue_name_mismatch_count": 0,
                    "mapping_signature_sha256": hashlib.sha256(
                        json.dumps(
                            mapping_payload, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                })
            if selection_name is None:
                output_indices = None
                output_indices_by_member = None
            elif selection_name == "molecular_payload":
                if reference_payload_identity is None:
                    raise StateCoordinateExportError(
                        "canonical member molecular payload identity is unavailable"
                    )
                output_indices_by_member = {}
                for member_id in sorted(by_member):
                    payload_identity, payload_indices = _member_payload_map(
                        atoms,
                        plan,
                        member_id,
                        policy=str(project["common_atom_policy"]),
                    )
                    if payload_identity != reference_payload_identity:
                        raise StateCoordinateExportError(
                            f"member {member_id!r} molecular payload does not exactly "
                            "match the canonical member topology"
                        )
                    output_indices_by_member[member_id] = payload_indices
                output_indices = None
            else:
                selections = project.get("selections")
                assert isinstance(selections, dict)
                definition = selections[str(selection_name)]
                assert isinstance(definition, dict)
                selected_output = select_atoms(
                    atoms, definition, str(selection_name)
                )
                if tuple(
                    atom.match_key(str(project["common_atom_policy"]))
                    for atom in selected_output
                ) != output_identity:
                    raise StateCoordinateExportError(
                        "replica molecular payload does not exactly match the reference order"
                    )
                output_indices = tuple(atom.atom_index for atom in selected_output)
                output_indices_by_member = None
            topologies[(system_id, replica_id)] = (output_atoms, reference_path)
            reconstruction = tuple(sorted({
                int(atom_index)
                for member in members if isinstance(member, dict)
                for field in ("analysis_atom_indices", "alignment_atom_indices")
                for atom_index in member[field]
            } | set(output_indices or ()) | {
                int(atom_index)
                for indices in (output_indices_by_member or {}).values()
                for atom_index in indices
            }))
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                wanted = {
                    frame_index
                    for requested_system, requested_replica, requested_segment,
                    frame_index, _ in requested_set
                    if (
                        requested_system, requested_replica, requested_segment
                    ) == (system_id, replica_id, segment_id)
                }
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                if not wanted and processor.policy != "unwrap_continuous":
                    continue
                reader_selection = (
                    None if processor.policy == "unwrap_continuous" else wanted
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit, reader_selection
                ):
                    physical_key = (
                        system_id, replica_id, segment_id, raw_frame.frame_index
                    )
                    member_ids = requested_by_frame.get(physical_key)
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction,
                    )
                    if not member_ids:
                        continue
                    for member_id in sorted(member_ids):
                        member = by_member.get(member_id)
                        if member is None:
                            raise StateCoordinateExportError(
                                f"requested oligomer member {member_id!r} is absent"
                            )
                        if output_indices is None and output_indices_by_member is None:
                            captured[(*physical_key, member_id)] = align_member_coordinates(
                                frame.coordinates_angstrom,
                                member,
                                reference_frame.coordinates_angstrom,
                                reference_member,
                            )
                        else:
                            member_alignment = member.get("alignment_atom_indices")
                            reference_alignment = reference_member.get(
                                "alignment_atom_indices"
                            )
                            if not isinstance(member_alignment, list) or not isinstance(
                                reference_alignment, list
                            ):
                                raise StateCoordinateExportError(
                                    "member alignment indices are unavailable"
                                )
                            transform = best_fit_transform(
                                tuple(
                                    frame.coordinates_angstrom[int(index)]
                                    for index in member_alignment
                                ),
                                tuple(
                                    reference_frame.coordinates_angstrom[int(index)]
                                    for index in reference_alignment
                                ),
                            )
                            member_output_indices = (
                                output_indices_by_member[member_id]
                                if output_indices_by_member is not None
                                else output_indices
                            )
                            if member_output_indices is None:
                                raise StateCoordinateExportError(
                                    "member molecular payload indices are unavailable"
                                )
                            captured[(*physical_key, member_id)] = apply_transform(
                                tuple(
                                    frame.coordinates_angstrom[index]
                                    for index in member_output_indices
                                ),
                                transform,
                            )
    missing = sorted(requested_set.difference(captured))
    if missing:
        raise StateCoordinateExportError(
            f"{len(missing)} requested member observations were not found; first={missing[0]}"
        )
    return captured, topologies, segment_order, alignment_mapping


def state_coordinate_exports_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Materialize declared state trajectories without changing any source file."""

    source_path = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source_path)
    settings = _settings(project)
    context = compile_project_context_file(source_path, hash_content=hash_content)
    source_report, candidates, state_field, distance_field = _source_candidates(
        source_path, settings, hash_content
    )
    selected = (
        _selected_state_rows(candidates, state_field, settings)
        if settings["write_trajectories"] else []
    )
    representatives = select_state_representatives(
        candidates,
        state_field=state_field,
        distance_field=distance_field,
        representatives_per_state=int(settings["representatives_per_state"]),
        maximum_states=int(settings["maximum_states"]),
        maximum_candidates=max(len(candidates), int(settings["maximum_total_frames"])),
    )
    member_mode = any("member_id" in row for row in [*selected, *representatives])
    if member_mode and any("member_id" not in row for row in [*selected, *representatives]):
        raise StateCoordinateExportError(
            "state assignments mix symmetry-expanded and physical-frame identities"
        )
    system_path = Path(str(context["system_manifest_path"]))
    if member_mode:
        common_pca = project.get("definitions", {}).get("common_pca", {})
        plan = common_pca.get("symmetry_expansion") if isinstance(common_pca, dict) else None
        if not isinstance(plan, dict) or plan.get("applicable") is not True:
            raise StateCoordinateExportError(
                "member-resolved assignments require definitions.common_pca.symmetry_expansion"
            )
        requested_members = {
            (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]), int(row["source_frame_index"]),
                str(row["member_id"]),
            )
            for row in [*selected, *representatives]
        }
        captured, topologies, segment_order, alignment_mapping = _capture_member_coordinates(
            project, source_path, system_path, requested_members, plan
        )
    else:
        requested = {
            (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]), int(row["source_frame_index"]),
            )
            for row in [*selected, *representatives]
        }
        captured, topologies, segment_order, alignment_mapping = _capture_coordinates(
            project, source_path, system_path, requested
        )

    output_root = resolve_manifest_path(str(project["analysis_output_root"]), source_path)
    parent = output_root / "08_clustering" / "state_coordinate_exports"
    final_directory = parent / str(settings["export_id"])
    if final_directory.exists():
        raise StateCoordinateExportError(
            f"export directory already exists and overwrite is prohibited: {final_directory}"
        )
    parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f".{settings['export_id']}.", dir=str(parent)
    ))
    output_records: List[Dict[str, object]] = []
    try:
        # State ensembles are pooled across simulation replicas and equivalent
        # oligomer members. Replica/member identities remain per-frame
        # provenance; they are not separate biological systems or separate
        # output trajectories.
        groups: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
        for row in selected:
            key = (int(row["state_id"]), str(row["system_id"]))
            groups.setdefault(key, []).append(row)
        for row in representatives:
            key = (int(row["state_id"]), str(row["system_id"]))
            groups.setdefault(key, [])
        for (state_id, system_id), rows in sorted(groups.items()):
            topology_candidates = sorted(
                (key, value) for key, value in topologies.items()
                if key[0] == system_id
            )
            if not topology_candidates:
                raise StateCoordinateExportError(
                    f"no captured topology is available for system {system_id!r}"
                )
            (_, canonical_replica), (atoms, topology_path) = topology_candidates[0]
            canonical_identity = tuple(
                (
                    atom.atom_name, atom.altloc, atom.residue_name, atom.chain_id,
                    atom.residue_number, atom.insertion_code, atom.element,
                )
                for atom in atoms
            )
            for (_, replica_id), (candidate_atoms, _) in topology_candidates[1:]:
                candidate_identity = tuple(
                    (
                        atom.atom_name, atom.altloc, atom.residue_name, atom.chain_id,
                        atom.residue_number, atom.insertion_code, atom.element,
                    )
                    for atom in candidate_atoms
                )
                if candidate_identity != canonical_identity:
                    raise StateCoordinateExportError(
                        f"system {system_id!r} replica topologies differ between "
                        f"{canonical_replica!r} and {replica_id!r}; pooled export "
                        "requires identical selected atom identity and order"
                    )
            ordered = sorted(rows, key=lambda row: (
                str(row["replica_id"]),
                segment_order[(
                    system_id, str(row["replica_id"]), str(row["segment_id"])
                )],
                int(row["source_frame_index"]),
                str(row.get("member_id", "")),
            ))
            frames = [
                (
                    row,
                    captured[(
                        system_id, str(row["replica_id"]), str(row["segment_id"]),
                        int(row["source_frame_index"]),
                        *((str(row["member_id"]),) if member_mode else ()),
                    )],
                )
                for row in ordered
            ]
            relative = (
                Path(f"state-{state_id:04d}")
                / f"system-{_safe_slug(system_id)}"
            )
            directory = temporary_directory / relative
            directory.mkdir(parents=True, exist_ok=False)
            trajectory_path = None
            if frames:
                trajectory_path = directory / f"trajectory.{settings['trajectory_format']}"
                if settings["trajectory_format"] == "pdb":
                    _write_pdb(trajectory_path, atoms, frames, multi_model=True)
                else:
                    _write_xyz(trajectory_path, atoms, frames)
            representative_rows = [
                row for row in representatives
                if int(row["state_id"]) == state_id
                and str(row["system_id"]) == system_id
            ]
            provenance_rows = [*rows, *representative_rows]
            representative_files = []
            for representative in representative_rows:
                rank = int(representative["representative_rank"])
                key = (
                    system_id, str(representative["replica_id"]),
                    str(representative["segment_id"]),
                    int(representative["source_frame_index"]),
                    *((str(representative["member_id"]),) if member_mode else ()),
                )
                representative_path = directory / f"representative-{rank:02d}.pdb"
                _write_pdb(
                    representative_path,
                    atoms,
                    [(representative, captured[key])],
                    multi_model=False,
                )
                representative_files.append({
                    "representative_rank": rank,
                    "path": str(representative_path.relative_to(temporary_directory)),
                    "sha256": sha256_file(representative_path),
                    **{field: representative[field] for field in _IDENTITY},
                    **(
                        {"member_id": representative["member_id"]}
                        if member_mode else {}
                    ),
                })
            output_records.append({
                "state_id": state_id,
                "system_id": system_id,
                "pooled_replica_ids": sorted({
                    str(row["replica_id"]) for row in provenance_rows
                }),
                **({
                    "pooled_member_ids": sorted({
                        str(row["member_id"]) for row in provenance_rows
                    })
                } if member_mode else {}),
                "topology_path": str(topology_path),
                "topology_sha256": sha256_file(topology_path),
                "trajectory_path": (
                    str(trajectory_path.relative_to(temporary_directory))
                    if trajectory_path is not None else None
                ),
                "trajectory_sha256": (
                    sha256_file(trajectory_path) if trajectory_path is not None else None
                ),
                "trajectory_frame_count": len(frames),
                "trajectory_frame_provenance": [
                    {
                        **{field: row[field] for field in _IDENTITY},
                        **({"member_id": row["member_id"]} if member_mode else {}),
                    }
                    for row in ordered
                ],
                "representatives": representative_files,
            })
        manifest = {
            "module_id": "state_coordinate_exports",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(source_path),
            "project_manifest_sha256": context["project_manifest_sha256"],
            "system_manifest_path": str(system_path),
            "system_manifest_sha256": context["system_manifest_sha256"],
            "input_content_signature_sha256": context["input_content_signature_sha256"],
            "source_module_id": settings["source"],
            "source_contract_signature_sha256": source_report.get("contract_signature_sha256"),
            "settings": settings,
            "coordinate_representation": (
                "declared molecular payload after independent per-member protein/nucleic-acid alignment"
                if member_mode else
                "declared molecular payload after the project's periodic reconstruction and PCA-view alignment transform"
            ),
            "coordinate_selection": settings.get("coordinate_selection", "all"),
            "alignment_mapping": {
                "mode": (
                    "equivalent_oligomer_member_to_canonical_member"
                    if member_mode else "common_atom_correspondence"
                ),
                "selection_id": (
                    "oligomer_member_alignment"
                    if member_mode else project["definitions"]["common_pca"][
                        "alignment_selection"
                    ]
                ),
                "policy": project["common_atom_policy"],
                "minimum_reference_coverage": project["definitions"][
                    "common_pca"
                ].get("minimum_reference_coverage", 0.95),
                "replicas": alignment_mapping,
            },
            "state_count": len({row["state_id"] for row in representatives}),
            "exported_frame_count": len(selected),
            "representative_count": len(representatives),
            "observation_accounting": {
                "source_physical_frame_count": len({
                    (
                        str(row["system_id"]), str(row["replica_id"]),
                        str(row["segment_id"]), int(row["source_frame_index"]),
                    )
                    for row in selected
                }),
                "exported_observation_count": len(selected),
                "representative_physical_frame_count": len({
                    (
                        str(row["system_id"]), str(row["replica_id"]),
                        str(row["segment_id"]), int(row["source_frame_index"]),
                    )
                    for row in representatives
                }),
                "representative_observation_count": len(representatives),
                "symmetry_expanded": member_mode,
                "member_observations_are_independent_replicas": False
                if member_mode else None,
            },
            "outputs": output_records,
            "limitations": [
                (
                    "State trajectories are disabled by configuration; observed representative structures remain mandatory."
                    if not settings["write_trajectories"] else
                    "State trajectories are descriptive subsets and do not establish metastability, convergence, kinetics, or mechanism."
                ),
                (
                    "Coordinates use the declared periodic reconstruction and independent canonical-member alignment."
                    if member_mode else
                    "Coordinates use the declared periodic reconstruction and the PCA view's declared protein/nucleic-acid alignment."
                ),
                *(
                    [
                        "Equivalent-member coordinates are independently aligned to the canonical reference member.",
                        "Equivalent-member observations are pooled within each system/state while replica and member provenance remains explicit; members are paired representations, not independent replicas.",
                    ]
                    if member_mode else []
                ),
                "Different biological systems remain separate; replicas are pooled only after exact selected-topology identity validation.",
            ],
        }
        manifest_path = temporary_directory / "export-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_directory, final_directory)
    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise
    return {
        **manifest,
        "export_directory": str(final_directory),
        "export_manifest_path": str(final_directory / "export-manifest.json"),
        "coordinate_files_written": sum(
            int(row["trajectory_path"] is not None) + len(row["representatives"])
            for row in output_records
        ),
        "error_count": 0,
        "warning_count": sum(
            issue.get("severity") == "warning"
            for issue in source_report.get("issues", [])
            if isinstance(issue, dict)
        ),
        "issues": [
            issue for issue in source_report.get("issues", []) if isinstance(issue, dict)
        ],
    }


def state_coordinate_exports_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return state_coordinate_exports_project(project_path, hash_content=hash_content)
    except (
        AlternativeClusteringError,
        AtomMappingError,
        ClusteringAnalysisError,
        CoordinateReadError,
        GeometryError,
        ManifestValidationError,
        PCAAnalysisError,
        PCAFESAnalysisError,
        PeriodicReconstructionError,
        RepresentativeFrameError,
        StateCoordinateExportError,
        OSError,
        ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "state_coordinate_exports",
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
                    "code": "STATE_COORDINATE_EXPORT_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
