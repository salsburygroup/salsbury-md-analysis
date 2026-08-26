"""Experimental trajectory-derived multivalent molecular-bridge networks.

A bridge is one mediator residue (or supported monatomic ion) simultaneously
contacting at least two distinct solute residues in one selected frame.  The
native observation is retained as a hyperedge; pairwise residue edges are a
separately labeled projection for network interoperability.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .chemical_identity import (
    ION_RESIDUES, NUCLEIC_RESIDUES, PROTEIN_RESIDUES, WATER_RESIDUES,
)
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import MomentError, sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .trajectory_contracts import (
    TrajectoryContractError, frame_axis_value, normalize_segment_axis,
)
from .validation import positive_integer
from .water_mediated_hydrogen_bonds import (
    WaterMediatedHydrogenBondError, neighbor_pairs_within,
)


class MultivalentBridgeError(ValueError):
    """Raised when a multivalent-bridge contract cannot be evaluated safely."""


ResidueKey = Tuple[str, int, str, str]
EdgeKey = Tuple[str, str, str]


def _finite_positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise MultivalentBridgeError(f"{name} must be finite and positive")
    return float(value)


def _string_set(value: object, name: str) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise MultivalentBridgeError(f"{name} must be an array of nonempty strings")
    normalized = tuple(sorted({item.strip().upper() for item in value}))
    if len(normalized) != len(value):
        raise MultivalentBridgeError(f"{name} values must be unique")
    return normalized


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("multivalent_molecular_bridges")
        if isinstance(definitions, dict) else None
    )
    if not isinstance(raw, dict):
        raise MultivalentBridgeError(
            "definitions.multivalent_molecular_bridges must be an object"
        )
    required = {
        "frame_stride", "frame_selection", "maximum_frames",
        "include_supported_ions", "include_recognized_waters",
        "mediator_residue_names",
        "solute_residue_classes", "solute_residue_names",
        "mediator_atom_elements", "solute_atom_elements",
        "contact_cutoff_angstrom", "water_contact_cutoff_angstrom",
        "minimum_distinct_residues",
        "maximum_neighbor_pairs_per_frame", "maximum_bridge_records",
        "minimum_evaluated_frames_per_system",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise MultivalentBridgeError(
            "multivalent-bridge settings mismatch; missing="
            + ",".join(missing) + "; unknown=" + ",".join(unknown)
        )
    for name in ("include_supported_ions", "include_recognized_waters"):
        if not isinstance(raw[name], bool):
            raise MultivalentBridgeError(f"{name} must be boolean")
    stride = positive_integer(
        raw["frame_stride"], "frame_stride", error_type=MultivalentBridgeError
    )
    result = dict(raw)
    result.update({
        "frame_stride": stride,
        "frame_selection": normalize_frame_selection(
            raw["frame_selection"], stride, error_type=MultivalentBridgeError,
        ),
        "maximum_frames": positive_integer(
            raw["maximum_frames"], "maximum_frames",
            error_type=MultivalentBridgeError,
        ),
        "mediator_residue_names": _string_set(
            raw["mediator_residue_names"], "mediator_residue_names"
        ),
        "solute_residue_names": _string_set(
            raw["solute_residue_names"], "solute_residue_names"
        ),
        "mediator_atom_elements": _string_set(
            raw["mediator_atom_elements"], "mediator_atom_elements"
        ),
        "solute_atom_elements": _string_set(
            raw["solute_atom_elements"], "solute_atom_elements"
        ),
        "contact_cutoff_angstrom": _finite_positive(
            raw["contact_cutoff_angstrom"], "contact_cutoff_angstrom"
        ),
        "water_contact_cutoff_angstrom": _finite_positive(
            raw["water_contact_cutoff_angstrom"],
            "water_contact_cutoff_angstrom",
        ),
        "minimum_distinct_residues": positive_integer(
            raw["minimum_distinct_residues"], "minimum_distinct_residues",
            error_type=MultivalentBridgeError,
        ),
        "maximum_neighbor_pairs_per_frame": positive_integer(
            raw["maximum_neighbor_pairs_per_frame"],
            "maximum_neighbor_pairs_per_frame", error_type=MultivalentBridgeError,
        ),
        "maximum_bridge_records": positive_integer(
            raw["maximum_bridge_records"], "maximum_bridge_records",
            error_type=MultivalentBridgeError,
        ),
        "minimum_evaluated_frames_per_system": positive_integer(
            raw["minimum_evaluated_frames_per_system"],
            "minimum_evaluated_frames_per_system", error_type=MultivalentBridgeError,
        ),
    })
    classes = _string_set(raw["solute_residue_classes"], "solute_residue_classes")
    if not classes or not set(classes) <= {"PROTEIN", "NUCLEIC_ACID", "OTHER"}:
        raise MultivalentBridgeError(
            "solute_residue_classes must contain protein, nucleic_acid, or other"
        )
    result["solute_residue_classes"] = tuple(value.lower() for value in classes)
    if (
        not result["include_supported_ions"]
        and not result["include_recognized_waters"]
        and not result["mediator_residue_names"]
    ):
        raise MultivalentBridgeError(
            "at least one mediator source must be enabled or declared"
        )
    return result


def _residue_key(atom: AtomRecord) -> ResidueKey:
    return atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name


def _residue_id(key: ResidueKey) -> str:
    chain, number, insertion, name = key
    return f"{chain or '_'}:{number}{insertion}:{name}"


def _identity(key: ResidueKey, atom_indices: Sequence[int]) -> Dict[str, object]:
    return {
        "residue_id": _residue_id(key),
        "chain_id": key[0],
        "residue_number": key[1],
        "insertion_code": key[2],
        "residue_name": key[3],
        "atom_indices": list(atom_indices),
    }


def _residue_class(name: str) -> str:
    normalized = name.upper()
    if normalized in PROTEIN_RESIDUES:
        return "protein"
    if normalized in NUCLEIC_RESIDUES:
        return "nucleic_acid"
    return "other"


def _element_selected(atom: AtomRecord, allowed: Sequence[str]) -> bool:
    element = atom.element.strip().upper()
    return element != "H" and (not allowed or element in allowed)


def _topology_groups(
    atoms: Sequence[AtomRecord], settings: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[ResidueKey, List[int]] = defaultdict(list)
    for atom in atoms:
        grouped[_residue_key(atom)].append(atom.atom_index)
    mediator_names = set(settings["mediator_residue_names"])
    explicit_solute_names = set(settings["solute_residue_names"])
    solute_classes = set(settings["solute_residue_classes"])
    mediator_elements = settings["mediator_atom_elements"]
    solute_elements = settings["solute_atom_elements"]
    mediators: List[Dict[str, object]] = []
    solutes: List[Dict[str, object]] = []
    for key, indices in sorted(grouped.items(), key=lambda row: (row[0], row[1])):
        name = key[3].strip().upper()
        is_supported_ion = bool(settings["include_supported_ions"]) and name in ION_RESIDUES
        is_recognized_water = (
            bool(settings["include_recognized_waters"])
            and name in WATER_RESIDUES
        )
        is_mediator = is_supported_ion or is_recognized_water or name in mediator_names
        if is_mediator:
            selected = [
                index for index in indices
                if (
                    atoms[index].element.strip().upper() == "O"
                    if is_recognized_water
                    else _element_selected(
                        atoms[index], mediator_elements  # type: ignore[arg-type]
                    )
                )
            ]
            if selected:
                identity = _identity(key, selected)
                identity.update({
                    "mediator_type": "WATER" if is_recognized_water else name,
                    "mediator_kind": (
                        "supported_ion" if is_supported_ion
                        else "recognized_water" if is_recognized_water
                        else "declared_residue"
                    ),
                    "mediator_id": f"{_residue_id(key)}:a{min(indices)}",
                })
                mediators.append(identity)
            continue
        if name in WATER_RESIDUES or name in ION_RESIDUES:
            continue
        residue_class = _residue_class(name)
        if name not in explicit_solute_names and residue_class not in solute_classes:
            continue
        selected = [
            index for index in indices
            if _element_selected(atoms[index], solute_elements)  # type: ignore[arg-type]
        ]
        if selected:
            identity = _identity(key, selected)
            identity["residue_class"] = residue_class
            solutes.append(identity)
    if len(solutes) < int(settings["minimum_distinct_residues"]):
        raise MultivalentBridgeError(
            "topology contains fewer selected solute residues than the bridge minimum"
        )
    return mediators, solutes


def _runs(
    rows: Sequence[Tuple[Mapping[str, object], bool]],
) -> List[Dict[str, object]]:
    """Return true-state runs without crossing a declared segment boundary."""

    result: List[Dict[str, object]] = []
    start = 0
    while start < len(rows):
        if not rows[start][1]:
            start += 1
            continue
        end = start + 1
        while end < len(rows) and rows[end][1]:
            end += 1
        first = rows[start][0]
        last = rows[end - 1][0]
        first_axis = first["axis_value"]
        last_axis = last["axis_value"]
        assert isinstance(first_axis, (int, float))
        assert isinstance(last_axis, (int, float))
        result.append({
            "start_source_frame_index": first["source_frame_index"],
            "end_source_frame_index": last["source_frame_index"],
            "selected_observation_count": end - start,
            "source_frame_span": (
                int(last["source_frame_index"])
                - int(first["source_frame_index"])
            ),
            "axis_kind": first["axis_kind"],
            "axis_start": first_axis,
            "axis_end": last_axis,
            "axis_span": float(last_axis) - float(first_axis),
            "left_boundary_censored": start == 0,
            "right_boundary_censored": end == len(rows),
        })
        start = end
    return result


def _run_summary(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    complete = [
        row for row in runs
        if not row["left_boundary_censored"] and not row["right_boundary_censored"]
    ]
    return {
        "event_count": len(runs),
        "complete_event_count": len(complete),
        "selected_observation_count_summary": sample_summary(
            float(row["selected_observation_count"]) for row in runs
        ),
        "complete_selected_observation_count_summary": (
            sample_summary(float(row["selected_observation_count"]) for row in complete)
            if complete else None
        ),
        "axis_span_summary": sample_summary(float(row["axis_span"]) for row in runs),
        "complete_axis_span_summary": (
            sample_summary(float(row["axis_span"]) for row in complete)
            if complete else None
        ),
    }


def _distribution(counter: Mapping[int, int]) -> List[Dict[str, object]]:
    total = sum(counter.values())
    return [
        {
            "distinct_residue_count": count,
            "mediator_frame_count": occurrences,
            "fraction": occurrences / total,
        }
        for count, occurrences in sorted(counter.items())
    ]


def multivalent_molecular_bridges_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(str(context["system_manifest_path"]))
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    frame_plan, frame_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=MultivalentBridgeError,
    )
    if int(frame_report["selected_frame_count"]) > int(settings["maximum_frames"]):
        raise MultivalentBridgeError("maximum_frames gate exceeded by frame selection")

    frame_summaries: List[Dict[str, object]] = []
    bridge_records: List[Dict[str, object]] = []
    bridge_events: List[Dict[str, object]] = []
    segment_reports: List[Dict[str, object]] = []
    mediator_states: Dict[Tuple[str, str, str, str], List[Tuple[Mapping[str, object], bool]]] = {}
    segment_edge_states: Dict[
        Tuple[str, str, str], List[Tuple[Mapping[str, object], set[EdgeKey]]]
    ] = {}
    mediator_info: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    mediator_frames: Counter[Tuple[str, str, str]] = Counter()
    mediator_bridge_frames: Counter[Tuple[str, str, str]] = Counter()
    mediator_interchain_frames: Counter[Tuple[str, str, str]] = Counter()
    mediator_multiplicity: Dict[Tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    type_frame_hits: Counter[Tuple[str, str]] = Counter()
    replica_frames: Counter[Tuple[str, str]] = Counter()
    system_frames: Counter[str] = Counter()
    edge_frame_hits: Counter[Tuple[str, str, EdgeKey]] = Counter()
    edge_mediator_occurrences: Counter[Tuple[str, str, EdgeKey]] = Counter()
    topology_reports: List[Dict[str, object]] = []

    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            mediators, solutes = _topology_groups(atoms, settings)
            mediator_owner = {
                int(index): position
                for position, row in enumerate(mediators)
                for index in row["atom_indices"]  # type: ignore[union-attr]
            }
            solute_owner = {
                int(index): position
                for position, row in enumerate(solutes)
                for index in row["atom_indices"]  # type: ignore[union-attr]
            }
            mediator_atom_indices = sorted(mediator_owner)
            solute_atom_indices = sorted(solute_owner)
            evaluated_indices = tuple(sorted(set(mediator_atom_indices) | set(solute_atom_indices)))
            topology_reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "topology_path": str(topology_path),
                "mediator_count": len(mediators),
                "mediator_type_counts": dict(sorted(Counter(
                    str(row["mediator_type"]) for row in mediators
                ).items())),
                "solute_residue_count": len(solutes),
                "mediators": mediators,
                "solute_residues": solutes,
            })
            for row in mediators:
                key = (system_id, replica_id, str(row["mediator_id"]))
                mediator_info[key] = row
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                selected_indices = frame_plan[(system_id, replica_id, segment_id)]
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                axis = normalize_segment_axis(
                    segment, str(output_time_unit) if output_time_unit else None
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                states_by_mediator = {
                    str(row["mediator_id"]): [] for row in mediators
                }
                edge_rows: List[Tuple[Mapping[str, object], set[EdgeKey]]] = []
                evaluated = 0
                segment_bridge_records = 0
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, processor.policy),
                ):
                    selected = frame_selected(
                        raw_frame.frame_index, selected_indices,
                        int(settings["frame_stride"]),
                    )
                    if not selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        evaluated_indices,
                    )
                    if not selected:
                        continue
                    evaluated += 1
                    replica_frames[(system_id, replica_id)] += 1
                    system_frames[system_id] += 1
                    meta: Dict[str, object] = {
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_unit": (
                            str(output_time_unit)
                            if axis["kind"] == "physical_time" else "sample"
                        ),
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                    }
                    try:
                        pairs = neighbor_pairs_within(
                            frame.coordinates_angstrom,
                            mediator_atom_indices,
                            solute_atom_indices,
                            max(
                                float(settings["contact_cutoff_angstrom"]),
                                float(settings["water_contact_cutoff_angstrom"]),
                            ),
                            frame.cell_vectors_angstrom,
                            maximum_pairs=int(settings["maximum_neighbor_pairs_per_frame"]),
                        )
                    except WaterMediatedHydrogenBondError as exc:
                        raise MultivalentBridgeError(str(exc)) from exc
                    minimum_distances: Dict[Tuple[int, int], float] = {}
                    for mediator_atom, solute_atom, distance in pairs:
                        mediator = mediators[mediator_owner[mediator_atom]]
                        applicable_cutoff = (
                            float(settings["water_contact_cutoff_angstrom"])
                            if mediator["mediator_kind"] == "recognized_water"
                            else float(settings["contact_cutoff_angstrom"])
                        )
                        if distance > applicable_cutoff:
                            continue
                        owner_pair = (
                            mediator_owner[mediator_atom], solute_owner[solute_atom]
                        )
                        previous = minimum_distances.get(owner_pair)
                        if previous is None or distance < previous:
                            minimum_distances[owner_pair] = distance
                    contacts_by_mediator: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
                    for (mediator_position, solute_position), distance in sorted(
                        minimum_distances.items()
                    ):
                        contacts_by_mediator[mediator_position].append(
                            (solute_position, distance)
                        )
                    frame_edges: set[EdgeKey] = set()
                    frame_bridge_types: set[str] = set()
                    frame_bridge_count = 0
                    interchain_count = 0
                    for mediator_position, mediator in enumerate(mediators):
                        contacts = contacts_by_mediator.get(mediator_position, [])
                        multiplicity = len(contacts)
                        mediator_id = str(mediator["mediator_id"])
                        summary_key = (system_id, replica_id, mediator_id)
                        mediator_frames[summary_key] += 1
                        mediator_multiplicity[summary_key][multiplicity] += 1
                        is_bridge = multiplicity >= int(
                            settings["minimum_distinct_residues"]
                        )
                        states_by_mediator[mediator_id].append((meta, is_bridge))
                        if not is_bridge:
                            continue
                        mediator_bridge_frames[summary_key] += 1
                        frame_bridge_types.add(str(mediator["mediator_type"]))
                        contacted_rows = [
                            {
                                "residue": solutes[position],
                                "minimum_distance_angstrom": distance,
                            }
                            for position, distance in contacts
                        ]
                        chains = {
                            str(solutes[position]["chain_id"])
                            for position, _ in contacts
                        }
                        is_interchain = len(chains) >= 2
                        mediator_interchain_frames[summary_key] += int(is_interchain)
                        interchain_count += int(is_interchain)
                        frame_bridge_count += 1
                        bridge = {
                            **meta,
                            "mediator": mediator,
                            "distinct_residue_count": multiplicity,
                            "distinct_chain_count": len(chains),
                            "interchain_bridge": is_interchain,
                            "contacted_residues": contacted_rows,
                        }
                        bridge_records.append(bridge)
                        segment_bridge_records += 1
                        if len(bridge_records) > int(settings["maximum_bridge_records"]):
                            raise MultivalentBridgeError(
                                "maximum_bridge_records gate exceeded"
                            )
                        for left, right in combinations(
                            sorted(
                                str(solutes[position]["residue_id"])
                                for position, _ in contacts
                            ),
                            2,
                        ):
                            edge = (str(mediator["mediator_type"]), left, right)
                            frame_edges.add(edge)
                            edge_mediator_occurrences[(system_id, replica_id, edge)] += 1
                    for edge in frame_edges:
                        edge_frame_hits[(system_id, replica_id, edge)] += 1
                    for mediator_type in frame_bridge_types:
                        type_frame_hits[(system_id, mediator_type)] += 1
                    edge_rows.append((meta, frame_edges))
                    frame_summaries.append({
                        **meta,
                        "configured_mediator_count": len(mediators),
                        "active_bridge_count": frame_bridge_count,
                        "active_interchain_bridge_count": interchain_count,
                        "active_projected_edge_count": len(frame_edges),
                    })
                for mediator_id, rows in states_by_mediator.items():
                    mediator_states[(system_id, replica_id, segment_id, mediator_id)] = rows
                segment_edge_states[(system_id, replica_id, segment_id)] = edge_rows
                segment_reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "evaluated_frame_count": evaluated,
                    "bridge_record_count": segment_bridge_records,
                })

    if not mediator_info:
        raise MultivalentBridgeError(
            "project contains no configured mediator residues, recognized waters, "
            "or supported ions"
        )
    for system_id in sorted(system_frames):
        if system_frames[system_id] < int(settings["minimum_evaluated_frames_per_system"]):
            raise MultivalentBridgeError(
                f"system {system_id} produced {system_frames[system_id]} selected frames; "
                "minimum_evaluated_frames_per_system was not met"
            )
    if not frame_summaries:
        raise MultivalentBridgeError("no trajectory frames were evaluated")

    events_by_mediator: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for (system_id, replica_id, segment_id, mediator_id), rows in sorted(
        mediator_states.items()
    ):
        for event in _runs(rows):
            record = {
                "system_id": system_id,
                "replica_id": replica_id,
                "segment_id": segment_id,
                "mediator_id": mediator_id,
                **event,
            }
            bridge_events.append(record)
            events_by_mediator[(system_id, replica_id, mediator_id)].append(record)

    edge_events: Dict[Tuple[str, str, EdgeKey], List[Dict[str, object]]] = defaultdict(list)
    for (system_id, replica_id, segment_id), rows in sorted(segment_edge_states.items()):
        universe = sorted({edge for _, active in rows for edge in active})
        for edge in universe:
            flags = [(meta, edge in active) for meta, active in rows]
            for event in _runs(flags):
                edge_events[(system_id, replica_id, edge)].append({
                    "segment_id": segment_id, **event,
                })

    mediator_summaries = []
    type_counts: Counter[Tuple[str, str]] = Counter()
    type_bridge_counts: Counter[Tuple[str, str]] = Counter()
    type_multiplicity: Dict[Tuple[str, str], Counter[int]] = defaultdict(Counter)
    type_events: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for key in sorted(mediator_frames):
        system_id, replica_id, mediator_id = key
        info = mediator_info[key]
        evaluated = mediator_frames[key]
        bridge_count = mediator_bridge_frames[key]
        mediator_type = str(info["mediator_type"])
        type_key = (system_id, mediator_type)
        type_counts[type_key] += evaluated
        type_bridge_counts[type_key] += bridge_count
        type_multiplicity[type_key].update(mediator_multiplicity[key])
        type_events[type_key].extend(events_by_mediator[key])
        mediator_summaries.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "mediator": info,
            "evaluated_frame_count": evaluated,
            "bridge_frame_count": bridge_count,
            "bridge_occupancy": bridge_count / evaluated,
            "interchain_bridge_frame_count": mediator_interchain_frames[key],
            "interchain_bridge_occupancy": mediator_interchain_frames[key] / evaluated,
            "multiplicity_distribution": _distribution(mediator_multiplicity[key]),
            "bridge_residence": _run_summary(events_by_mediator[key]),
        })

    mediator_type_summaries = [
        {
            "system_id": key[0],
            "mediator_type": key[1],
            "evaluated_mediator_frame_count": type_counts[key],
            "bridge_mediator_frame_count": type_bridge_counts[key],
            "bridge_occupancy": type_bridge_counts[key] / type_counts[key],
            "bridge_occupancy_definition": (
                "bridge mediator-frames divided by all evaluated mediator-frames "
                "of this type"
            ),
            "frames_with_any_type_bridge": type_frame_hits[key],
            "frame_bridge_occupancy": type_frame_hits[key] / system_frames[key[0]],
            "mean_active_bridge_mediators_per_frame": (
                type_bridge_counts[key] / system_frames[key[0]]
            ),
            "multiplicity_distribution": _distribution(type_multiplicity[key]),
            "bridge_residence": _run_summary(type_events[key]),
        }
        for key in sorted(type_counts)
    ]

    system_summaries = []
    for system_id in sorted(system_frames):
        frames = [
            row for row in frame_summaries if row["system_id"] == system_id
        ]
        records = [
            row for row in bridge_records if row["system_id"] == system_id
        ]
        inventory: Counter[str] = Counter()
        for topology in topology_reports:
            if topology["system_id"] != system_id:
                continue
            inventory.update(topology["mediator_type_counts"])  # type: ignore[arg-type]
        evaluated = len(frames)
        system_summaries.append({
            "system_id": system_id,
            "evaluated_frame_count": evaluated,
            "replica_count": sum(
                1 for candidate in replica_frames if candidate[0] == system_id
            ),
            "topology_mediator_type_counts_across_replicas": dict(
                sorted(inventory.items())
            ),
            "bridge_hyperedge_record_count": len(records),
            "mean_active_bridges_per_frame": (
                sum(int(row["active_bridge_count"]) for row in frames) / evaluated
            ),
            "frames_with_any_bridge": sum(
                int(row["active_bridge_count"]) > 0 for row in frames
            ),
            "frames_with_interchain_bridge": sum(
                int(row["active_interchain_bridge_count"]) > 0 for row in frames
            ),
        })

    projected_edges = []
    for key in sorted(edge_frame_hits):
        system_id, replica_id, edge = key
        evaluated = replica_frames[(system_id, replica_id)]
        runs = edge_events[key]
        projected_edges.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "mediator_type": edge[0],
            "residue_i": edge[1],
            "residue_j": edge[2],
            "evaluated_frame_count": evaluated,
            "frames_with_bridge": edge_frame_hits[key],
            "bridge_occupancy": edge_frame_hits[key] / evaluated,
            "mediator_frame_occurrence_count": edge_mediator_occurrences[key],
            "bridge_residence": _run_summary(runs),
        })

    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if int(frame_report["selected_frame_count"]) < int(frame_report["source_frame_count"]):
        issues.append({
            "severity": "warning",
            "code": "FRAME_SUBSAMPLING",
            "message": (
                f"multivalent bridges evaluated {frame_report['selected_frame_count']} "
                f"of {frame_report['source_frame_count']} source frames; residence "
                "runs describe consecutive selected observations"
            ),
        })
    if not bridge_records:
        issues.append({
            "severity": "warning",
            "code": "NO_MULTIVALENT_BRIDGES_OBSERVED",
            "message": "no configured mediator contacted the required number of residues",
        })
    return {
        "module_id": "multivalent_molecular_bridges",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "frame_selection": frame_report,
        "topology_reports": topology_reports,
        "segment_reports": segment_reports,
        "frame_summaries": frame_summaries,
        "bridge_hyperedges": bridge_records,
        "bridge_events": bridge_events,
        "system_summaries": system_summaries,
        "mediator_summaries": mediator_summaries,
        "mediator_type_summaries": mediator_type_summaries,
        "projected_residue_edges": projected_edges,
        "hyperedge_contract": (
            "one mediator residue simultaneously contacts the listed distinct "
            "solute residues in one selected frame"
        ),
        "pairwise_projection_contract": (
            "each k-residue bridge contributes all k choose 2 residue pairs; "
            "pairwise edges are projections and do not replace the retained hyperedge"
        ),
        "residence_contract": (
            "consecutive bridge-positive selected observations within one declared "
            "trajectory segment; boundary-censored events remain labeled"
        ),
        "frame_identity_contract": (
            "system_id, replica_id, segment_id, and source_frame_index permit "
            "exact post-analysis joins to conformational-state assignments or "
            "other frame-aligned observables"
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Distance-defined bridges are geometric and do not establish binding affinity, energetic stabilization, phase separation, or mechanism.",
            "Contact cutoffs, mediator protonation, atom-element filters, and the minimum residue multiplicity require chemistry-specific sensitivity analysis.",
            "Recognized-water bridges use water oxygen proximity to selected solute atoms; the separate water-mediated hydrogen-bond module provides donor, acceptor, and angular chemistry.",
            "A multivalent mediator is one topology residue in this implementation; multi-residue mediator molecules require an explicit future grouping contract.",
            "Selected-observation residence runs are descriptive, segment-safe, and boundary-censored; subsampled runs are not continuous-time lifetimes.",
            "Pairwise projection expands a k-residue hyperedge into k choose 2 edges and can overemphasize high-multiplicity mediators; the native hyperedges remain authoritative.",
            "Replica and system identities remain explicit; pooled frame counts are not independent-replica uncertainty.",
        ],
    }


def multivalent_molecular_bridges_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return multivalent_molecular_bridges_project(
            project_path, hash_content=hash_content
        )
    except (
        AtomMappingError, CoordinateReadError, ManifestValidationError,
        MomentError, MultivalentBridgeError, PeriodicReconstructionError,
        TrajectoryContractError, OSError,
    ) as exc:
        messages = (
            list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        )
        return {
            "module_id": "multivalent_molecular_bridges",
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
                    "code": "MULTIVALENT_MOLECULAR_BRIDGES_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
