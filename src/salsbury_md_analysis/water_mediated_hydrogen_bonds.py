"""Sparse, topology-aware one-water hydrogen-bond network analysis.

The implementation discovers solute chemistry and water molecules from atom
identity plus explicit connectivity.  It uses a spatial cell list rather than
forming the solute-by-water Cartesian product, and stores only observed paths.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CellVectors, CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .hydrogen_bond_chemistry import (
    WATER_RESIDUES, chemistry_summary, infer_atom_chemical_roles, scope_allows,
)
from .hydrogen_bond_discovery import (
    HydrogenBondDiscoveryError, _cutoff_definitions, _positive_float,
)
from .hydrogen_bonds import angle_degrees, distance_angstrom
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor, PeriodicReconstructionError, load_connectivity,
)
from .trajectory_contracts import (
    TrajectoryContractError, frame_axis_value, normalize_segment_axis,
)
from .validation import positive_integer


class WaterMediatedHydrogenBondError(ValueError):
    """Raised when a water-network contract cannot be evaluated safely."""


def _residue_key(atom: AtomRecord) -> Tuple[str, int, str, str]:
    return atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name


def _same_residue(first: AtomRecord, second: AtomRecord) -> bool:
    return _residue_key(first) == _residue_key(second)


def _adjacency(atom_count: int, bonds: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {index: [] for index in range(atom_count)}
    for first, second in bonds:
        result[first].append(second)
        result[second].append(first)
    for values in result.values():
        values.sort()
    return result


def discover_waters(
    atoms: Sequence[AtomRecord], bonds: Sequence[Tuple[int, int]]
) -> List[Dict[str, object]]:
    """Return connectivity-backed water oxygen/hydrogen identities."""

    adjacency = _adjacency(len(atoms), bonds)
    recognized = {
        atom.atom_index for atom in atoms
        if atom.residue_name.strip().upper() in WATER_RESIDUES
    }
    oxygen_indices = sorted(
        index for index in recognized if atoms[index].element.upper() == "O"
    )
    hydrogen_indices = {
        index for index in recognized if atoms[index].element.upper() == "H"
    }
    waters = []
    hydrogen_owners: Dict[int, List[int]] = defaultdict(list)
    for oxygen in oxygen_indices:
        hydrogens = sorted(
            index for index in adjacency[oxygen]
            if index in hydrogen_indices
        )
        if not hydrogens:
            raise WaterMediatedHydrogenBondError(
                f"recognized water oxygen {oxygen} has no connectivity-declared O-H bond"
            )
        for hydrogen in hydrogens:
            hydrogen_owners[hydrogen].append(oxygen)
        atom = atoms[oxygen]
        key = _residue_key(atom)
        water_id = (
            f"W:{key[0] or '_'}:{key[3]}:{key[1]}:{key[2] or '_'}:O{oxygen}"
        )
        waters.append({
            "water_id": water_id,
            "oxygen_atom_index": oxygen,
            "hydrogen_atom_indices": hydrogens,
            "oxygen_identity": atoms[oxygen].as_dict(),
        })
    multiply_owned = {
        hydrogen: owners for hydrogen, owners in hydrogen_owners.items()
        if len(owners) != 1
    }
    if multiply_owned:
        hydrogen, owners = sorted(multiply_owned.items())[0]
        raise WaterMediatedHydrogenBondError(
            f"recognized water hydrogen {hydrogen} is bonded to multiple water oxygens: {owners}"
        )
    unattached = sorted(hydrogen_indices.difference(hydrogen_owners))
    if unattached:
        raise WaterMediatedHydrogenBondError(
            f"recognized water hydrogen {unattached[0]} is not bonded to a recognized water oxygen"
        )
    if not waters:
        raise WaterMediatedHydrogenBondError(
            "topology contains no recognized, connectivity-backed water molecules"
        )
    return waters


def discover_solute_endpoints(
    atoms: Sequence[AtomRecord], bonds: Sequence[Tuple[int, int]]
) -> Tuple[List[Dict[str, object]], Dict[int, object], Dict[int, List[int]]]:
    """Return a fixed solute endpoint universe before frame evaluation."""

    roles = infer_atom_chemical_roles(atoms, bonds)
    adjacency = _adjacency(len(atoms), bonds)
    endpoints = []
    donor_hydrogens: Dict[int, List[int]] = {}
    for atom_index, role in sorted(roles.items()):
        if role.entity_class not in {"protein", "nucleic_acid", "ligand"}:
            continue
        hydrogens = [
            index for index in adjacency[atom_index]
            if atoms[index].element.upper() == "H"
        ]
        if role.donor:
            if not hydrogens:
                raise WaterMediatedHydrogenBondError(
                    f"discovered donor atom {atom_index} has no explicit bonded hydrogen"
                )
            donor_hydrogens[atom_index] = hydrogens
        if role.donor or role.acceptor:
            endpoints.append({
                "endpoint_id": f"E{atom_index}",
                "atom_index": atom_index,
                "identity": atoms[atom_index].as_dict(),
                "chemistry": role.as_dict(),
                "donor_hydrogen_atom_indices": hydrogens if role.donor else [],
            })
    if len(endpoints) < 2:
        raise WaterMediatedHydrogenBondError(
            "automatic chemistry produced fewer than two solute hydrogen-bond endpoints"
        )
    return endpoints, roles, donor_hydrogens


def neighbor_pairs_within(
    coordinates: Sequence[Sequence[float]],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    cutoff: float,
    cell: CellVectors | None,
    *,
    maximum_pairs: int,
) -> List[Tuple[int, int, float]]:
    """Return exact nearby cross-set pairs using nonperiodic or periodic bins.

    Periodic fractional bin reach is bounded by reciprocal-vector norms.  The
    final distance check uses the suite's exact triclinic minimum-image routine.
    """

    if cutoff <= 0.0 or not math.isfinite(cutoff):
        raise WaterMediatedHydrogenBondError("neighbor cutoff must be finite and positive")
    if not left_indices or not right_indices:
        return []
    points = np.asarray(coordinates, dtype=float)
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    if cell is None:
        bin_counts = None
        transformed = points / cutoff
        for index in right_indices:
            key = tuple(math.floor(value) for value in transformed[index])
            buckets[key].append(index)  # type: ignore[arg-type]

        def keys_for(index: int) -> Set[Tuple[int, int, int]]:
            center = tuple(math.floor(value) for value in transformed[index])
            return {
                tuple(center[axis] + offset[axis] for axis in range(3))
                for offset in product((-1, 0, 1), repeat=3)
            }
    else:
        cell_matrix = np.asarray(cell, dtype=float)
        try:
            inverse = np.linalg.inv(cell_matrix)
        except np.linalg.LinAlgError as exc:
            raise WaterMediatedHydrogenBondError("periodic cell is singular") from exc
        fractional = (points @ inverse) % 1.0
        reaches = cutoff * np.linalg.norm(inverse, axis=0)
        bin_counts = tuple(
            max(1, int(math.floor(1.0 / max(float(reach), 1.0e-15))))
            for reach in reaches
        )
        transformed = fractional * np.asarray(bin_counts, dtype=float)
        for index in right_indices:
            key = tuple(
                int(math.floor(transformed[index, axis])) % bin_counts[axis]
                for axis in range(3)
            )
            buckets[key].append(index)
        offsets = tuple(
            range(-int(math.ceil(reaches[axis] * bin_counts[axis])),
                  int(math.ceil(reaches[axis] * bin_counts[axis])) + 1)
            for axis in range(3)
        )

        def keys_for(index: int) -> Set[Tuple[int, int, int]]:
            center = tuple(
                int(math.floor(transformed[index, axis])) % bin_counts[axis]
                for axis in range(3)
            )
            return {
                tuple((center[axis] + offset[axis]) % bin_counts[axis] for axis in range(3))
                for offset in product(*offsets)
            }

    pairs = []
    for left in left_indices:
        for key in sorted(keys_for(left)):
            for right in buckets.get(key, ()):
                distance = distance_angstrom(coordinates[left], coordinates[right], cell)
                if distance <= cutoff:
                    pairs.append((left, right, distance))
                    if len(pairs) > maximum_pairs:
                        raise WaterMediatedHydrogenBondError(
                            "maximum_neighbor_pairs_per_frame gate exceeded"
                        )
    pairs.sort()
    return pairs


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("water_mediated_hydrogen_bond_networks")
        if isinstance(definitions, dict) else None
    )
    required = {
        "chemistry_policy", "interaction_scope", "water_identity_policy",
        "maximum_bridge_length", "exclude_same_residue_endpoints", "frame_stride",
        "frame_selection",
        "cutoff_policy", "maximum_reference_donor_hydrogen_bond_angstrom",
        "neighbor_search", "maximum_solute_endpoints", "maximum_waters",
        "maximum_evaluated_frames", "maximum_neighbor_pairs_per_frame",
        "maximum_bridge_paths_per_frame", "maximum_sparse_records",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise WaterMediatedHydrogenBondError(
            "definitions.water_mediated_hydrogen_bond_networks fields do not match the contract"
        )
    constants = {
        "chemistry_policy": "automatic_topology_templates_v1",
        "water_identity_policy": "standard_residue_names_connectivity_v1",
        "maximum_bridge_length": 1,
        "neighbor_search": "cell_list_v1",
    }
    for label, expected in constants.items():
        if raw[label] != expected:
            raise WaterMediatedHydrogenBondError(f"{label} must be {expected!r}")
    scopes = {
        "all_solute", "protein_protein", "protein_ligand", "protein_nucleic_acid",
        "nucleic_acid_nucleic_acid", "nucleic_acid_ligand", "ligand_ligand",
    }
    if raw["interaction_scope"] not in scopes:
        raise WaterMediatedHydrogenBondError("interaction_scope is not supported")
    if not isinstance(raw["exclude_same_residue_endpoints"], bool):
        raise WaterMediatedHydrogenBondError(
            "exclude_same_residue_endpoints must be boolean"
        )
    result = dict(raw)
    for label in (
        "frame_stride", "maximum_solute_endpoints", "maximum_waters",
        "maximum_evaluated_frames", "maximum_neighbor_pairs_per_frame",
        "maximum_bridge_paths_per_frame", "maximum_sparse_records",
    ):
        result[label] = positive_integer(
            raw[label], label, error_type=WaterMediatedHydrogenBondError
        )
    result["maximum_reference_donor_hydrogen_bond_angstrom"] = _positive_float(
        raw["maximum_reference_donor_hydrogen_bond_angstrom"],
        "maximum_reference_donor_hydrogen_bond_angstrom",
    )
    result["frame_selection"] = normalize_frame_selection(
        raw["frame_selection"], int(result["frame_stride"]),
        error_type=WaterMediatedHydrogenBondError,
    )
    result["cutoff_definitions"] = _cutoff_definitions(raw["cutoff_policy"])
    return result


def _frame_selection_plan(
    system: Mapping[str, object], system_path: Path, coordinate_unit: str,
    settings: Mapping[str, object],
) -> Tuple[Dict[Tuple[str, str, str], Set[int] | None], Dict[str, object]]:
    """Plan fixed-stride or balanced per-replica frame selection.

    Uniform budgets operate over each replica's concatenated segment order, so
    long multi-segment trajectories and every declared replica contribute.  A
    global evaluation cap remains a fail-closed resource guard.
    """
    return plan_frame_selection(
        system, system_path, coordinate_unit, settings["frame_selection"],
        frame_stride=int(settings["frame_stride"]),
        maximum_selected_frames=int(settings["maximum_evaluated_frames"]),
        error_type=WaterMediatedHydrogenBondError,
    )


def _edge_rank(edge: Mapping[str, object]) -> Tuple[float, float, int]:
    return (
        float(edge["donor_acceptor_distance_angstrom"]),
        -float(edge["donor_hydrogen_acceptor_angle_degrees"]),
        int(edge["hydrogen_atom_index"]),
    )


def _scoped_edge_pairs(
    edges: Sequence[Dict[str, object]], roles: Mapping[int, object], scope: str,
) -> Iterable[Tuple[Dict[str, object], Dict[str, object]]]:
    """Yield only endpoint pairs admissible under ``interaction_scope``.

    Generating every within-water pair before applying scope can be quadratic
    at a dense protein--solvent interface.  Partitioning by entity class
    preserves the exact path contract while bounding work to the requested
    scientific interaction class.
    """
    ordered = sorted(
        edges, key=lambda row: (int(row["endpoint_atom_index"]), str(row["endpoint_role"]))
    )
    if scope == "all_solute":
        yield from combinations(ordered, 2)
        return
    entities = {
        "protein_protein": ("protein", "protein"),
        "protein_ligand": ("protein", "ligand"),
        "protein_nucleic_acid": ("protein", "nucleic_acid"),
        "nucleic_acid_nucleic_acid": ("nucleic_acid", "nucleic_acid"),
        "nucleic_acid_ligand": ("nucleic_acid", "ligand"),
        "ligand_ligand": ("ligand", "ligand"),
    }
    if scope not in entities:
        raise WaterMediatedHydrogenBondError("interaction_scope is not supported")
    first_entity, second_entity = entities[scope]
    by_entity: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for edge in ordered:
        entity = str(roles[int(edge["endpoint_atom_index"])].entity_class)  # type: ignore[attr-defined]
        by_entity[entity].append(edge)
    first_edges = by_entity[first_entity]
    second_edges = by_entity[second_entity]
    if first_entity == second_entity:
        yield from combinations(first_edges, 2)
        return
    # The declared class order makes each cross-class pair unique.
    yield from product(first_edges, second_edges)


def _bridge_identity(
    first: Tuple[int, str], second: Tuple[int, str]
) -> Tuple[str, Tuple[int, str], Tuple[int, str], str]:
    ordered = tuple(sorted((first, second)))
    roles = {ordered[0][1], ordered[1][1]}
    relation = (
        "donor_acceptor" if roles == {"donor", "acceptor"}
        else "donor_donor" if roles == {"donor"}
        else "acceptor_acceptor"
    )
    return (
        f"E{ordered[0][0]}:{ordered[0][1]}--E{ordered[1][0]}:{ordered[1][1]}",
        ordered[0], ordered[1], relation,
    )


def _direct_present(
    coordinates: Sequence[Sequence[float]],
    donor: int,
    acceptor: int,
    hydrogens: Sequence[int],
    cutoff: Mapping[str, object],
    cell: CellVectors | None,
) -> bool:
    distance = distance_angstrom(coordinates[donor], coordinates[acceptor], cell)
    if distance > float(cutoff["maximum_donor_acceptor_distance_angstrom"]):
        return False
    return any(
        angle_degrees(
            coordinates[donor], coordinates[hydrogen], coordinates[acceptor], cell
        ) >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
        for hydrogen in hydrogens
    )


def evaluate_water_bridge_frame(
    coordinates: Sequence[Sequence[float]],
    atoms: Sequence[AtomRecord],
    endpoints: Sequence[Mapping[str, object]],
    roles: Mapping[int, object],
    donor_hydrogens: Mapping[int, Sequence[int]],
    waters: Sequence[Mapping[str, object]],
    cutoff_definitions: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
    cell: CellVectors | None,
) -> Dict[str, object]:
    """Evaluate one frame into sparse, cutoff-resolved one-water paths."""

    maximum_distance = max(
        float(cutoff["maximum_donor_acceptor_distance_angstrom"])
        for cutoff in cutoff_definitions
    )
    water_by_oxygen = {int(water["oxygen_atom_index"]): water for water in waters}
    water_oxygens = sorted(water_by_oxygen)
    donors = sorted(donor_hydrogens)
    acceptors = sorted(
        int(endpoint["atom_index"]) for endpoint in endpoints
        if bool(endpoint["chemistry"]["acceptor"])  # type: ignore[index]
    )
    max_pairs = int(settings["maximum_neighbor_pairs_per_frame"])
    solute_to_water = neighbor_pairs_within(
        coordinates, donors, water_oxygens, maximum_distance, cell,
        maximum_pairs=max_pairs,
    )
    water_to_solute = neighbor_pairs_within(
        coordinates, water_oxygens, acceptors, maximum_distance, cell,
        maximum_pairs=max_pairs,
    )
    if len(solute_to_water) + len(water_to_solute) > max_pairs:
        raise WaterMediatedHydrogenBondError(
            "combined maximum_neighbor_pairs_per_frame gate exceeded"
        )
    # One best edge per water, endpoint role, and cutoff.  This removes
    # duplicate explicit hydrogens without changing endpoint occupancy.
    edge_maps: List[Dict[Tuple[str, int, str], Dict[str, object]]] = [
        {} for _ in cutoff_definitions
    ]
    for donor, oxygen, distance in solute_to_water:
        water = water_by_oxygen[oxygen]
        for hydrogen in donor_hydrogens[donor]:
            angle = angle_degrees(
                coordinates[donor], coordinates[hydrogen], coordinates[oxygen], cell
            )
            edge = {
                "endpoint_atom_index": donor, "endpoint_role": "donor",
                "orientation": "solute_to_water", "water_id": water["water_id"],
                "donor_atom_index": donor, "hydrogen_atom_index": hydrogen,
                "acceptor_atom_index": oxygen,
                "donor_acceptor_distance_angstrom": distance,
                "donor_hydrogen_acceptor_angle_degrees": angle,
            }
            for cutoff_index, cutoff in enumerate(cutoff_definitions):
                if (
                    distance <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
                    and angle >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
                ):
                    key = (str(water["water_id"]), donor, "donor")
                    prior = edge_maps[cutoff_index].get(key)
                    if prior is None or _edge_rank(edge) < _edge_rank(prior):
                        edge_maps[cutoff_index][key] = edge
    for oxygen, acceptor, distance in water_to_solute:
        water = water_by_oxygen[oxygen]
        for hydrogen in water["hydrogen_atom_indices"]:  # type: ignore[union-attr]
            angle = angle_degrees(
                coordinates[oxygen], coordinates[int(hydrogen)], coordinates[acceptor], cell
            )
            edge = {
                "endpoint_atom_index": acceptor, "endpoint_role": "acceptor",
                "orientation": "water_to_solute", "water_id": water["water_id"],
                "donor_atom_index": oxygen, "hydrogen_atom_index": int(hydrogen),
                "acceptor_atom_index": acceptor,
                "donor_acceptor_distance_angstrom": distance,
                "donor_hydrogen_acceptor_angle_degrees": angle,
            }
            for cutoff_index, cutoff in enumerate(cutoff_definitions):
                if (
                    distance <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
                    and angle >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
                ):
                    key = (str(water["water_id"]), acceptor, "acceptor")
                    prior = edge_maps[cutoff_index].get(key)
                    if prior is None or _edge_rank(edge) < _edge_rank(prior):
                        edge_maps[cutoff_index][key] = edge

    paths_by_cutoff: List[List[Dict[str, object]]] = []
    bridge_dictionary: Dict[str, Dict[str, object]] = {}
    for cutoff_index, (cutoff, edge_map) in enumerate(zip(cutoff_definitions, edge_maps)):
        by_water: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for (water_id, _endpoint, _role), edge in edge_map.items():
            by_water[water_id].append(edge)
        paths = []
        for water_id, edges in sorted(by_water.items()):
            for first, second in _scoped_edge_pairs(
                edges, roles, str(settings["interaction_scope"])
            ):
                first_key = (int(first["endpoint_atom_index"]), str(first["endpoint_role"]))
                second_key = (int(second["endpoint_atom_index"]), str(second["endpoint_role"]))
                if first_key[0] == second_key[0]:
                    continue
                if (
                    bool(settings["exclude_same_residue_endpoints"])
                    and _same_residue(atoms[first_key[0]], atoms[second_key[0]])
                ):
                    continue
                first_entity = roles[first_key[0]].entity_class  # type: ignore[attr-defined]
                second_entity = roles[second_key[0]].entity_class  # type: ignore[attr-defined]
                if not scope_allows(
                    str(first_entity), str(second_entity), str(settings["interaction_scope"])
                ):
                    continue
                bridge_id, ordered_first, ordered_second, relation = _bridge_identity(
                    first_key, second_key
                )
                edge_by_key = {first_key: first, second_key: second}
                ordered_edges = [edge_by_key[ordered_first], edge_by_key[ordered_second]]
                direct = False
                if relation == "donor_acceptor":
                    donor = ordered_first if ordered_first[1] == "donor" else ordered_second
                    acceptor = ordered_first if ordered_first[1] == "acceptor" else ordered_second
                    direct = _direct_present(
                        coordinates, donor[0], acceptor[0], donor_hydrogens[donor[0]],
                        cutoff, cell,
                    )
                bridge_dictionary.setdefault(bridge_id, {
                    "bridge_id": bridge_id,
                    "first_endpoint_atom_index": ordered_first[0],
                    "first_endpoint_role": ordered_first[1],
                    "second_endpoint_atom_index": ordered_second[0],
                    "second_endpoint_role": ordered_second[1],
                    "relation": relation,
                })
                paths.append({
                    "bridge_id": bridge_id, "water_id": water_id,
                    "cutoff_id": cutoff["cutoff_id"], "relation": relation,
                    "direct_hydrogen_bond_present": direct,
                    "edges": ordered_edges,
                })
                if len(paths) > int(settings["maximum_bridge_paths_per_frame"]):
                    raise WaterMediatedHydrogenBondError(
                        "maximum_bridge_paths_per_frame gate exceeded"
                    )
        paths_by_cutoff.append(paths)
    if sum(len(paths) for paths in paths_by_cutoff) > int(
        settings["maximum_bridge_paths_per_frame"]
    ):
        raise WaterMediatedHydrogenBondError(
            "cutoff-expanded maximum_bridge_paths_per_frame gate exceeded"
        )
    return {
        "neighbor_pair_count": len(solute_to_water) + len(water_to_solute),
        "paths_by_cutoff": paths_by_cutoff,
        "bridge_dictionary": bridge_dictionary,
    }


def _residence_runs(
    frame_records: Sequence[Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    any_water_runs: List[Dict[str, object]] = []
    same_water_runs: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for frame in frame_records:
        grouped[(
            str(frame["system_id"]), str(frame["replica_id"]), str(frame["segment_id"])
        )].append(frame)
    for key, frames in sorted(grouped.items()):
        frames.sort(key=lambda row: int(row["source_frame_index"]))
        bridge_waters = [
            {
                str(bridge_id): set(str(value) for value in waters)
                for bridge_id, waters in frame["primary_bridge_water_ids"].items()  # type: ignore[union-attr]
            }
            for frame in frames
        ]
        bridge_ids = sorted({bridge for values in bridge_waters for bridge in values})

        def append_runs(
            values: Sequence[bool], target: List[Dict[str, object]],
            bridge_id: str, water_id: str | None,
        ) -> None:
            start = None
            for index in range(len(values) + 1):
                present = index < len(values) and values[index]
                if present and start is None:
                    start = index
                if not present and start is not None:
                    end = index - 1
                    target.append({
                        "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
                        "bridge_id": bridge_id, **({"water_id": water_id} if water_id else {}),
                        "sampled_frame_count": end - start + 1,
                        "start_source_frame_index": frames[start]["source_frame_index"],
                        "end_source_frame_index": frames[end]["source_frame_index"],
                        "axis_kind": frames[start]["axis_kind"],
                        "start_axis_value": frames[start]["axis_value"],
                        "end_axis_value": frames[end]["axis_value"],
                        "left_censored": start == 0,
                        "right_censored": end == len(values) - 1,
                    })
                    start = None

        for bridge_id in bridge_ids:
            append_runs(
                [bridge_id in values for values in bridge_waters],
                any_water_runs, bridge_id, None,
            )
            water_ids = sorted({
                water for values in bridge_waters for water in values.get(bridge_id, set())
            })
            for water_id in water_ids:
                append_runs(
                    [water_id in values.get(bridge_id, set()) for values in bridge_waters],
                    same_water_runs, bridge_id, water_id,
                )
    return any_water_runs, same_water_runs


def water_mediated_hydrogen_bond_networks_project(
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
    cutoff_definitions = settings["cutoff_definitions"]
    frame_selection_plan, frame_selection_report = _frame_selection_plan(
        system, system_path, coordinate_unit, settings
    )
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    frame_records: List[Dict[str, object]] = []
    frame_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    stats: Dict[Tuple[str, str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"present": 0, "multiplicity_total": 0, "multiplicity_max": 0,
                 "direct_coincident": 0}
    )
    bridge_dictionary: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    endpoint_dictionary: List[Dict[str, object]] = []
    water_dictionary: List[Dict[str, object]] = []
    chemistry_reports = []
    sparse_record_count = 0
    total_neighbor_pairs = 0
    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise WaterMediatedHydrogenBondError(
                    f"{system_id}/{replica_id} requires explicit connectivity"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            bonds, connectivity_provenance = load_connectivity(connectivity_path, len(atoms))
            endpoints, roles, donor_hydrogens = discover_solute_endpoints(atoms, bonds)
            waters = discover_waters(atoms, bonds)
            if len(endpoints) > int(settings["maximum_solute_endpoints"]):
                raise WaterMediatedHydrogenBondError(
                    "maximum_solute_endpoints gate exceeded"
                )
            if len(waters) > int(settings["maximum_waters"]):
                raise WaterMediatedHydrogenBondError("maximum_waters gate exceeded")
            endpoint_dictionary.extend({
                "system_id": system_id, "replica_id": replica_id, **row,
            } for row in endpoints)
            water_dictionary.extend({
                "system_id": system_id, "replica_id": replica_id, **row,
            } for row in waters)
            chemistry_report = chemistry_summary(roles)
            chemistry_reports.append({
                "system_id": system_id, "replica_id": replica_id, **chemistry_report,
                "water_molecule_count": len(waters),
            })
            provisional = int(
                chemistry_report["chemistry_confidence_atom_counts"].get("provisional", 0)
            )
            issues.append({
                "severity": "warning" if provisional else "info",
                "code": "WATER_HBOND_AUTO_CHEMISTRY_PROVISIONAL" if provisional else "WATER_HBOND_AUTO_CHEMISTRY_TEMPLATED",
                "location": f"{system_id}/{replica_id}",
                "message": (
                    f"Automatic solute chemistry includes {provisional} provisional ligand atoms."
                    if provisional else
                    "Automatic solute chemistry used standard protein/nucleic-acid templates."
                ),
                "connectivity": connectivity_provenance,
            })
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            reconstruction_atom_indices = tuple(sorted(
                {int(row["atom_index"]) for row in endpoints}
                | {
                    int(index)
                    for hydrogens in donor_hydrogens.values()
                    for index in hydrogens
                }
            ))
            reference = processor.process(
                next(iter_coordinate_frames(topology_path, coordinate_unit)),
                str(topology_path),
                reconstruction_atom_indices,
            )
            donor_h_pairs = [
                (donor, hydrogen)
                for donor, hydrogens in donor_hydrogens.items() for hydrogen in hydrogens
            ] + [
                (int(water["oxygen_atom_index"]), int(hydrogen))
                for water in waters for hydrogen in water["hydrogen_atom_indices"]  # type: ignore[union-attr]
            ]
            for donor, hydrogen in donor_h_pairs:
                if distance_angstrom(
                    reference.coordinates_angstrom[donor],
                    reference.coordinates_angstrom[hydrogen],
                    reference.cell_vectors_angstrom,
                ) > float(settings["maximum_reference_donor_hydrogen_bond_angstrom"]):
                    raise WaterMediatedHydrogenBondError(
                        f"reference donor-H bond {donor}-{hydrogen} exceeds the declared gate"
                    )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                axis = normalize_segment_axis(
                    segment, str(output_time_unit) if output_time_unit else None
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                # For independent-frame policies, the DCD reader can validate
                # envelopes while seeking past unselected coordinate payloads.
                # Continuous unwrapping must decode every frame to maintain its
                # image history, even when only a uniform subset is reported.
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
                    # make_whole reconstructs each frame independently, so nonselected
                    # frames may be skipped before the O(atom+bonds) traversal.  Continuous
                    # unwrapping is deliberately different: every raw frame must update the
                    # image history even when its output is not sampled.
                    if not selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if not selected:
                        continue
                    if len(frame_records) >= int(settings["maximum_evaluated_frames"]):
                        raise WaterMediatedHydrogenBondError(
                            "maximum_evaluated_frames gate exceeded"
                        )
                    evaluated = evaluate_water_bridge_frame(
                        frame.coordinates_angstrom, atoms, endpoints, roles, donor_hydrogens,
                        waters, cutoff_definitions, settings, frame.cell_vectors_angstrom,
                    )
                    total_neighbor_pairs += int(evaluated["neighbor_pair_count"])
                    bridge_dictionary.update({
                        (system_id, replica_id, str(bridge_id)): {
                            "system_id": system_id, "replica_id": replica_id, **row,
                        }
                        for bridge_id, row in evaluated["bridge_dictionary"].items()  # type: ignore[union-attr]
                    })
                    paths_by_cutoff = evaluated["paths_by_cutoff"]
                    sparse_record_count += sum(len(paths) for paths in paths_by_cutoff)
                    if sparse_record_count > int(settings["maximum_sparse_records"]):
                        raise WaterMediatedHydrogenBondError(
                            "maximum_sparse_records gate exceeded"
                        )
                    key3 = (system_id, replica_id, segment_id)
                    frame_counts[key3] += 1
                    cutoff_bridge_waters = {}
                    primary_bridge_water_ids: Dict[str, List[str]] = {}
                    for cutoff, paths in zip(cutoff_definitions, paths_by_cutoff):
                        grouped_paths: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
                        for path in paths:
                            grouped_paths[str(path["bridge_id"])].append(path)
                        cutoff_bridge_waters[str(cutoff["cutoff_id"])] = {
                            bridge_id: sorted({str(path["water_id"]) for path in values})
                            for bridge_id, values in sorted(grouped_paths.items())
                        }
                        if cutoff["cutoff_id"] == "primary":
                            primary_bridge_water_ids = cutoff_bridge_waters["primary"]
                        for bridge_id, values in grouped_paths.items():
                            stat = stats[(*key3, str(cutoff["cutoff_id"]), bridge_id)]
                            multiplicity = len({str(path["water_id"]) for path in values})
                            stat["present"] += 1
                            stat["multiplicity_total"] += multiplicity
                            stat["multiplicity_max"] = max(stat["multiplicity_max"], multiplicity)
                            stat["direct_coincident"] += int(any(
                                bool(path["direct_hydrogen_bond_present"]) for path in values
                            ))
                    primary_paths = paths_by_cutoff[0]
                    frame_records.append({
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id, "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "representation": "sparse_observed_paths_v1",
                        "neighbor_pair_count": evaluated["neighbor_pair_count"],
                        "primary_paths": primary_paths,
                        "primary_bridge_water_ids": primary_bridge_water_ids,
                        "cutoff_bridge_water_ids": cutoff_bridge_waters,
                    })
    if not endpoint_dictionary or not water_dictionary:
        raise WaterMediatedHydrogenBondError("system manifest contains no replicas")
    occupancies = []
    for key, stat in sorted(stats.items()):
        frame_count = frame_counts[key[:3]]
        cutoff = next(row for row in cutoff_definitions if row["cutoff_id"] == key[3])
        relation = bridge_dictionary[(key[0], key[1], key[4])]["relation"]
        occupancies.append({
            "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
            "cutoff_id": key[3], "cutoff_kind": cutoff["kind"],
            "bridge_id": key[4], "evaluated_frame_count": frame_count,
            "present_frame_count": stat["present"],
            "occupancy_fraction": stat["present"] / frame_count,
            "mean_bridging_water_multiplicity_when_present": (
                stat["multiplicity_total"] / stat["present"]
            ),
            "maximum_bridging_water_multiplicity": stat["multiplicity_max"],
            "direct_coincident_frame_count": (
                stat["direct_coincident"] if relation == "donor_acceptor" else None
            ),
            "direct_coincident_fraction_of_bridge_present_frames": (
                stat["direct_coincident"] / stat["present"]
                if relation == "donor_acceptor" else None
            ),
        })
    any_water_runs, same_water_runs = _residence_runs(frame_records)
    primary_occupancies = [row for row in occupancies if row["cutoff_id"] == "primary"]
    node_values: Dict[Tuple[str, str, str, int], Dict[str, float]] = defaultdict(
        lambda: {"degree": 0.0, "occupancy_weighted_degree": 0.0}
    )
    network_edges = []
    for row in primary_occupancies:
        bridge = bridge_dictionary[(
            str(row["system_id"]), str(row["replica_id"]), str(row["bridge_id"])
        )]
        for endpoint in (
            int(bridge["first_endpoint_atom_index"]),
            int(bridge["second_endpoint_atom_index"]),
        ):
            node_key = (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]), endpoint,
            )
            node_values[node_key]["degree"] += 1.0
            node_values[node_key]["occupancy_weighted_degree"] += float(row["occupancy_fraction"])
        network_edges.append({**row, **bridge})
    representative_frames = []
    for bridge_key in sorted(bridge_dictionary):
        system_id, replica_id, bridge_id = bridge_key
        candidates = []
        for frame in frame_records:
            if (
                str(frame["system_id"]) != system_id
                or str(frame["replica_id"]) != replica_id
            ):
                continue
            paths = [
                path for path in frame["primary_paths"]  # type: ignore[union-attr]
                if path["bridge_id"] == bridge_id
            ]
            if paths:
                worst_distance = min(
                    max(float(edge["donor_acceptor_distance_angstrom"]) for edge in path["edges"])
                    for path in paths
                )
                best_angle = max(
                    min(float(edge["donor_hydrogen_acceptor_angle_degrees"]) for edge in path["edges"])
                    for path in paths
                )
                candidates.append((
                    -len({str(path["water_id"]) for path in paths}),
                    worst_distance, -best_angle,
                    str(frame["system_id"]), str(frame["replica_id"]),
                    str(frame["segment_id"]), int(frame["source_frame_index"]), frame,
                ))
        if candidates:
            selected = min(candidates)[-1]
            representative_frames.append({
                "bridge_id": bridge_id, "system_id": selected["system_id"],
                "replica_id": selected["replica_id"],
                "segment_id": selected["segment_id"],
                "source_frame_index": selected["source_frame_index"],
                "axis_kind": selected["axis_kind"], "axis_value": selected["axis_value"],
                "water_ids": selected["primary_bridge_water_ids"][bridge_id],
                "selection_rule": "maximum_water_multiplicity_then_best_geometry_then_earliest_identity",
            })
    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": str(source),
            "message": (
                "Water-network analysis evaluated "
                f"{frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "water_mediated_hydrogen_bond_networks",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "frame_selection": frame_selection_report,
        "geometry_contract": {
            "path_length": "exactly one bridging water",
            "edge_definition": "explicit donor-hydrogen-acceptor geometry",
            "periodicity": "exact triclinic minimum-image edge geometry",
            "neighbor_search": "conservative fractional/nonperiodic cell list plus exact distance filter",
        },
        "cutoff_definitions": cutoff_definitions,
        "chemistry_reports": chemistry_reports,
        "endpoint_dictionary": endpoint_dictionary,
        "water_dictionary": water_dictionary,
        "observed_bridge_dictionary": [
            bridge_dictionary[key] for key in sorted(bridge_dictionary)
        ],
        "bridge_dictionary_contract": (
            "Observation-derived union for descriptive occupancy only; it is not a frozen, "
            "outcome-independent feature universe for non-nested machine learning."
        ),
        "evaluated_frame_count": len(frame_records),
        "evaluated_neighbor_pair_count": total_neighbor_pairs,
        "sparse_path_record_count": sparse_record_count,
        "frame_networks": frame_records,
        "bridge_occupancies": occupancies,
        "any_water_bridge_residence_runs": any_water_runs,
        "same_water_bridge_residence_runs": same_water_runs,
        "network_nodes": [
            {"system_id": key[0], "replica_id": key[1], "segment_id": key[2],
             "endpoint_atom_index": key[3], **values}
            for key, values in sorted(node_values.items())
        ],
        "network_edges": network_edges,
        "representative_frames": representative_frames,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Only one-water paths are represented; multi-water wires and proton hopping are not modeled.",
            "Water identity and protonation come from the fixed topology and do not model reactive exchange.",
            "Residence runs are sampled-frame, segment-safe descriptors with explicit boundary censoring; they are not kinetic rates.",
            "The observed bridge dictionary is outcome-derived and cannot be used as a non-nested predictive feature-selection step.",
            "Geometry and occupancy do not establish energetic importance, affinity, causality, or mechanism.",
        ],
    }


def water_mediated_hydrogen_bond_networks_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return water_mediated_hydrogen_bond_networks_project(
            project_path, hash_content=hash_content
        )
    except (
        ManifestValidationError, WaterMediatedHydrogenBondError,
        HydrogenBondDiscoveryError, AtomMappingError, CoordinateReadError,
        PeriodicReconstructionError, TrajectoryContractError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "water_mediated_hydrogen_bond_networks",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "WATER_MEDIATED_HYDROGEN_BOND_INVALID",
                 "message": message}
                for message in messages
            ],
        }
