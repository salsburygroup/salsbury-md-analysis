"""Connectivity-backed hydrogen-bond candidate discovery and frame matrices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .hydrogen_bond_chemistry import (
    AtomChemicalRole,
    chemistry_summary,
    infer_atom_chemical_roles,
    scope_allows,
)
from .hydrogen_bonds import hydrogen_bond_present, distance_angstrom
from .hydrogen_bond_sparse import (
    CompiledSparseHydrogenBondEvaluator,
    LazySpatialHydrogenBondEvaluator,
    PACKED_CUTOFF_COUNT_CODEC,
    PACKED_EVENT_CODEC,
    pack_sparse_cutoff_counts,
    pack_sparse_present_geometry,
    unpack_sparse_present_events,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor, PeriodicReconstructionError, load_connectivity,
)
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    merge_frame_selection_reports,
    restore_source_provenance,
    unique_issues,
)
from .trajectory_contracts import (
    TrajectoryContractError, frame_axis_value, normalize_segment_axis,
)
from .validation import positive_integer


class HydrogenBondDiscoveryError(ValueError):
    """Raised when candidate discovery lacks explicit chemistry/connectivity."""


AtomIdentityKey = Tuple[str, int, str, str, str]
CandidateIdentityKey = Tuple[AtomIdentityKey, AtomIdentityKey, AtomIdentityKey]
CandidateIndexKey = Tuple[int, int, int]
HarmonizedCandidateKey = Union[CandidateIdentityKey, CandidateIndexKey]
DonorHydrogenIdentityKey = Tuple[AtomIdentityKey, AtomIdentityKey, str]
AcceptorIdentityKey = Tuple[AtomIdentityKey, str]


def _atom_identity_key(atom: AtomRecord) -> AtomIdentityKey:
    """Return the position-stable identity used across homologous systems."""

    return (
        atom.chain_id,
        atom.residue_number,
        atom.insertion_code,
        atom.atom_name,
        atom.altloc,
    )


def _candidate_identity_key(
    atoms: Sequence[AtomRecord], triple: Tuple[int, int, int]
) -> CandidateIdentityKey:
    return tuple(_atom_identity_key(atoms[index]) for index in triple)  # type: ignore[return-value]


def _identity_bond_id(key: CandidateIdentityKey) -> str:
    def label(atom: AtomIdentityKey) -> str:
        chain, residue, insertion, name, altloc = atom
        return f"{chain}:{residue}{insertion}:{name}:{altloc}"

    return f"D[{label(key[0])}]-H[{label(key[1])}]-A[{label(key[2])}]"


def discover_candidate_bonds(
    atoms: Sequence[AtomRecord],
    bonds: Sequence[Tuple[int, int]],
    donor_atom_indices: Sequence[int],
    acceptor_atom_indices: Sequence[int],
    *,
    allowed_donor_elements: Sequence[str],
    allowed_acceptor_elements: Sequence[str],
    exclude_same_residue: bool,
) -> List[Dict[str, object]]:
    """Enumerate donor-H-acceptor triples from declared atoms and covalent bonds."""

    atom_count = len(atoms)
    selected = list(donor_atom_indices) + list(acceptor_atom_indices)
    if not donor_atom_indices or not acceptor_atom_indices:
        raise HydrogenBondDiscoveryError("donor and acceptor selections must be nonempty")
    if any(index < 0 or index >= atom_count for index in selected):
        raise HydrogenBondDiscoveryError("donor or acceptor atom index exceeds topology")
    donor_elements = {value.upper() for value in allowed_donor_elements}
    acceptor_elements = {value.upper() for value in allowed_acceptor_elements}
    adjacency: Dict[int, List[int]] = {index: [] for index in range(atom_count)}
    for left, right in bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    candidates = []
    for donor in donor_atom_indices:
        if atoms[donor].element.upper() not in donor_elements:
            raise HydrogenBondDiscoveryError(
                f"declared donor atom {donor} has disallowed element {atoms[donor].element}"
            )
        hydrogens = sorted(
            neighbor for neighbor in adjacency[donor]
            if atoms[neighbor].element.upper() == "H"
        )
        if not hydrogens:
            raise HydrogenBondDiscoveryError(
                f"declared donor atom {donor} has no connectivity-declared hydrogen"
            )
        for hydrogen in hydrogens:
            for acceptor in acceptor_atom_indices:
                if acceptor in {donor, hydrogen}:
                    continue
                if atoms[acceptor].element.upper() not in acceptor_elements:
                    raise HydrogenBondDiscoveryError(
                        f"declared acceptor atom {acceptor} has disallowed element {atoms[acceptor].element}"
                    )
                same_residue = (
                    atoms[donor].chain_id == atoms[acceptor].chain_id
                    and atoms[donor].residue_number == atoms[acceptor].residue_number
                    and atoms[donor].insertion_code == atoms[acceptor].insertion_code
                )
                if exclude_same_residue and same_residue:
                    continue
                bond_id = f"D{donor}-H{hydrogen}-A{acceptor}"
                candidates.append({
                    "bond_id": bond_id,
                    "donor_atom_index": donor,
                    "hydrogen_atom_index": hydrogen,
                    "acceptor_atom_index": acceptor,
                    "donor_identity": atoms[donor].as_dict(),
                    "hydrogen_identity": atoms[hydrogen].as_dict(),
                    "acceptor_identity": atoms[acceptor].as_dict(),
                })
    if not candidates:
        raise HydrogenBondDiscoveryError("declared selections produce no candidate bonds")
    return candidates


def _automatic_candidate_triples(
    atoms: Sequence[AtomRecord],
    bonds: Sequence[Tuple[int, int]],
    *,
    interaction_scope: str,
    exclude_same_residue: bool,
) -> Tuple[List[Tuple[int, int, int]], Dict[int, AtomChemicalRole]]:
    """Enumerate automatic candidate indices without nested output records."""

    roles = infer_atom_chemical_roles(atoms, bonds)
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(atoms))}
    for left, right in bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    acceptors = [
        (index, role) for index, role in sorted(roles.items()) if role.acceptor
    ]
    triples: List[Tuple[int, int, int]] = []
    for donor, donor_role in sorted(roles.items()):
        if not donor_role.donor:
            continue
        hydrogens = sorted(
            neighbor for neighbor in adjacency[donor]
            if atoms[neighbor].element.upper() == "H"
        )
        for hydrogen in hydrogens:
            for acceptor, acceptor_role in acceptors:
                if acceptor in {donor, hydrogen}:
                    continue
                if not scope_allows(
                    donor_role.entity_class, acceptor_role.entity_class,
                    interaction_scope,
                ):
                    continue
                same_residue = (
                    atoms[donor].chain_id == atoms[acceptor].chain_id
                    and atoms[donor].residue_number == atoms[acceptor].residue_number
                    and atoms[donor].insertion_code == atoms[acceptor].insertion_code
                )
                if exclude_same_residue and same_residue:
                    continue
                triples.append((donor, hydrogen, acceptor))
    if not triples:
        raise HydrogenBondDiscoveryError(
            "automatic chemistry produced no direct hydrogen-bond candidates; "
            "check explicit hydrogens, connectivity, and declared interaction_scope"
        )
    return triples, roles


def _automatic_endpoint_identity_sets(
    atoms: Sequence[AtomRecord],
    bonds: Sequence[Tuple[int, int]],
) -> Tuple[
    set[DonorHydrogenIdentityKey],
    set[AcceptorIdentityKey],
    Dict[int, AtomChemicalRole],
]:
    """Return compact donor-H and acceptor identities without a Cartesian product."""

    roles = infer_atom_chemical_roles(atoms, bonds)
    adjacency: Dict[int, List[int]] = {index: [] for index in range(len(atoms))}
    for left, right in bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    donors: set[DonorHydrogenIdentityKey] = set()
    acceptors: set[AcceptorIdentityKey] = set()
    for atom_index, role in sorted(roles.items()):
        if role.donor:
            for hydrogen in sorted(adjacency[atom_index]):
                if atoms[hydrogen].element.upper() == "H":
                    donors.add((
                        _atom_identity_key(atoms[atom_index]),
                        _atom_identity_key(atoms[hydrogen]),
                        role.entity_class,
                    ))
        if role.acceptor:
            acceptors.add((_atom_identity_key(atoms[atom_index]), role.entity_class))
    if not donors or not acceptors:
        raise HydrogenBondDiscoveryError(
            "automatic chemistry produced no donor-H groups or acceptors"
        )
    return donors, acceptors, roles


def _conceptual_endpoint_candidate_count(
    donors: Sequence[DonorHydrogenIdentityKey],
    acceptors: Sequence[AcceptorIdentityKey],
    *,
    interaction_scope: str,
    exclude_same_residue: bool,
) -> int:
    """Count the implicit candidate universe without materializing its pairs."""

    count = 0
    for donor, hydrogen, donor_class in donors:
        for acceptor, acceptor_class in acceptors:
            if donor == acceptor or hydrogen == acceptor:
                continue
            if not scope_allows(donor_class, acceptor_class, interaction_scope):
                continue
            if (
                exclude_same_residue
                and donor[:3] == acceptor[:3]
            ):
                continue
            count += 1
    return count


def _automatic_endpoint_identity_intersection(
    system: Mapping[str, object],
    system_path: Path,
    settings: Mapping[str, object],
) -> Tuple[
    set[DonorHydrogenIdentityKey],
    set[AcceptorIdentityKey],
    Dict[str, object],
]:
    """Intersect compact eligible endpoints across homologous replicas."""

    common_donors: set[DonorHydrogenIdentityKey] | None = None
    common_acceptors: set[AcceptorIdentityKey] | None = None
    reports: List[Dict[str, object]] = []
    topology_cache: Dict[
        Tuple[str, str],
        Tuple[
            frozenset[DonorHydrogenIdentityKey],
            frozenset[AcceptorIdentityKey],
            int,
        ],
    ] = {}
    for raw_system in system["systems"]:  # type: ignore[index]
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} requires explicit connectivity"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            cache_key = (str(topology_path), str(connectivity_path))
            cached = topology_cache.get(cache_key)
            if cached is None:
                _, atoms = read_topology_atoms(topology_path)
                bonds, _ = load_connectivity(connectivity_path, len(atoms))
                donors, acceptors, _ = _automatic_endpoint_identity_sets(atoms, bonds)
                raw_count = _conceptual_endpoint_candidate_count(
                    sorted(donors), sorted(acceptors),
                    interaction_scope=str(settings["interaction_scope"]),
                    exclude_same_residue=bool(settings["exclude_same_residue"]),
                )
                topology_cache[cache_key] = (
                    frozenset(donors), frozenset(acceptors), raw_count,
                )
            else:
                donors = set(cached[0])
                acceptors = set(cached[1])
                raw_count = cached[2]
            reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "raw_donor_hydrogen_group_count": len(donors),
                "raw_acceptor_count": len(acceptors),
                "raw_conceptual_candidate_count": raw_count,
            })
            if common_donors is None:
                common_donors = donors
                common_acceptors = acceptors
            else:
                common_donors.intersection_update(donors)
                assert common_acceptors is not None
                common_acceptors.intersection_update(acceptors)
    if not common_donors or not common_acceptors:
        raise HydrogenBondDiscoveryError(
            "automatic endpoint dictionaries have no common identity intersection"
        )
    conceptual_count = _conceptual_endpoint_candidate_count(
        sorted(common_donors), sorted(common_acceptors),
        interaction_scope=str(settings["interaction_scope"]),
        exclude_same_residue=bool(settings["exclude_same_residue"]),
    )
    if not conceptual_count:
        raise HydrogenBondDiscoveryError("common endpoint universe has no candidates")
    for report in reports:
        report["common_donor_hydrogen_group_count"] = len(common_donors)
        report["common_acceptor_count"] = len(common_acceptors)
        report["common_conceptual_candidate_count"] = conceptual_count
    return common_donors, common_acceptors, {
        "policy": "intersection_by_endpoint_identity_lazy_v3",
        "identity_fields": [
            "chain_id", "residue_number", "insertion_code", "atom_name", "altloc",
        ],
        "residue_name_policy": "ignored_for_homologous_position_mapping",
        "common_donor_hydrogen_group_count": len(common_donors),
        "common_acceptor_count": len(common_acceptors),
        "common_candidate_count": conceptual_count,
        "materialized_precoordinate_candidate_count": 0,
        "replica_endpoint_dictionaries": reports,
        "selection_basis": (
            "common topology-backed donor-H and acceptor endpoint identities; "
            "their Cartesian product remains implicit and coordinates are not read"
        ),
    }


def discover_automatic_candidate_bonds(
    atoms: Sequence[AtomRecord],
    bonds: Sequence[Tuple[int, int]],
    *,
    interaction_scope: str,
    exclude_same_residue: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Discover direct H bonds from topology-backed chemical identity.

    The caller supplies no atom index selections.  Protein and nucleic-acid
    roles are determined by templates; untemplated residues retain explicit
    provisional chemistry provenance in the output dictionary.
    """

    triples, roles = _automatic_candidate_triples(
        atoms, bonds,
        interaction_scope=interaction_scope,
        exclude_same_residue=exclude_same_residue,
    )
    candidates = [{
        "bond_id": f"D{donor}-H{hydrogen}-A{acceptor}",
        "donor_atom_index": donor,
        "hydrogen_atom_index": hydrogen,
        "acceptor_atom_index": acceptor,
        "donor_identity": atoms[donor].as_dict(),
        "hydrogen_identity": atoms[hydrogen].as_dict(),
        "acceptor_identity": atoms[acceptor].as_dict(),
        "donor_chemistry": roles[donor].as_dict(),
        "acceptor_chemistry": roles[acceptor].as_dict(),
        "interaction_stratum": (
            f"{roles[donor].entity_class}_to_{roles[acceptor].entity_class}"
        ),
    } for donor, hydrogen, acceptor in triples]
    return candidates, chemistry_summary(roles)


def _index_array(value: object, label: str) -> List[int]:
    if (
        not isinstance(value, list) or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        or len(set(value)) != len(value)
    ):
        raise HydrogenBondDiscoveryError(f"{label} must contain unique nonnegative integers")
    return list(value)


def _element_array(value: object, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise HydrogenBondDiscoveryError(f"{label} must be a nonempty element array")
    normalized = [str(item).strip().upper() for item in value]
    if any(not item or not item.isalpha() for item in normalized) or len(set(normalized)) != len(normalized):
        raise HydrogenBondDiscoveryError(f"{label} must contain unique element symbols")
    return normalized


def _positive_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0.0
    ):
        raise HydrogenBondDiscoveryError(f"{label} must be finite and positive")
    return float(value)


def _angle(value: object, label: str) -> float:
    result = _positive_float(value, label)
    if result >= 180.0:
        raise HydrogenBondDiscoveryError(f"{label} must be below 180 degrees")
    return result


def _cutoff_definitions(policy: object) -> List[Dict[str, object]]:
    """Return a primary rule and an outcome-independent sensitivity grid."""

    if not isinstance(policy, dict):
        raise HydrogenBondDiscoveryError("cutoff_policy must be an object")
    preset = policy.get("preset")
    if preset == "mdanalysis_compatible_v1":
        if set(policy) != {"preset"}:
            raise HydrogenBondDiscoveryError(
                "mdanalysis_compatible_v1 cutoff_policy must contain only preset"
            )
        primary_distance, primary_angle = 3.0, 150.0
        distances, angles = (3.0, 3.2, 3.5), (120.0, 135.0, 150.0)
    elif preset == "custom_v1":
        required = {
            "preset", "maximum_donor_acceptor_distance_angstrom",
            "minimum_donor_hydrogen_acceptor_angle_degrees",
            "distance_sensitivity_angstrom", "angle_sensitivity_degrees",
        }
        if set(policy) != required:
            raise HydrogenBondDiscoveryError(
                "custom_v1 cutoff_policy requires primary distance/angle and both sensitivity arrays"
            )
        primary_distance = _positive_float(
            policy["maximum_donor_acceptor_distance_angstrom"],
            "cutoff_policy.maximum_donor_acceptor_distance_angstrom",
        )
        primary_angle = _angle(
            policy["minimum_donor_hydrogen_acceptor_angle_degrees"],
            "cutoff_policy.minimum_donor_hydrogen_acceptor_angle_degrees",
        )
        raw_distances = policy["distance_sensitivity_angstrom"]
        raw_angles = policy["angle_sensitivity_degrees"]
        if not isinstance(raw_distances, list) or not isinstance(raw_angles, list):
            raise HydrogenBondDiscoveryError("custom_v1 cutoff sensitivity values must be arrays")
        distances = tuple(sorted({_positive_float(value, "cutoff_policy.distance_sensitivity_angstrom") for value in raw_distances}))
        angles = tuple(sorted({_angle(value, "cutoff_policy.angle_sensitivity_degrees") for value in raw_angles}))
        if not distances or not angles:
            raise HydrogenBondDiscoveryError("custom_v1 cutoff sensitivity arrays must be nonempty")
    else:
        raise HydrogenBondDiscoveryError(
            "cutoff_policy.preset must be mdanalysis_compatible_v1 or custom_v1"
        )
    sensitivity = [
        (distance, angle) for distance in distances for angle in angles
        if (distance, angle) != (primary_distance, primary_angle)
    ]
    if len(sensitivity) > 24:
        raise HydrogenBondDiscoveryError("cutoff sensitivity grid exceeds the 24-combination resource gate")
    result = [{
        "cutoff_id": "primary",
        "kind": "primary",
        "maximum_donor_acceptor_distance_angstrom": primary_distance,
        "minimum_donor_hydrogen_acceptor_angle_degrees": primary_angle,
    }]
    result.extend({
        "cutoff_id": f"sensitivity_da{distance:g}_angle{angle:g}",
        "kind": "sensitivity",
        "maximum_donor_acceptor_distance_angstrom": distance,
        "minimum_donor_hydrogen_acceptor_angle_degrees": angle,
    } for distance, angle in sensitivity)
    return result


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("hydrogen_bond_discovery") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict) or "chemistry_policy" not in raw:
        raise HydrogenBondDiscoveryError(
            "definitions.hydrogen_bond_discovery requires a chemistry_policy"
        )
    common = {
        "chemistry_policy", "exclude_same_residue", "water_policy", "frame_stride",
        "maximum_reference_donor_hydrogen_bond_angstrom", "maximum_candidate_bonds",
        "maximum_feature_observations",
    }
    optional = {
        "output_mode", "candidate_chunk_size", "frame_selection",
        "candidate_harmonization",
    }
    if raw["chemistry_policy"] == "explicit_atoms_connectivity_v1":
        required = common | {
            "donor_atom_indices", "acceptor_atom_indices", "allowed_donor_elements",
            "allowed_acceptor_elements", "maximum_donor_acceptor_distance_angstrom",
            "minimum_donor_hydrogen_acceptor_angle_degrees",
        }
        if set(raw) - optional != required or set(raw) - required - optional:
            raise HydrogenBondDiscoveryError(
                "explicit_atoms_connectivity_v1 fields do not match the legacy-compatible contract"
            )
        cutoffs = [{
            "cutoff_id": "primary", "kind": "primary",
            "maximum_donor_acceptor_distance_angstrom": _positive_float(raw["maximum_donor_acceptor_distance_angstrom"], "maximum_donor_acceptor_distance_angstrom"),
            "minimum_donor_hydrogen_acceptor_angle_degrees": _angle(raw["minimum_donor_hydrogen_acceptor_angle_degrees"], "minimum_donor_hydrogen_acceptor_angle_degrees"),
        }]
        result: Dict[str, object] = {
            "mode": "explicit", "chemistry_policy": raw["chemistry_policy"],
            "donor_atom_indices": _index_array(raw["donor_atom_indices"], "donor_atom_indices"),
            "acceptor_atom_indices": _index_array(raw["acceptor_atom_indices"], "acceptor_atom_indices"),
            "allowed_donor_elements": _element_array(raw["allowed_donor_elements"], "allowed_donor_elements"),
            "allowed_acceptor_elements": _element_array(raw["allowed_acceptor_elements"], "allowed_acceptor_elements"),
            "cutoff_definitions": cutoffs,
            "candidate_harmonization": "strict_v1",
        }
        if raw.get("candidate_harmonization", "strict_v1") != "strict_v1":
            raise HydrogenBondDiscoveryError(
                "explicit atom chemistry supports only candidate_harmonization=strict_v1"
            )
    elif raw["chemistry_policy"] == "automatic_topology_templates_v1":
        required = common | {"interaction_scope", "cutoff_policy"}
        if set(raw) - optional != required or set(raw) - required - optional:
            raise HydrogenBondDiscoveryError(
                "automatic_topology_templates_v1 fields do not match the contract"
            )
        scope = raw["interaction_scope"]
        allowed_scopes = {
            "all_solute", "protein_protein", "protein_ligand", "protein_nucleic_acid",
            "nucleic_acid_nucleic_acid", "nucleic_acid_ligand", "ligand_ligand",
        }
        if scope not in allowed_scopes:
            raise HydrogenBondDiscoveryError(
                "interaction_scope must be one of: " + ", ".join(sorted(allowed_scopes))
            )
        result = {
            "mode": "automatic", "chemistry_policy": raw["chemistry_policy"],
            "interaction_scope": str(scope),
            "cutoff_definitions": _cutoff_definitions(raw["cutoff_policy"]),
        }
        # Generic multi-system and mixed protein--DNA campaigns commonly have
        # valid system-specific candidates. Use an outcome-independent common
        # chemical-position intersection unless a publication lock explicitly
        # asks for exact dictionary equality. Raw atom indices are not stable
        # when variants add or remove atoms.
        candidate_harmonization = raw.get(
            "candidate_harmonization", "intersection_by_atom_identity_v2"
        )
        if candidate_harmonization not in {
            "strict_v1", "intersection_by_atom_index_v1",
            "intersection_by_atom_identity_v2",
        }:
            raise HydrogenBondDiscoveryError(
                "candidate_harmonization must be strict_v1, "
                "intersection_by_atom_index_v1, or "
                "intersection_by_atom_identity_v2"
            )
        result["candidate_harmonization"] = candidate_harmonization
    else:
        raise HydrogenBondDiscoveryError(
            "chemistry_policy must be explicit_atoms_connectivity_v1 or automatic_topology_templates_v1"
        )
    if raw["water_policy"] != "exclude":
        raise HydrogenBondDiscoveryError("water_policy currently supports only exclude")
    if not isinstance(raw["exclude_same_residue"], bool):
        raise HydrogenBondDiscoveryError("exclude_same_residue must be boolean")
    output_mode = raw.get("output_mode", "dense_v1")
    if output_mode not in {
        "dense_v1", "sparse_implicit_zero_v1", "sparse_packed_v2",
        "sparse_spatial_observed_union_v3",
    }:
        raise HydrogenBondDiscoveryError(
            "output_mode must be dense_v1, sparse_implicit_zero_v1, "
            "sparse_packed_v2, or sparse_spatial_observed_union_v3"
        )
    result.update({
        "exclude_same_residue": raw["exclude_same_residue"], "water_policy": "exclude",
        "frame_stride": positive_integer(raw["frame_stride"], "frame_stride", error_type=HydrogenBondDiscoveryError),
        "maximum_reference_donor_hydrogen_bond_angstrom": _positive_float(raw["maximum_reference_donor_hydrogen_bond_angstrom"], "maximum_reference_donor_hydrogen_bond_angstrom"),
        "maximum_candidate_bonds": positive_integer(raw["maximum_candidate_bonds"], "maximum_candidate_bonds", error_type=HydrogenBondDiscoveryError),
        "maximum_feature_observations": positive_integer(raw["maximum_feature_observations"], "maximum_feature_observations", error_type=HydrogenBondDiscoveryError),
        "output_mode": output_mode,
        "candidate_chunk_size": positive_integer(
            raw.get("candidate_chunk_size", 4096),
            "candidate_chunk_size",
            error_type=HydrogenBondDiscoveryError,
        ),
    })
    result["frame_selection"] = normalize_frame_selection(
        raw.get("frame_selection"), int(result["frame_stride"]),
        error_type=HydrogenBondDiscoveryError,
    )
    primary = result["cutoff_definitions"][0]  # type: ignore[index]
    result["maximum_donor_acceptor_distance_angstrom"] = primary["maximum_donor_acceptor_distance_angstrom"]  # type: ignore[index]
    result["minimum_donor_hydrogen_acceptor_angle_degrees"] = primary["minimum_donor_hydrogen_acceptor_angle_degrees"]  # type: ignore[index]
    return result


def _candidate_key(candidate: Mapping[str, object]) -> Tuple[int, int, int]:
    return (
        int(candidate["donor_atom_index"]),
        int(candidate["hydrogen_atom_index"]),
        int(candidate["acceptor_atom_index"]),
    )


def _automatic_candidate_intersection(
    system: Mapping[str, object], system_path: Path, settings: Mapping[str, object],
) -> Tuple[set[Tuple[int, int, int]], Dict[str, object]]:
    """Build an outcome-independent common feature universe for homologous systems."""

    reports = []
    union: set[Tuple[int, int, int]] = set()
    common: set[Tuple[int, int, int]] | None = None
    for raw_system in system["systems"]:  # type: ignore[index]
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} requires explicit connectivity for donor-H discovery"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            bonds, _ = load_connectivity(connectivity_path, len(atoms))
            triples, _ = _automatic_candidate_triples(
                atoms, bonds,
                interaction_scope=str(settings["interaction_scope"]),
                exclude_same_residue=bool(settings["exclude_same_residue"]),
            )
            keys = set(triples)
            union.update(keys)
            if common is None:
                common = keys
            else:
                common.intersection_update(keys)
            reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "raw_candidate_count": len(keys),
            })
            del triples, keys
    if common is None:
        raise HydrogenBondDiscoveryError("system manifest contains no replicas")
    if not common:
        raise HydrogenBondDiscoveryError(
            "automatic candidate dictionaries have no common atom-index intersection"
        )
    for report in reports:
        report["common_candidate_count"] = len(common)
        report["excluded_noncommon_candidate_count"] = (
            int(report["raw_candidate_count"]) - len(common)
        )
    return common, {
        "policy": "intersection_by_atom_index_v1",
        "common_candidate_count": len(common),
        "union_candidate_count": len(union),
        "excluded_from_common_union_count": len(union - common),
        "replica_dictionaries": reports,
        "selection_basis": (
            "candidate atom-index triples present in every replica before any "
            "trajectory coordinates or occupancies are evaluated"
        ),
    }


def _automatic_candidate_identity_intersection(
    system: Mapping[str, object], system_path: Path, settings: Mapping[str, object],
) -> Tuple[set[CandidateIdentityKey], Dict[str, object]]:
    """Build a common universe using stable chain/residue/atom identities.

    Residue name is deliberately omitted, matching the suite's ``position``
    common-atom policy, so chemically corresponding atoms in variants such as
    THY and EdU can be compared. Atom name, chain, residue number, insertion
    code, and altloc must still match exactly and uniquely.
    """

    reports = []
    union: set[CandidateIdentityKey] = set()
    common: set[CandidateIdentityKey] | None = None
    for raw_system in system["systems"]:  # type: ignore[index]
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} requires explicit connectivity for donor-H discovery"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            bonds, _ = load_connectivity(connectivity_path, len(atoms))
            triples, _ = _automatic_candidate_triples(
                atoms, bonds,
                interaction_scope=str(settings["interaction_scope"]),
                exclude_same_residue=bool(settings["exclude_same_residue"]),
            )
            keys = {_candidate_identity_key(atoms, triple) for triple in triples}
            if len(keys) != len(triples):
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} has ambiguous duplicate candidate atom identities"
                )
            union.update(keys)
            if common is None:
                common = keys
            else:
                common.intersection_update(keys)
            reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "raw_candidate_count": len(keys),
            })
            del triples, keys
    if common is None:
        raise HydrogenBondDiscoveryError("system manifest contains no replicas")
    if not common:
        raise HydrogenBondDiscoveryError(
            "automatic candidate dictionaries have no common atom-identity intersection"
        )
    for report in reports:
        report["common_candidate_count"] = len(common)
        report["excluded_noncommon_candidate_count"] = (
            int(report["raw_candidate_count"]) - len(common)
        )
    return common, {
        "policy": "intersection_by_atom_identity_v2",
        "identity_fields": [
            "chain_id", "residue_number", "insertion_code", "atom_name", "altloc",
        ],
        "residue_name_policy": "ignored_for_homologous_position_mapping",
        "common_candidate_count": len(common),
        "union_candidate_count": len(union),
        "excluded_from_common_union_count": len(union - common),
        "replica_dictionaries": reports,
        "selection_basis": (
            "candidate chemical-position triples present in every replica before "
            "trajectory coordinates or occupancies are evaluated"
        ),
    }


def _automatic_candidate_harmonization(
    system: Mapping[str, object], system_path: Path, settings: Mapping[str, object],
) -> Tuple[set[HarmonizedCandidateKey], Dict[str, object]]:
    policy = str(settings["candidate_harmonization"])
    if policy == "intersection_by_atom_identity_v2":
        keys, report = _automatic_candidate_identity_intersection(
            system, system_path, settings
        )
        return set(keys), report
    if policy == "intersection_by_atom_index_v1":
        keys, report = _automatic_candidate_intersection(system, system_path, settings)
        return set(keys), report
    raise HydrogenBondDiscoveryError(
        "automatic candidate harmonization requires an intersection policy"
    )


def _sparse_atom_dictionary(
    candidates: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Deduplicate candidate atom identity and chemistry for sparse reports."""

    atoms: Dict[int, Dict[str, object]] = {}
    for candidate in candidates:
        for role_name in ("donor", "hydrogen", "acceptor"):
            atom_index = int(candidate[f"{role_name}_atom_index"])
            entry = atoms.setdefault(atom_index, {
                "atom_index": atom_index,
                "identity": candidate[f"{role_name}_identity"],
            })
            chemistry_key = f"{role_name}_chemistry"
            if chemistry_key in candidate:
                prior = entry.get("hydrogen_bond_chemistry")
                if prior is not None and prior != candidate[chemistry_key]:
                    raise HydrogenBondDiscoveryError(
                        f"atom {atom_index} has inconsistent sparse chemistry records"
                    )
                entry["hydrogen_bond_chemistry"] = candidate[chemistry_key]
    return [atoms[index] for index in sorted(atoms)]


def _automatic_sparse_atom_dictionary(
    atoms: Sequence[AtomRecord],
    roles: Mapping[int, AtomChemicalRole],
    triples: Sequence[Tuple[int, int, int]],
) -> List[Dict[str, object]]:
    """Build sparse atom chemistry once without per-candidate nesting."""

    used: Dict[int, Dict[str, object]] = {}
    for donor, hydrogen, acceptor in triples:
        for atom_index in (donor, hydrogen, acceptor):
            used.setdefault(atom_index, {
                "atom_index": atom_index,
                "identity": atoms[atom_index].as_dict(),
            })
        for atom_index in (donor, acceptor):
            chemistry = roles[atom_index].as_dict()
            prior = used[atom_index].get("hydrogen_bond_chemistry")
            if prior is not None and prior != chemistry:
                raise HydrogenBondDiscoveryError(
                    f"atom {atom_index} has inconsistent automatic chemistry"
                )
            used[atom_index]["hydrogen_bond_chemistry"] = chemistry
    return [used[index] for index in sorted(used)]


def _identity_harmonized_candidate_dictionaries(
    atoms: Sequence[AtomRecord],
    roles: Mapping[int, AtomChemicalRole],
    triples: Sequence[Tuple[int, int, int]],
    harmonized_candidate_keys: set[HarmonizedCandidateKey],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Map stable cross-system identities to replica-local evaluator indices.

    The local dictionary is used only to address coordinates in this replica.
    The canonical dictionaries use synthetic stable indices so reducer equality
    remains meaningful when a variant inserts or removes topology atoms.
    """

    keys = sorted(harmonized_candidate_keys)
    if keys and not isinstance(keys[0][0], tuple):
        raise HydrogenBondDiscoveryError(
            "atom-identity harmonization received raw atom-index keys"
        )
    identity_keys = [key for key in keys]  # type: ignore[misc]
    triple_by_key: Dict[CandidateIdentityKey, Tuple[int, int, int]] = {}
    for triple in triples:
        key = _candidate_identity_key(atoms, triple)
        if key in triple_by_key:
            raise HydrogenBondDiscoveryError(
                "replica has ambiguous duplicate candidate atom identities"
            )
        triple_by_key[key] = triple
    missing = [key for key in identity_keys if key not in triple_by_key]
    if missing:
        raise HydrogenBondDiscoveryError(
            "replica lacks a globally harmonized candidate atom identity"
        )

    atom_keys = sorted({atom_key for key in identity_keys for atom_key in key})
    stable_atom_index = {key: index for index, key in enumerate(atom_keys)}

    def identity_record(key: AtomIdentityKey) -> Dict[str, object]:
        return {
            "chain_id": key[0],
            "residue_number": key[1],
            "insertion_code": key[2],
            "atom_name": key[3],
            "altloc": key[4],
        }

    local_candidates: List[Dict[str, object]] = []
    canonical_candidates: List[Dict[str, object]] = []
    for key in identity_keys:
        donor, hydrogen, acceptor = triple_by_key[key]
        bond_id = _identity_bond_id(key)
        local_candidates.append({
            "bond_id": bond_id,
            "donor_atom_index": donor,
            "hydrogen_atom_index": hydrogen,
            "acceptor_atom_index": acceptor,
            "interaction_stratum": (
                f"{roles[donor].entity_class}_to_{roles[acceptor].entity_class}"
            ),
        })
        canonical_candidates.append({
            "bond_id": bond_id,
            "donor_atom_index": stable_atom_index[key[0]],
            "hydrogen_atom_index": stable_atom_index[key[1]],
            "acceptor_atom_index": stable_atom_index[key[2]],
            "donor_identity": identity_record(key[0]),
            "hydrogen_identity": identity_record(key[1]),
            "acceptor_identity": identity_record(key[2]),
            "interaction_stratum": (
                f"{roles[donor].entity_class}_to_{roles[acceptor].entity_class}"
            ),
        })
    canonical_atoms = [{
        "atom_index": stable_atom_index[key],
        "identity": identity_record(key),
    } for key in atom_keys]
    return local_candidates, canonical_candidates, canonical_atoms


def _identity_record(key: AtomIdentityKey) -> Dict[str, object]:
    return {
        "chain_id": key[0],
        "residue_number": key[1],
        "insertion_code": key[2],
        "atom_name": key[3],
        "altloc": key[4],
    }


def _lazy_local_endpoint_rows(
    atoms: Sequence[AtomRecord],
    bonds: Sequence[Tuple[int, int]],
    common_donors: set[DonorHydrogenIdentityKey],
    common_acceptors: set[AcceptorIdentityKey],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[int, AtomChemicalRole]]:
    """Resolve common endpoint identities to one replica's local atom indices."""

    needed_identities = {
        identity
        for donor, hydrogen, _ in common_donors
        for identity in (donor, hydrogen)
    } | {identity for identity, _ in common_acceptors}
    identity_to_index: Dict[AtomIdentityKey, int] = {}
    for atom in atoms:
        key = _atom_identity_key(atom)
        if key not in needed_identities:
            continue
        if key in identity_to_index:
            raise HydrogenBondDiscoveryError(
                "topology contains duplicate identities among retained solute endpoints"
            )
        identity_to_index[key] = atom.atom_index
    _, _, roles = _automatic_endpoint_identity_sets(atoms, bonds)
    bond_set = {tuple(sorted((left, right))) for left, right in bonds}
    donor_rows = []
    for donor_key, hydrogen_key, entity_class in sorted(common_donors):
        donor = identity_to_index.get(donor_key)
        hydrogen = identity_to_index.get(hydrogen_key)
        if donor is None or hydrogen is None:
            raise HydrogenBondDiscoveryError("replica lacks a common donor-H endpoint")
        if tuple(sorted((donor, hydrogen))) not in bond_set:
            raise HydrogenBondDiscoveryError("common donor-H endpoint lacks connectivity")
        role = roles.get(donor)
        if role is None or not role.donor or role.entity_class != entity_class:
            raise HydrogenBondDiscoveryError("common donor endpoint chemistry differs")
        donor_rows.append({
            "donor_atom_index": donor,
            "hydrogen_atom_index": hydrogen,
            "donor_identity_key": donor_key,
            "hydrogen_identity_key": hydrogen_key,
            "entity_class": entity_class,
            "residue_key": donor_key[:3],
        })
    acceptor_rows = []
    for acceptor_key, entity_class in sorted(common_acceptors):
        acceptor = identity_to_index.get(acceptor_key)
        if acceptor is None:
            raise HydrogenBondDiscoveryError("replica lacks a common acceptor endpoint")
        role = roles.get(acceptor)
        if role is None or not role.acceptor or role.entity_class != entity_class:
            raise HydrogenBondDiscoveryError("common acceptor endpoint chemistry differs")
        acceptor_rows.append({
            "acceptor_atom_index": acceptor,
            "acceptor_identity_key": acceptor_key,
            "entity_class": entity_class,
            "residue_key": acceptor_key[:3],
        })
    return donor_rows, acceptor_rows, roles


def _hydrogen_bond_discovery_project_lazy_partial(
    source: Path,
    project: Mapping[str, object],
    settings: Mapping[str, object],
    context: Mapping[str, object],
    system: Mapping[str, object],
    system_path: Path,
    frame_selection_plan: Mapping[Tuple[str, str, str], set[int]],
    frame_selection_report: Mapping[str, object],
    issues: List[Dict[str, object]],
    common_donors: set[DonorHydrogenIdentityKey],
    common_acceptors: set[AcceptorIdentityKey],
    endpoint_report: Mapping[str, object],
    *,
    hash_content: bool,
) -> Dict[str, object]:
    """Evaluate a lazy endpoint universe and return one reducible sparse partial."""

    conceptual_count = int(endpoint_report["common_candidate_count"])
    selected_count = int(frame_selection_report["selected_frame_count"])
    conceptual_observations = conceptual_count * selected_count
    if conceptual_count > int(settings["maximum_candidate_bonds"]):
        raise HydrogenBondDiscoveryError("maximum_candidate_bonds gate exceeded")
    if conceptual_observations > int(settings["maximum_feature_observations"]):
        raise HydrogenBondDiscoveryError(
            "maximum_feature_observations conceptual-universe gate exceeded"
        )

    cutoff_definitions = settings["cutoff_definitions"]
    candidate_index_by_key: Dict[CandidateIdentityKey, int] = {}
    candidate_dictionary: List[Dict[str, object]] = []
    frame_records: List[Dict[str, object]] = []
    chemistry_reports: List[Dict[str, object]] = []
    spatial_pairs = 0
    explicit_geometry = 0
    present_events = 0
    evaluated_frames = 0
    output_time_unit = project.get("time_unit")
    coordinate_unit = str(project["coordinate_unit"])

    for raw_system in system["systems"]:  # type: ignore[index]
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} requires explicit connectivity"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            bonds, connectivity_provenance = load_connectivity(
                connectivity_path, len(atoms)
            )
            donor_rows, acceptor_rows, roles = _lazy_local_endpoint_rows(
                atoms, bonds, common_donors, common_acceptors
            )
            chemistry_reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                **chemistry_summary(roles),
                "common_donor_hydrogen_group_count": len(donor_rows),
                "common_acceptor_count": len(acceptor_rows),
            })
            evaluator = LazySpatialHydrogenBondEvaluator(
                donor_rows,
                acceptor_rows,
                cutoff_definitions,  # type: ignore[arg-type]
                str(settings["interaction_scope"]),
                bool(settings["exclude_same_residue"]),
                int(settings["maximum_candidate_bonds"]),
            )
            reference = next(iter_coordinate_frames(topology_path, coordinate_unit))
            for row in donor_rows:
                donor = int(row["donor_atom_index"])
                hydrogen = int(row["hydrogen_atom_index"])
                if distance_angstrom(
                    reference.coordinates_angstrom[donor],
                    reference.coordinates_angstrom[hydrogen],
                    reference.cell_vectors_angstrom,
                ) > float(settings["maximum_reference_donor_hydrogen_bond_angstrom"]):
                    raise HydrogenBondDiscoveryError(
                        "reference donor-H distance exceeds gate"
                    )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                axis = normalize_segment_axis(
                    segment, str(output_time_unit) if output_time_unit else None
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    selected_indices,
                ):
                    selected = frame_selected(
                        raw_frame.frame_index,
                        selected_indices,
                        int(settings["frame_stride"]),
                    )
                    if not selected:
                        continue
                    frame = raw_frame
                    evaluated = evaluator.evaluate(
                        frame.coordinates_angstrom,
                        cell=frame.cell_vectors_angstrom,
                    )
                    spatial_pairs += int(evaluated["spatial_neighbor_pair_count"])
                    explicit_geometry += int(
                        evaluated["explicit_geometry_evaluation_count"]
                    )
                    geometry_rows = []
                    for event in evaluated["present_events"]:
                        key: CandidateIdentityKey = (
                            tuple(event["donor_identity_key"]),  # type: ignore[arg-type]
                            tuple(event["hydrogen_identity_key"]),  # type: ignore[arg-type]
                            tuple(event["acceptor_identity_key"]),  # type: ignore[arg-type]
                        )
                        candidate_index = candidate_index_by_key.get(key)
                        if candidate_index is None:
                            candidate_index = len(candidate_dictionary)
                            candidate_index_by_key[key] = candidate_index
                            candidate_dictionary.append({
                                "bond_id": _identity_bond_id(key),
                                "donor_identity": _identity_record(key[0]),
                                "hydrogen_identity": _identity_record(key[1]),
                                "acceptor_identity": _identity_record(key[2]),
                                "interaction_stratum": event["interaction_stratum"],
                            })
                        geometry_rows.append({
                            "candidate_index": candidate_index,
                            "donor_acceptor_distance_angstrom": event[
                                "donor_acceptor_distance_angstrom"
                            ],
                            "donor_hydrogen_acceptor_angle_degrees": event[
                                "donor_hydrogen_acceptor_angle_degrees"
                            ],
                            "present_cutoff_ids": event["present_cutoff_ids"],
                        })
                    geometry_rows.sort(key=lambda row: int(row["candidate_index"]))
                    present_events += len(geometry_rows)
                    frame_records.append({
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "candidate_count": len(candidate_dictionary),
                        **pack_sparse_present_geometry(
                            geometry_rows,
                            cutoff_definitions,  # type: ignore[arg-type]
                            max(1, len(candidate_dictionary)),
                        ),
                    })
                    evaluated_frames += 1
            provisional = int(
                chemistry_summary(roles)["chemistry_confidence_atom_counts"].get(
                    "provisional", 0
                )
            )
            issues.append({
                "severity": "warning" if provisional else "info",
                "code": (
                    "HBOND_AUTO_CHEMISTRY_PROVISIONAL"
                    if provisional else "HBOND_AUTO_CHEMISTRY_TEMPLATED"
                ),
                "location": f"{system_id}/{replica_id}",
                "message": (
                    f"Automatic chemistry has {provisional} provisional atoms."
                    if provisional else
                    "Automatic topology-template chemistry used standard templates."
                ),
                "connectivity": connectivity_provenance,
            })
    if evaluated_frames != selected_count:
        raise HydrogenBondDiscoveryError("selected/evaluated frame accounting mismatch")
    return {
        "module_id": "hydrogen_bond_discovery",
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
        "frame_selection": frame_selection_report,
        "geometry_contract": {
            "distance": "donor-acceptor minimum-image distance",
            "angle": "donor-hydrogen-acceptor angle at hydrogen",
            "distance_definition": "donor_acceptor_v1",
            "coordinate_reconstruction": "none_selected_frames_evaluated_raw_wrapped",
            "project_periodic_coordinate_policy": project[
                "periodic_coordinate_policy"
            ],
            "hydrogen_bond_coordinate_path": (
                "raw_wrapped_frame_with_exact_minimum_image_vectors_v1"
            ),
            "water_policy": "exclude",
            "sparse_geometry_engine": "spatial_cell_list_exact_periodic_v1",
        },
        "cutoff_definitions": cutoff_definitions,
        "candidate_harmonization": endpoint_report,
        "chemistry_reports": chemistry_reports,
        "frame_matrix_representation": "sparse_spatial_partial_v3",
        "candidate_dictionary": candidate_dictionary,
        "conceptual_candidate_count": conceptual_count,
        "materialized_observed_candidate_count": len(candidate_dictionary),
        "unobserved_zero_candidate_count": conceptual_count - len(candidate_dictionary),
        "evaluated_frame_count": evaluated_frames,
        "conceptual_candidate_frame_count": conceptual_observations,
        "spatial_neighbor_pair_count": spatial_pairs,
        "explicit_geometry_evaluation_count": explicit_geometry,
        "present_event_count": present_events,
        "frame_bond_matrix": frame_records,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The implicit topology-defined candidate universe is not materialized; candidates never satisfying any declared cutoff are summarized as exact global zeros.",
            "Direct water contacts and water-mediated paths require their separate contracts.",
            "Occupancy does not establish energetic or mechanistic importance.",
        ],
    }


def _hydrogen_bond_discovery_project_serial(
    project_path: Path,
    hash_content: bool = False,
    *,
    harmonized_candidate_keys_override: set[HarmonizedCandidateKey] | None = None,
    candidate_harmonization_report_override: Mapping[str, object] | None = None,
    common_donor_endpoints_override: set[DonorHydrogenIdentityKey] | None = None,
    common_acceptor_endpoints_override: set[AcceptorIdentityKey] | None = None,
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    output_time_unit = project.get("time_unit")
    coordinate_unit = str(project["coordinate_unit"])
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=HydrogenBondDiscoveryError,
    )
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    if settings["output_mode"] == "sparse_spatial_observed_union_v3":
        if (
            common_donor_endpoints_override is None
            or common_acceptor_endpoints_override is None
            or candidate_harmonization_report_override is None
        ):
            (
                common_donor_endpoints_override,
                common_acceptor_endpoints_override,
                endpoint_report,
            ) = _automatic_endpoint_identity_intersection(
                system, system_path, settings
            )
        else:
            endpoint_report = dict(candidate_harmonization_report_override)
        return _hydrogen_bond_discovery_project_lazy_partial(
            source,
            project,
            settings,
            context,
            system,
            system_path,
            frame_selection_plan,
            frame_selection_report,
            issues,
            common_donor_endpoints_override,
            common_acceptor_endpoints_override,
            endpoint_report,
            hash_content=hash_content,
        )
    harmonized_candidate_keys = None
    candidate_harmonization_report: Dict[str, object] = {
        "policy": str(settings["candidate_harmonization"]),
        "selection_basis": "exact candidate dictionary equality across replicas",
    }
    if harmonized_candidate_keys_override is not None:
        harmonized_candidate_keys = set(harmonized_candidate_keys_override)
        candidate_harmonization_report = dict(
            candidate_harmonization_report_override or candidate_harmonization_report
        )
    elif (
        settings["mode"] == "automatic"
        and str(settings["candidate_harmonization"]).startswith("intersection_by_atom_")
    ):
        harmonized_candidate_keys, candidate_harmonization_report = (
            _automatic_candidate_harmonization(system, system_path, settings)
        )
        excluded = int(
            candidate_harmonization_report["excluded_from_common_union_count"]
        )
        issues.append({
            "severity": "warning" if excluded else "info",
            "code": "HBOND_CANDIDATE_DICTIONARY_HARMONIZED",
            "location": str(source),
            "message": (
                f"Retained {candidate_harmonization_report['common_candidate_count']} "
                "candidate triples present in every replica before coordinate "
                f"evaluation; {excluded} system- or replica-specific triples remain outside "
                "this comparative matrix and require per-system discovery for interpretation."
            ),
        })
    planned_feature_observation_count = None
    if harmonized_candidate_keys is not None:
        planned_feature_observation_count = (
            len(harmonized_candidate_keys)
            * int(frame_selection_report["selected_frame_count"])
        )
        if planned_feature_observation_count > int(
            settings["maximum_feature_observations"]
        ):
            raise HydrogenBondDiscoveryError(
                "maximum_feature_observations gate exceeded before coordinate "
                f"evaluation: {len(harmonized_candidate_keys)} common candidates x "
                f"{frame_selection_report['selected_frame_count']} selected frames = "
                f"{planned_feature_observation_count}, gate "
                f"{settings['maximum_feature_observations']}"
            )
    canonical_dictionary = None
    canonical_atom_dictionary = None
    frame_records: List[Dict[str, object]] = []
    occupancy_counts: Dict[Tuple[str, str, str], object] = {}
    cutoff_counts: Dict[Tuple[str, str, str], object] = {}
    frame_counts: Dict[Tuple[str, str, str], int] = {}
    chemistry_reports: List[Dict[str, object]] = []
    cutoff_definitions = settings["cutoff_definitions"]  # primary first
    evaluated_count = 0
    feature_observation_count = 0
    spatial_neighbor_pair_count = 0
    explicit_geometry_evaluation_count = 0
    sparse_output_mode = settings["output_mode"] in {
        "sparse_implicit_zero_v1", "sparse_packed_v2",
    }
    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str) or not connectivity_value.strip():
                raise HydrogenBondDiscoveryError(
                    f"{system_id}/{replica_id} requires explicit connectivity for donor-H discovery"
                )
            connectivity_path = resolve_manifest_path(connectivity_value, system_path)
            bonds, connectivity_provenance = load_connectivity(connectivity_path, len(atoms))
            candidate_atom_dictionary = None
            canonical_compact = None
            if settings["mode"] == "automatic":
                triples, roles = _automatic_candidate_triples(
                    atoms, bonds,
                    interaction_scope=str(settings["interaction_scope"]),
                    exclude_same_residue=bool(settings["exclude_same_residue"]),
                )
                raw_candidate_count = len(triples)
                if (
                    harmonized_candidate_keys is not None
                    and settings["candidate_harmonization"]
                    == "intersection_by_atom_identity_v2"
                ):
                    (
                        candidates,
                        canonical_compact,
                        candidate_atom_dictionary,
                    ) = _identity_harmonized_candidate_dictionaries(
                        atoms, roles, triples, harmonized_candidate_keys
                    )
                    chemistry_report = chemistry_summary(roles)
                elif sparse_output_mode:
                    if harmonized_candidate_keys is not None:
                        triples = [
                            triple for triple in triples
                            if triple in harmonized_candidate_keys
                        ]
                    candidates = [{
                        "bond_id": f"D{donor}-H{hydrogen}-A{acceptor}",
                        "donor_atom_index": donor,
                        "hydrogen_atom_index": hydrogen,
                        "acceptor_atom_index": acceptor,
                        "interaction_stratum": (
                            f"{roles[donor].entity_class}_to_{roles[acceptor].entity_class}"
                        ),
                    } for donor, hydrogen, acceptor in triples]
                    chemistry_report = chemistry_summary(roles)
                    candidate_atom_dictionary = _automatic_sparse_atom_dictionary(
                        atoms, roles, triples
                    )
                else:
                    candidates, chemistry_report = discover_automatic_candidate_bonds(
                        atoms, bonds,
                        interaction_scope=str(settings["interaction_scope"]),
                        exclude_same_residue=bool(settings["exclude_same_residue"]),
                    )
                    raw_candidate_count = len(candidates)
                    if harmonized_candidate_keys is not None:
                        candidates = [
                            candidate for candidate in candidates
                            if _candidate_key(candidate) in harmonized_candidate_keys
                        ]
                del triples, roles
                if harmonized_candidate_keys is not None:
                    chemistry_report = {
                        **chemistry_report,
                        "raw_candidate_count": raw_candidate_count,
                        "harmonized_candidate_count": len(candidates),
                        "excluded_noncommon_candidate_count": (
                            raw_candidate_count - len(candidates)
                        ),
                    }
            else:
                candidates = discover_candidate_bonds(
                    atoms, bonds,
                    settings["donor_atom_indices"],  # type: ignore[arg-type]
                    settings["acceptor_atom_indices"],  # type: ignore[arg-type]
                    allowed_donor_elements=settings["allowed_donor_elements"],  # type: ignore[arg-type]
                    allowed_acceptor_elements=settings["allowed_acceptor_elements"],  # type: ignore[arg-type]
                    exclude_same_residue=bool(settings["exclude_same_residue"]),
                )
                chemistry_report = {
                    "chemistry_policy": "explicit_atoms_connectivity_v1",
                    "entity_atom_counts": {}, "chemistry_confidence_atom_counts": {},
                    "donor_atom_count": len(settings["donor_atom_indices"]),
                    "acceptor_atom_count": len(settings["acceptor_atom_indices"]),
                }
            chemistry_reports.append({
                "system_id": system_id, "replica_id": replica_id, **chemistry_report,
            })
            if len(candidates) > int(settings["maximum_candidate_bonds"]):
                raise HydrogenBondDiscoveryError("maximum_candidate_bonds gate exceeded")
            canonical_candidate_rows = (
                canonical_compact
                if settings["mode"] == "automatic" and canonical_compact is not None
                else candidates
            )
            compact = [
                {
                    **{key: row[key] for key in (
                    "bond_id", "donor_atom_index", "hydrogen_atom_index",
                    "acceptor_atom_index",
                    )},
                    **(
                        {"interaction_stratum": row["interaction_stratum"]}
                        if "interaction_stratum" in row else {}
                    ),
                }
                for row in canonical_candidate_rows
            ]
            sparse_mode = sparse_output_mode
            if canonical_dictionary is None:
                canonical_dictionary = (
                    compact
                    if sparse_mode
                    else canonical_candidate_rows
                )
                if sparse_mode:
                    canonical_atom_dictionary = (
                        candidate_atom_dictionary
                        if candidate_atom_dictionary is not None
                        else _sparse_atom_dictionary(candidates)
                    )
                if planned_feature_observation_count is None:
                    planned_feature_observation_count = (
                        len(candidates)
                        * int(frame_selection_report["selected_frame_count"])
                    )
                    if planned_feature_observation_count > int(
                        settings["maximum_feature_observations"]
                    ):
                        raise HydrogenBondDiscoveryError(
                            "maximum_feature_observations gate exceeded before "
                            f"coordinate evaluation: {len(candidates)} candidates x "
                            f"{frame_selection_report['selected_frame_count']} "
                            f"selected frames = {planned_feature_observation_count}, "
                            f"gate {settings['maximum_feature_observations']}"
                        )
            else:
                prior = [
                    {
                        **{key: row[key] for key in (
                            "bond_id", "donor_atom_index", "hydrogen_atom_index",
                            "acceptor_atom_index",
                        )},
                        **(
                            {"interaction_stratum": row["interaction_stratum"]}
                            if "interaction_stratum" in row else {}
                        ),
                    }
                    for row in canonical_dictionary
                ]
                if compact != prior:
                    raise HydrogenBondDiscoveryError(
                        "candidate dictionary differs across replicas; use harmonized atom mappings"
                    )
            sparse_evaluator = (
                CompiledSparseHydrogenBondEvaluator.compile(
                    candidates,
                    cutoff_definitions,  # type: ignore[arg-type]
                    int(settings["candidate_chunk_size"]),
                )
                if sparse_mode else None
            )
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            reconstruction_atom_indices = tuple(sorted({
                int(candidate[field])
                for candidate in candidates
                for field in (
                    "donor_atom_index", "hydrogen_atom_index", "acceptor_atom_index"
                )
            }))
            reference = processor.process(
                next(iter_coordinate_frames(topology_path, coordinate_unit)),
                str(topology_path),
                reconstruction_atom_indices,
            )
            for candidate in candidates:
                donor = int(candidate["donor_atom_index"])
                hydrogen = int(candidate["hydrogen_atom_index"])
                reference_bond = distance_angstrom(
                    reference.coordinates_angstrom[donor],
                    reference.coordinates_angstrom[hydrogen],
                    reference.cell_vectors_angstrom,
                )
                if reference_bond > float(settings["maximum_reference_donor_hydrogen_bond_angstrom"]):
                    raise HydrogenBondDiscoveryError(
                        f"candidate {candidate['bond_id']} reference donor-H distance exceeds gate"
                    )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
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
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if not selected:
                        continue
                    sparse_mode = settings["output_mode"] in {
                        "sparse_implicit_zero_v1", "sparse_packed_v2",
                    }
                    if sparse_mode:
                        assert sparse_evaluator is not None
                        sparse = sparse_evaluator.evaluate(
                            frame.coordinates_angstrom,
                            cell=frame.cell_vectors_angstrom,
                        )
                        spatial_neighbor_pair_count += int(
                            sparse.get("spatial_neighbor_pair_count", len(candidates))
                        )
                        explicit_geometry_evaluation_count += int(
                            sparse.get(
                                "explicit_geometry_evaluation_count", len(candidates)
                            )
                        )
                        present_by_cutoff = sparse["present_candidate_indices_by_cutoff"]
                        binary = None
                    else:
                        binary = []
                        distances = []
                        angles = []
                        for candidate in candidates:
                            present, distance, angle = hydrogen_bond_present(
                                frame.coordinates_angstrom[int(candidate["donor_atom_index"])],
                                frame.coordinates_angstrom[int(candidate["hydrogen_atom_index"])],
                                frame.coordinates_angstrom[int(candidate["acceptor_atom_index"])],
                                float(settings["maximum_donor_acceptor_distance_angstrom"]),
                                float(settings["minimum_donor_hydrogen_acceptor_angle_degrees"]),
                                frame.cell_vectors_angstrom,
                            )
                            binary.append(int(present))
                            distances.append(distance)
                            angles.append(angle)
                    evaluated_count += 1
                    feature_observation_count += len(candidates)
                    if feature_observation_count > int(settings["maximum_feature_observations"]):
                        raise HydrogenBondDiscoveryError("maximum_feature_observations gate exceeded")
                    key = (system_id, replica_id, segment_id)
                    frame_counts[key] = frame_counts.get(key, 0) + 1
                    locator = {
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                    }
                    if sparse_mode:
                        primary_indices = present_by_cutoff[0]  # type: ignore[index]
                        counts = occupancy_counts.setdefault(key, {})
                        per_cutoff = cutoff_counts.setdefault(
                            key, [dict() for _ in cutoff_definitions]
                        )
                        for index in primary_indices:
                            counts[index] = counts.get(index, 0) + 1  # type: ignore[union-attr]
                        for cutoff_index, indices in enumerate(present_by_cutoff):  # type: ignore[arg-type]
                            cutoff_map = per_cutoff[cutoff_index]  # type: ignore[index]
                            for index in indices:
                                cutoff_map[index] = cutoff_map.get(index, 0) + 1
                        if settings["output_mode"] == "sparse_packed_v2":
                            frame_records.append({
                                **locator,
                                "candidate_count": len(candidates),
                                **pack_sparse_present_geometry(
                                    sparse["present_geometry"],  # type: ignore[arg-type]
                                    cutoff_definitions,  # type: ignore[arg-type]
                                    len(candidates),
                                ),
                            })
                        else:
                            frame_records.append({
                                **locator,
                                "representation": "sparse_implicit_zero_v1",
                                "candidate_count": len(candidates),
                                "primary_present_candidate_indices": primary_indices,
                                "present_bond_ids": [candidates[index]["bond_id"] for index in primary_indices],
                                "cutoff_present_candidate_indices": {
                                    str(cutoff["cutoff_id"]): indices
                                    for cutoff, indices in zip(cutoff_definitions, present_by_cutoff)  # type: ignore[arg-type]
                                },
                                "present_geometry": sparse["present_geometry"],
                            })
                    else:
                        counts = occupancy_counts.setdefault(key, [0] * len(candidates))
                        per_cutoff = cutoff_counts.setdefault(
                            key, [[0] * len(cutoff_definitions) for _ in candidates]
                        )
                        for index, present in enumerate(binary):  # type: ignore[arg-type]
                            counts[index] += present  # type: ignore[index]
                            for cutoff_index, cutoff in enumerate(cutoff_definitions):
                                if (
                                    distances[index] <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
                                    and angles[index] >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
                                ):
                                    per_cutoff[index][cutoff_index] += 1  # type: ignore[index]
                        frame_records.append({
                            **locator,
                            "binary_values": binary,
                            "present_bond_ids": [
                                candidates[index]["bond_id"]
                                for index, present in enumerate(binary) if present  # type: ignore[arg-type]
                            ],
                            "donor_acceptor_distances_angstrom": distances,
                            "donor_hydrogen_acceptor_angles_degrees": angles,
                        })
            if settings["mode"] == "automatic":
                provisional = int(chemistry_report["chemistry_confidence_atom_counts"].get("provisional", 0))
                issues.append({
                    "severity": "warning" if provisional else "info",
                    "code": "HBOND_AUTO_CHEMISTRY_PROVISIONAL" if provisional else "HBOND_AUTO_CHEMISTRY_TEMPLATED",
                    "location": f"{system_id}/{replica_id}",
                    "message": (
                        f"Automatic topology-template chemistry found {provisional} atoms using the "
                        "generic ligand fallback; validate ligand protonation, formal charge, and bond order "
                        "before publication use."
                        if provisional else
                        "Automatic topology-template chemistry used standard protein/nucleic-acid templates."
                    ),
                    "connectivity": connectivity_provenance,
                })
            else:
                issues.append({
                    "severity": "warning",
                    "code": "HBOND_DISCOVERY_CHEMISTRY_SCOPE_EXPLICIT",
                    "location": f"{system_id}/{replica_id}",
                    "message": (
                        "Candidate chemistry is limited to declared donor/acceptor atoms, "
                        "allowed elements, and connectivity-declared donor hydrogens"
                    ),
                    "connectivity": connectivity_provenance,
                })
    assert canonical_dictionary is not None
    candidate_stratum_counts: Dict[str, int] = {}
    for candidate in canonical_dictionary:
        stratum = candidate.get("interaction_stratum")
        if isinstance(stratum, str):
            candidate_stratum_counts[stratum] = (
                candidate_stratum_counts.get(stratum, 0) + 1
            )
    occupancies = []
    cutoff_occupancies = []
    packed_cutoff_occupancy_segments = []
    for key, counts in sorted(occupancy_counts.items()):
        frame_count = frame_counts[key]
        if settings["output_mode"] in {"sparse_implicit_zero_v1", "sparse_packed_v2"}:
            primary_items = sorted(counts.items())  # type: ignore[union-attr]
        else:
            primary_items = enumerate(counts)  # type: ignore[arg-type]
        for candidate_index, count in primary_items:
            candidate = canonical_dictionary[candidate_index]
            occupancies.append({
                "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
                "bond_id": candidate["bond_id"], "evaluated_frame_count": frame_count,
                "present_frame_count": count, "occupancy_fraction": count / frame_count,
            })
        if settings["output_mode"] == "sparse_packed_v2":
            packed_cutoff_occupancy_segments.append({
                "system_id": key[0],
                "replica_id": key[1],
                "segment_id": key[2],
                **pack_sparse_cutoff_counts(
                    cutoff_counts[key],  # type: ignore[arg-type]
                    cutoff_definitions,  # type: ignore[arg-type]
                    len(canonical_dictionary),
                    frame_count,
                ),
            })
            cutoff_items = ()
        elif settings["output_mode"] == "sparse_implicit_zero_v1":
            cutoff_items = (
                (cutoff_index, candidate_index, count)
                for cutoff_index, values in enumerate(cutoff_counts[key])  # type: ignore[union-attr]
                for candidate_index, count in sorted(values.items())
            )
        else:
            cutoff_items = (
                (cutoff_index, candidate_index, per_candidate[cutoff_index])
                for candidate_index, per_candidate in enumerate(cutoff_counts[key])  # type: ignore[union-attr]
                for cutoff_index in range(len(cutoff_definitions))
            )
        for cutoff_index, candidate_index, count in cutoff_items:
            candidate = canonical_dictionary[candidate_index]
            cutoff = cutoff_definitions[cutoff_index]
            cutoff_occupancies.append({
                "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
                "bond_id": candidate["bond_id"], "cutoff_id": cutoff["cutoff_id"],
                "cutoff_kind": cutoff["kind"],
                "maximum_donor_acceptor_distance_angstrom": cutoff["maximum_donor_acceptor_distance_angstrom"],
                "minimum_donor_hydrogen_acceptor_angle_degrees": cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"],
                "evaluated_frame_count": frame_count, "present_frame_count": count,
                "occupancy_fraction": count / frame_count,
            })
    if evaluated_count != int(frame_selection_report["selected_frame_count"]):
        raise HydrogenBondDiscoveryError(
            "selected/evaluated frame accounting mismatch: "
            f"selected={frame_selection_report['selected_frame_count']}, "
            f"evaluated={evaluated_count}"
        )
    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": str(source),
            "message": (
                "Hydrogen-bond discovery evaluated "
                f"{frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "hydrogen_bond_discovery",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "frame_selection": frame_selection_report,
        "observation_accounting": {
            "source_physical_frame_count": int(
                frame_selection_report["source_frame_count"]
            ),
            "selected_physical_frame_count": evaluated_count,
            "symmetry_expanded_observation_count": evaluated_count,
            "candidate_frame_feature_observation_count": feature_observation_count,
            "spatial_neighbor_pair_count": (
                spatial_neighbor_pair_count if sparse_output_mode else None
            ),
            "explicit_geometry_evaluation_count": (
                explicit_geometry_evaluation_count if sparse_output_mode else None
            ),
            "geometry_evaluation_avoidance_fraction": (
                1.0 - explicit_geometry_evaluation_count / feature_observation_count
                if sparse_output_mode and feature_observation_count else None
            ),
            "subsampling_triggered": (
                evaluated_count < int(frame_selection_report["source_frame_count"])
            ),
        },
        "geometry_contract": {
            "distance": "donor-acceptor minimum-image distance when a periodic cell is present",
            "angle": "donor-hydrogen-acceptor angle at hydrogen",
            "distance_definition": "donor_acceptor_v1",
            "coordinate_reconstruction": project["periodic_coordinate_policy"],
            "water_policy": "exclude",
            "sparse_geometry_engine": (
                "spatial_cell_list_exact_periodic_v1"
                if sparse_output_mode else None
            ),
        },
        "cutoff_definitions": cutoff_definitions,
        "candidate_harmonization": candidate_harmonization_report,
        "chemistry_reports": chemistry_reports,
        "frame_matrix_representation": settings["output_mode"],
        "sparse_zero_contract": (
            "Candidates absent from a frame's sparse present-event payload are "
            "evaluated zeros; occupancy tables omit zero-present candidates."
            if settings["output_mode"] in {
                "sparse_implicit_zero_v1", "sparse_packed_v2",
            } else None
        ),
        "packed_event_codec": (
            PACKED_EVENT_CODEC if settings["output_mode"] == "sparse_packed_v2" else None
        ),
        "cutoff_occupancy_representation": (
            "sparse_packed_cutoff_counts_v1"
            if settings["output_mode"] == "sparse_packed_v2" else "json_rows_v1"
        ),
        "packed_cutoff_count_codec": (
            PACKED_CUTOFF_COUNT_CODEC
            if settings["output_mode"] == "sparse_packed_v2" else None
        ),
        "candidate_dictionary": canonical_dictionary,
        "candidate_stratum_counts": candidate_stratum_counts,
        "atom_dictionary": (
            canonical_atom_dictionary
            if settings["output_mode"] in {
                "sparse_implicit_zero_v1", "sparse_packed_v2",
            } else None
        ),
        "candidate_count": len(canonical_dictionary),
        "planned_feature_observation_count": planned_feature_observation_count,
        "evaluated_frame_count": evaluated_count,
        "feature_observation_count": feature_observation_count,
        "frame_bond_matrix": frame_records,
        "occupancies": occupancies,
        "cutoff_occupancies": (
            None if settings["output_mode"] == "sparse_packed_v2"
            else cutoff_occupancies
        ),
        "packed_cutoff_occupancy_segments": (
            packed_cutoff_occupancy_segments
            if settings["output_mode"] == "sparse_packed_v2" else None
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Automatic mode uses standard protein/nucleic-acid templates and records untemplated ligand N/O/S inference as provisional; it cannot establish a missing or incorrect protonation, formal charge, bond order, or tautomer state.",
            "The legacy explicit mode retains declared atom-index selections for publication-locked comparisons.",
            "Donor-hydrogen association comes only from explicit connectivity and therefore fails closed when connectivity is absent.",
            "This direct-bond module excludes water-mediated paths; use the separately validated one-water network contract.",
            "The full candidate dictionary is retained without outcome-dependent occupancy filtering, preventing a hidden feature-selection step.",
            "For homologous multi-system comparisons, intersection_by_atom_identity_v2 retains chemical-position candidate triples present in every replica before coordinates are read; system-specific chemical candidates remain available only through separate per-system discovery reports.",
            "Hydrogen-bond occupancy does not establish energetic or mechanistic importance.",
        ],
    }


def _reduce_hbond_discovery_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in (
            "module_id", "settings", "geometry_contract", "cutoff_definitions",
            "candidate_harmonization", "frame_matrix_representation",
            "candidate_dictionary", "candidate_count",
        ):
            if report.get(key) != first.get(key):
                raise HydrogenBondDiscoveryError(
                    f"replica hydrogen-bond reports disagree on {key}"
                )
    first["frame_selection"] = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    for key in (
        "chemistry_reports", "frame_bond_matrix", "occupancies",
        "packed_cutoff_occupancy_segments",
    ):
        if first.get(key) is not None:
            first[key] = [
                row for report in reports for row in (report.get(key) or [])
            ]
    if first.get("cutoff_occupancies") is not None:
        first["cutoff_occupancies"] = [
            row for report in reports
            for row in (report.get("cutoff_occupancies") or [])
        ]
    evaluated = sum(int(report.get("evaluated_frame_count", 0)) for report in reports)
    feature_count = int(first["candidate_count"]) * evaluated
    spatial_pairs = sum(
        int(report.get("observation_accounting", {}).get(
            "spatial_neighbor_pair_count", 0
        ) or 0)
        for report in reports
    )
    explicit_geometry = sum(
        int(report.get("observation_accounting", {}).get(
            "explicit_geometry_evaluation_count", 0
        ) or 0)
        for report in reports
    )
    if feature_count > int(first["settings"]["maximum_feature_observations"]):  # type: ignore[index]
        raise HydrogenBondDiscoveryError(
            "parallel hydrogen-bond feature count exceeds maximum_feature_observations"
        )
    first["planned_feature_observation_count"] = feature_count
    first["evaluated_frame_count"] = evaluated
    first["feature_observation_count"] = feature_count
    frame_selection = first["frame_selection"]
    first["observation_accounting"] = {
        "source_physical_frame_count": int(frame_selection["source_frame_count"]),
        "selected_physical_frame_count": evaluated,
        "symmetry_expanded_observation_count": evaluated,
        "candidate_frame_feature_observation_count": feature_count,
        "spatial_neighbor_pair_count": (
            spatial_pairs
            if first["frame_matrix_representation"] in {
                "sparse_implicit_zero_v1", "sparse_packed_v2",
            } else None
        ),
        "explicit_geometry_evaluation_count": (
            explicit_geometry
            if first["frame_matrix_representation"] in {
                "sparse_implicit_zero_v1", "sparse_packed_v2",
            } else None
        ),
        "geometry_evaluation_avoidance_fraction": (
            1.0 - explicit_geometry / feature_count
            if feature_count
            and first["frame_matrix_representation"] in {
                "sparse_implicit_zero_v1", "sparse_packed_v2",
            } else None
        ),
        "subsampling_triggered": evaluated < int(frame_selection["source_frame_count"]),
    }
    issues = [
        issue for issue in unique_issues(reports)
        if issue.get("code") not in {
            "FRAME_SUBSAMPLING", "HBOND_CANDIDATE_DICTIONARY_HARMONIZED",
        }
    ]
    harmonization = first["candidate_harmonization"]
    excluded = int(harmonization.get("excluded_from_common_union_count", 0))
    if str(harmonization.get("policy", "")).startswith("intersection_by_atom_"):
        issues.append({
            "severity": "warning" if excluded else "info",
            "code": "HBOND_CANDIDATE_DICTIONARY_HARMONIZED",
            "location": source_context["project_manifest_path"],
            "message": (
                f"Retained {harmonization.get('common_candidate_count')} candidate "
                f"triples before coordinate evaluation; {excluded} noncommon triples "
                "remain outside the comparative matrix."
            ),
        })
    if evaluated < int(frame_selection["source_frame_count"]):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": source_context["project_manifest_path"],
            "message": (
                f"Hydrogen-bond discovery evaluated {evaluated} of "
                f"{frame_selection['source_frame_count']} source frames under "
                f"{frame_selection['mode']}"
            ),
        })
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(issue.get("severity") == "warning" for issue in issues)
    restore_source_provenance(first, source_context)
    return first


def _identity_from_record(value: object) -> AtomIdentityKey:
    if not isinstance(value, Mapping):
        raise HydrogenBondDiscoveryError("candidate identity record is malformed")
    try:
        return (
            str(value["chain_id"]),
            int(value["residue_number"]),
            str(value["insertion_code"]),
            str(value["atom_name"]),
            str(value["altloc"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HydrogenBondDiscoveryError(
            "candidate identity record is incomplete"
        ) from exc


def _reduce_lazy_hbond_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    """Merge replica-local observed dictionaries into one pooled sparse union."""

    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports:
        for key in (
            "module_id", "settings", "geometry_contract", "cutoff_definitions",
            "candidate_harmonization", "conceptual_candidate_count",
        ):
            if report.get(key) != first.get(key):
                raise HydrogenBondDiscoveryError(
                    f"lazy replica hydrogen-bond reports disagree on {key}"
                )

    stratum_by_key: Dict[CandidateIdentityKey, str] = {}
    local_keys_by_report: List[List[CandidateIdentityKey]] = []
    for report in reports:
        local_keys = []
        for row in report.get("candidate_dictionary", []):
            key = (
                _identity_from_record(row.get("donor_identity")),
                _identity_from_record(row.get("hydrogen_identity")),
                _identity_from_record(row.get("acceptor_identity")),
            )
            stratum = str(row.get("interaction_stratum"))
            prior = stratum_by_key.setdefault(key, stratum)
            if prior != stratum:
                raise HydrogenBondDiscoveryError(
                    "observed candidate interaction stratum differs across replicas"
                )
            local_keys.append(key)
        if len(set(local_keys)) != len(local_keys):
            raise HydrogenBondDiscoveryError(
                "replica observed candidate dictionary contains duplicates"
            )
        local_keys_by_report.append(local_keys)

    global_keys = sorted(stratum_by_key)
    if not global_keys:
        raise HydrogenBondDiscoveryError(
            "no hydrogen bond satisfied any declared cutoff in selected frames"
        )
    global_index = {key: index for index, key in enumerate(global_keys)}
    atom_keys = sorted({atom for key in global_keys for atom in key})
    atom_index = {key: index for index, key in enumerate(atom_keys)}
    candidate_dictionary = [{
        "bond_id": _identity_bond_id(key),
        "donor_atom_index": atom_index[key[0]],
        "hydrogen_atom_index": atom_index[key[1]],
        "acceptor_atom_index": atom_index[key[2]],
        "donor_identity": _identity_record(key[0]),
        "hydrogen_identity": _identity_record(key[1]),
        "acceptor_identity": _identity_record(key[2]),
        "interaction_stratum": stratum_by_key[key],
    } for key in global_keys]
    atom_dictionary = [{
        "atom_index": atom_index[key],
        "identity": _identity_record(key),
    } for key in atom_keys]

    cutoff_definitions = first["cutoff_definitions"]
    frame_records = []
    frame_counts: Dict[Tuple[str, str, str], int] = {}
    cutoff_counts: Dict[Tuple[str, str, str], List[Dict[int, int]]] = {}
    for report, local_keys in zip(reports, local_keys_by_report):
        local_to_global = {
            local_index: global_index[key]
            for local_index, key in enumerate(local_keys)
        }
        for frame in report.get("frame_bond_matrix", []):
            events = unpack_sparse_present_events(frame)
            geometry_rows = []
            key = (
                str(frame["system_id"]),
                str(frame["replica_id"]),
                str(frame["segment_id"]),
            )
            frame_counts[key] = frame_counts.get(key, 0) + 1
            per_cutoff = cutoff_counts.setdefault(
                key, [dict() for _ in cutoff_definitions]
            )
            for event in events:
                local_index = int(event["candidate_index"])
                if local_index not in local_to_global:
                    raise HydrogenBondDiscoveryError(
                        "packed event references an unknown local candidate"
                    )
                candidate_index = local_to_global[local_index]
                mask = int(event["cutoff_mask"])
                matched = []
                for cutoff_index, cutoff in enumerate(cutoff_definitions):
                    if mask & (1 << cutoff_index):
                        matched.append(str(cutoff["cutoff_id"]))
                        counts = per_cutoff[cutoff_index]
                        counts[candidate_index] = counts.get(candidate_index, 0) + 1
                geometry_rows.append({
                    "candidate_index": candidate_index,
                    "donor_acceptor_distance_angstrom": float(
                        event["donor_acceptor_distance_angstrom"]
                    ),
                    "donor_hydrogen_acceptor_angle_degrees": float(
                        event["donor_hydrogen_acceptor_angle_degrees"]
                    ),
                    "present_cutoff_ids": matched,
                })
            geometry_rows.sort(key=lambda row: int(row["candidate_index"]))
            frame_records.append({
                key_name: frame[key_name]
                for key_name in (
                    "system_id", "replica_id", "segment_id", "source_frame_index",
                    "axis_kind", "axis_value",
                )
            } | {
                "candidate_count": len(candidate_dictionary),
                **pack_sparse_present_geometry(
                    geometry_rows,
                    cutoff_definitions,  # type: ignore[arg-type]
                    len(candidate_dictionary),
                ),
            })

    occupancies = []
    packed_cutoff_segments = []
    for segment_key, per_cutoff in sorted(cutoff_counts.items()):
        frame_count = frame_counts[segment_key]
        for candidate_index, count in sorted(per_cutoff[0].items()):
            occupancies.append({
                "system_id": segment_key[0],
                "replica_id": segment_key[1],
                "segment_id": segment_key[2],
                "bond_id": candidate_dictionary[candidate_index]["bond_id"],
                "evaluated_frame_count": frame_count,
                "present_frame_count": count,
                "occupancy_fraction": count / frame_count,
            })
        packed_cutoff_segments.append({
            "system_id": segment_key[0],
            "replica_id": segment_key[1],
            "segment_id": segment_key[2],
            **pack_sparse_cutoff_counts(
                per_cutoff,
                cutoff_definitions,  # type: ignore[arg-type]
                len(candidate_dictionary),
                frame_count,
            ),
        })

    evaluated = sum(int(report["evaluated_frame_count"]) for report in reports)
    conceptual_count = int(first["conceptual_candidate_count"])
    conceptual_observations = conceptual_count * evaluated
    spatial_pairs = sum(int(report["spatial_neighbor_pair_count"]) for report in reports)
    explicit_geometry = sum(
        int(report["explicit_geometry_evaluation_count"]) for report in reports
    )
    present_event_count = sum(int(report["present_event_count"]) for report in reports)
    frame_selection = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    candidate_stratum_counts: Dict[str, int] = {}
    for row in candidate_dictionary:
        stratum = str(row["interaction_stratum"])
        candidate_stratum_counts[stratum] = candidate_stratum_counts.get(stratum, 0) + 1
    issues = unique_issues(reports)
    first.update({
        "frame_selection": frame_selection,
        "chemistry_reports": [
            row for report in reports for row in report.get("chemistry_reports", [])
        ],
        "frame_matrix_representation": "sparse_spatial_observed_union_v3",
        "sparse_zero_contract": (
            "Candidates absent from a frame's packed event payload are exact zeros. "
            "Topology-eligible candidates absent from the pooled observed dictionary "
            "are exact global zeros under every declared cutoff."
        ),
        "packed_event_codec": PACKED_EVENT_CODEC,
        "cutoff_occupancy_representation": "sparse_packed_cutoff_counts_v1",
        "packed_cutoff_count_codec": PACKED_CUTOFF_COUNT_CODEC,
        "candidate_dictionary": candidate_dictionary,
        "atom_dictionary": atom_dictionary,
        "candidate_count": len(candidate_dictionary),
        "candidate_stratum_counts": candidate_stratum_counts,
        "conceptual_candidate_count": conceptual_count,
        "materialized_observed_candidate_count": len(candidate_dictionary),
        "unobserved_zero_candidate_count": conceptual_count - len(candidate_dictionary),
        "evaluated_frame_count": evaluated,
        "conceptual_candidate_frame_count": conceptual_observations,
        "spatial_neighbor_pair_count": spatial_pairs,
        "explicit_geometry_evaluation_count": explicit_geometry,
        "present_event_count": present_event_count,
        "observation_accounting": {
            "source_physical_frame_count": int(frame_selection["source_frame_count"]),
            "selected_physical_frame_count": evaluated,
            "conceptual_candidate_frame_count": conceptual_observations,
            "spatial_neighbor_pair_count": spatial_pairs,
            "explicit_geometry_evaluation_count": explicit_geometry,
            "present_event_count": present_event_count,
            "geometry_evaluation_avoidance_fraction": (
                1.0 - explicit_geometry / conceptual_observations
                if conceptual_observations else None
            ),
            "subsampling_triggered": (
                evaluated < int(frame_selection["source_frame_count"])
            ),
        },
        "frame_bond_matrix": frame_records,
        "occupancies": occupancies,
        "cutoff_occupancies": None,
        "packed_cutoff_occupancy_segments": packed_cutoff_segments,
        "issues": issues,
        "error_count": sum(issue.get("severity") == "error" for issue in issues),
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
    })
    restore_source_provenance(first, source_context)
    return first


def hydrogen_bond_discovery_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Discover chemistry globally, evaluate replicas in parallel, then reduce."""

    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    selection = settings.get("frame_selection")
    if (
        settings["output_mode"] != "sparse_spatial_observed_union_v3"
        and isinstance(selection, dict)
        and selection.get("mode") == "auto_resource_budget_v1"
    ):
        return _hydrogen_bond_discovery_project_serial(
            project_path, hash_content=hash_content
        )
    keys = None
    common_donors = None
    common_acceptors = None
    harmonization = None
    if settings["output_mode"] == "sparse_spatial_observed_union_v3":
        context = compile_project_context_file(source, hash_content=hash_content)
        system_path = Path(context["system_manifest_path"])
        common_donors, common_acceptors, harmonization = (
            _automatic_endpoint_identity_intersection(
                load_json(system_path), system_path, settings
            )
        )
    elif (
        settings["mode"] == "automatic"
        and str(settings["candidate_harmonization"]).startswith("intersection_by_atom_")
    ):
        context = compile_project_context_file(source, hash_content=hash_content)
        system_path = Path(context["system_manifest_path"])
        keys, harmonization = _automatic_candidate_harmonization(
            load_json(system_path), system_path, settings
        )
    return execute_replica_final_module(
        project_path,
        runner_id="hydrogen_bond_discovery",
        hash_content=hash_content,
        reducer=(
            _reduce_lazy_hbond_reports
            if settings["output_mode"] == "sparse_spatial_observed_union_v3"
            else _reduce_hbond_discovery_reports
        ),
        worker_payload={
            "harmonized_candidate_keys": (
                [
                    [list(atom) for atom in row]
                    if harmonization is not None
                    and harmonization.get("policy") == "intersection_by_atom_identity_v2"
                    else list(row)
                    for row in sorted(keys)
                ]
                if keys is not None else None
            ),
            "candidate_harmonization_report": harmonization,
            "common_donor_endpoints": (
                [
                    [list(donor), list(hydrogen), entity_class]
                    for donor, hydrogen, entity_class in sorted(common_donors)
                ]
                if common_donors is not None else None
            ),
            "common_acceptor_endpoints": (
                [
                    [list(acceptor), entity_class]
                    for acceptor, entity_class in sorted(common_acceptors)
                ]
                if common_acceptors is not None else None
            ),
        },
    )


def hydrogen_bond_discovery_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return hydrogen_bond_discovery_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, HydrogenBondDiscoveryError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError,
        OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "hydrogen_bond_discovery",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "HYDROGEN_BOND_DISCOVERY_INVALID", "message": message}
                for message in messages
            ],
        }
