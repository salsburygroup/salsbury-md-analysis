"""Connectivity-backed hydrogen-bond candidate discovery and frame matrices."""

from __future__ import annotations

import hashlib
import json
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
from .hydrogen_bond_chemistry import (
    AtomChemicalRole,
    chemistry_summary,
    infer_atom_chemical_roles,
    scope_allows,
)
from .hydrogen_bonds import hydrogen_bond_present, distance_angstrom
from .hydrogen_bond_sparse import (
    CompiledSparseHydrogenBondEvaluator,
    PACKED_CUTOFF_COUNT_CODEC,
    PACKED_EVENT_CODEC,
    pack_sparse_cutoff_counts,
    pack_sparse_present_geometry,
    unpack_sparse_cutoff_counts,
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
        # Evaluate every topology-defined candidate within each system.  A
        # separate chemical-position view provides comparable features without
        # discarding system-specific chemistry or assuming atom-index identity.
        candidate_harmonization = raw.get(
            "candidate_harmonization",
            "per_system_full_with_chemical_identity_comparison_v2",
        )
        if candidate_harmonization not in {
            "strict_v1", "intersection_by_atom_index_v1",
            "per_system_full_with_chemical_identity_comparison_v2",
        }:
            raise HydrogenBondDiscoveryError(
                "candidate_harmonization must be strict_v1, "
                "intersection_by_atom_index_v1, or "
                "per_system_full_with_chemical_identity_comparison_v2"
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
    }:
        raise HydrogenBondDiscoveryError(
            "output_mode must be dense_v1, sparse_implicit_zero_v1, or sparse_packed_v2"
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


def automatic_candidate_inventory(
    system: Mapping[str, object], system_path: Path, settings: Mapping[str, object],
) -> Dict[str, object]:
    """Inventory full per-replica candidates and shared chemical identities.

    The inventory is topology-only and can therefore be used by the resource
    planner without observing trajectory coordinates or occupancies.
    """

    rows = []
    unique_sets: List[set[Tuple[object, ...]]] = []
    union: set[Tuple[object, ...]] = set()
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
            identities = [
                _chemical_position_key((
                    atoms[donor].as_dict(), atoms[hydrogen].as_dict(),
                    atoms[acceptor].as_dict(),
                ))
                for donor, hydrogen, acceptor in triples
            ]
            counts: Dict[Tuple[object, ...], int] = {}
            for identity in identities:
                counts[identity] = counts.get(identity, 0) + 1
            unique = {identity for identity, count in counts.items() if count == 1}
            ambiguous = {identity for identity, count in counts.items() if count != 1}
            unique_sets.append(unique)
            union.update(unique)
            union.update(ambiguous)
            rows.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "raw_candidate_count": len(triples),
                "unique_chemical_identity_count": len(unique),
                "ambiguous_chemical_identity_count": len(ambiguous),
            })
    if not rows:
        raise HydrogenBondDiscoveryError("system manifest contains no replicas")
    common = set.intersection(*unique_sets)
    return {
        "policy": "per_system_full_with_chemical_identity_comparison_v2",
        "common_candidate_count": len(common),
        "union_candidate_count": len(union),
        "replica_dictionaries": rows,
        "total_candidate_count_across_replicas": sum(
            int(row["raw_candidate_count"]) for row in rows
        ),
        "maximum_candidate_count_per_replica": max(
            int(row["raw_candidate_count"]) for row in rows
        ),
        "mean_candidate_count_per_replica": (
            sum(int(row["raw_candidate_count"]) for row in rows) / len(rows)
        ),
        "selection_basis": (
            "Every topology-defined per-system candidate is retained. Shared "
            "identities are calculated from donor, hydrogen, and acceptor chemical "
            "role/topology positions before coordinates are read."
        ),
    }


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


def _hydrogen_bond_discovery_project_serial(
    project_path: Path,
    hash_content: bool = False,
    *,
    harmonized_candidate_keys_override: set[Tuple[int, int, int]] | None = None,
    candidate_harmonization_report_override: Mapping[str, object] | None = None,
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
        and settings["candidate_harmonization"] == "intersection_by_atom_index_v1"
    ):
        harmonized_candidate_keys, candidate_harmonization_report = (
            _automatic_candidate_intersection(system, system_path, settings)
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
                "candidate atom-index triples present in every replica before coordinate "
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
            if settings["mode"] == "automatic":
                if sparse_output_mode:
                    triples, roles = _automatic_candidate_triples(
                        atoms, bonds,
                        interaction_scope=str(settings["interaction_scope"]),
                        exclude_same_residue=bool(settings["exclude_same_residue"]),
                    )
                    raw_candidate_count = len(triples)
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
                    } for donor, hydrogen, acceptor in triples]
                    chemistry_report = chemistry_summary(roles)
                    candidate_atom_dictionary = _automatic_sparse_atom_dictionary(
                        atoms, roles, triples
                    )
                    del triples, roles
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
            compact = (
                candidates
                if settings["mode"] == "automatic" and sparse_output_mode
                else [
                    {key: row[key] for key in (
                        "bond_id", "donor_atom_index", "hydrogen_atom_index",
                        "acceptor_atom_index",
                    )}
                    for row in candidates
                ]
            )
            sparse_mode = sparse_output_mode
            if canonical_dictionary is None:
                canonical_dictionary = compact if sparse_mode else candidates
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
                prior = (
                    canonical_dictionary
                    if sparse_mode else [
                        {key: row[key] for key in (
                            "bond_id", "donor_atom_index", "hydrogen_atom_index",
                            "acceptor_atom_index",
                        )}
                        for row in canonical_dictionary
                    ]
                )
                if compact != prior:
                    raise HydrogenBondDiscoveryError(
                        "candidate dictionary differs across replicas; use harmonized atom mappings"
                    )
            if sparse_mode:
                candidates = compact
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
            "For homologous multi-system comparisons, intersection_by_atom_index_v1 retains only candidate triples present in every replica before coordinates are read; system-specific chemical candidates remain available only through separate per-system discovery reports.",
            "Hydrogen-bond occupancy does not establish energetic or mechanistic importance.",
        ],
    }


_COMPARISON_IDENTITY_FIELDS = (
    "chain_id", "residue_number", "insertion_code", "atom_name", "altloc",
    "element",
)


def _candidate_identity_records(
    report: Mapping[str, object],
) -> List[Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]]:
    """Return donor, hydrogen, and acceptor identities for every candidate."""

    atom_dictionary = report.get("atom_dictionary")
    atoms = {
        int(row["atom_index"]): row["identity"]
        for row in atom_dictionary or []  # type: ignore[union-attr]
        if isinstance(row, dict)
        and isinstance(row.get("atom_index"), int)
        and isinstance(row.get("identity"), dict)
    } if isinstance(atom_dictionary, list) else {}
    result = []
    candidates = report.get("candidate_dictionary")
    if not isinstance(candidates, list):
        raise HydrogenBondDiscoveryError("candidate_dictionary must be an array")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise HydrogenBondDiscoveryError(
                f"candidate_dictionary[{index}] must be an object"
            )
        identities = []
        for role in ("donor", "hydrogen", "acceptor"):
            identity = candidate.get(f"{role}_identity")
            if not isinstance(identity, dict):
                identity = atoms.get(int(candidate[f"{role}_atom_index"]))
            if not isinstance(identity, dict):
                raise HydrogenBondDiscoveryError(
                    f"candidate_dictionary[{index}] lacks {role} identity"
                )
            missing = set(_COMPARISON_IDENTITY_FIELDS).difference(identity)
            if missing:
                raise HydrogenBondDiscoveryError(
                    f"candidate_dictionary[{index}] {role} identity is missing "
                    + ", ".join(sorted(missing))
                )
            identities.append(identity)
        result.append((identities[0], identities[1], identities[2]))
    return result


def _chemical_position_key(
    identities: Sequence[Mapping[str, object]],
) -> Tuple[object, ...]:
    """Map homologous atoms by chemical role and topology position, not index."""

    return tuple(
        tuple(identity[field] for field in _COMPARISON_IDENTITY_FIELDS)
        for identity in identities
    )


def _chemical_feature_id(key: Tuple[object, ...]) -> str:
    encoded = json.dumps(key, separators=(",", ":"), ensure_ascii=True)
    return "HBOND-CHEM-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _system_feature_space(
    system_id: str,
    reports: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Reduce full candidate reports within one system across its replicas."""

    first = reports[0]
    stable = (
        "candidate_dictionary", "atom_dictionary", "candidate_count",
        "frame_matrix_representation", "cutoff_definitions",
    )
    for report in reports[1:]:
        for key in stable:
            if report.get(key) != first.get(key):
                raise HydrogenBondDiscoveryError(
                    f"replica candidate dictionaries disagree within system {system_id} on {key}"
                )
    evaluated = sum(int(report.get("evaluated_frame_count", 0)) for report in reports)
    feature_count = sum(int(report.get("feature_observation_count", 0)) for report in reports)
    view: Dict[str, object] = {
        "system_id": system_id,
        "replica_ids": sorted({
            str(row.get("replica_id"))
            for report in reports
            for row in report.get("chemistry_reports", [])  # type: ignore[union-attr]
            if isinstance(row, dict) and row.get("replica_id") is not None
        }),
        "candidate_dictionary": first["candidate_dictionary"],
        "atom_dictionary": first.get("atom_dictionary"),
        "candidate_count": int(first["candidate_count"]),
        "evaluated_frame_count": evaluated,
        "planned_feature_observation_count": sum(
            int(report.get("planned_feature_observation_count", 0)) for report in reports
        ),
        "feature_observation_count": feature_count,
        "frame_bond_matrix": [
            row for report in reports for row in report.get("frame_bond_matrix", [])  # type: ignore[union-attr]
        ],
        "occupancies": [
            row for report in reports for row in report.get("occupancies", [])  # type: ignore[union-attr]
        ],
        "cutoff_occupancies": (
            None if first.get("cutoff_occupancies") is None else [
                row for report in reports
                for row in report.get("cutoff_occupancies", [])  # type: ignore[union-attr]
            ]
        ),
        "packed_cutoff_occupancy_segments": (
            None if first.get("packed_cutoff_occupancy_segments") is None else [
                row for report in reports
                for row in report.get("packed_cutoff_occupancy_segments", [])  # type: ignore[union-attr]
            ]
        ),
        "chemistry_reports": [
            row for report in reports for row in report.get("chemistry_reports", [])  # type: ignore[union-attr]
        ],
        "frame_selection": merge_frame_selection_reports([
            report["frame_selection"] for report in reports
            if isinstance(report.get("frame_selection"), dict)
        ]),
    }
    return view


def _frame_locator(frame: Mapping[str, object]) -> Dict[str, object]:
    excluded = {
        "candidate_count", "representation", "binary_values", "present_bond_ids",
        "primary_present_candidate_indices", "cutoff_present_candidate_indices",
        "present_geometry", "donor_acceptor_distances_angstrom",
        "donor_hydrogen_acceptor_angles_degrees", "packed_event_codec",
        "packed_event_count", "cutoff_ids", "packed_present_events_b64",
    }
    return {key: value for key, value in frame.items() if key not in excluded}


def _remap_comparison_frame(
    frame: Mapping[str, object],
    local_to_common: Mapping[int, int],
    common_candidates: Sequence[Mapping[str, object]],
    cutoff_definitions: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Project one full per-system frame into the shared chemical feature space."""

    candidate_count = len(common_candidates)
    locator = _frame_locator(frame)
    if frame.get("representation") == "sparse_packed_v2":
        events = unpack_sparse_present_events(frame)
        geometry = []
        cutoff_ids = frame["cutoff_ids"]
        assert isinstance(cutoff_ids, list)
        for event in events:
            local_index = int(event["candidate_index"])
            if local_index not in local_to_common:
                continue
            mask = int(event["cutoff_mask"])
            geometry.append({
                "candidate_index": local_to_common[local_index],
                "donor_acceptor_distance_angstrom": float(
                    event["donor_acceptor_distance_angstrom"]
                ),
                "donor_hydrogen_acceptor_angle_degrees": float(
                    event["donor_hydrogen_acceptor_angle_degrees"]
                ),
                "present_cutoff_ids": [
                    cutoff_id for bit, cutoff_id in enumerate(cutoff_ids)
                    if mask & (1 << bit)
                ],
            })
        geometry.sort(key=lambda row: int(row["candidate_index"]))
        return {
            **locator, "candidate_count": candidate_count,
            **pack_sparse_present_geometry(
                geometry, cutoff_definitions, candidate_count
            ),
        }
    if frame.get("representation") == "sparse_implicit_zero_v1":
        by_cutoff = frame.get("cutoff_present_candidate_indices")
        if not isinstance(by_cutoff, dict):
            raise HydrogenBondDiscoveryError("sparse frame lacks cutoff index mapping")
        remapped_by_cutoff = {
            str(cutoff_id): sorted(
                local_to_common[index] for index in indices
                if index in local_to_common
            )
            for cutoff_id, indices in by_cutoff.items()
            if isinstance(indices, list)
        }
        primary = remapped_by_cutoff.get("primary", [])
        geometry = []
        for row in frame.get("present_geometry", []):  # type: ignore[union-attr]
            if not isinstance(row, dict):
                continue
            local_index = int(row["candidate_index"])
            if local_index in local_to_common:
                geometry.append({
                    **row, "candidate_index": local_to_common[local_index]
                })
        geometry.sort(key=lambda row: int(row["candidate_index"]))
        return {
            **locator,
            "representation": "sparse_implicit_zero_v1",
            "candidate_count": candidate_count,
            "primary_present_candidate_indices": primary,
            "present_bond_ids": [
                common_candidates[index]["bond_id"] for index in primary
            ],
            "cutoff_present_candidate_indices": remapped_by_cutoff,
            "present_geometry": geometry,
        }
    binary = frame.get("binary_values")
    distances = frame.get("donor_acceptor_distances_angstrom")
    angles = frame.get("donor_hydrogen_acceptor_angles_degrees")
    if not all(isinstance(value, list) for value in (binary, distances, angles)):
        raise HydrogenBondDiscoveryError("dense frame lacks its geometry arrays")
    common_to_local = {common: local for local, common in local_to_common.items()}
    ordered_local = [common_to_local[index] for index in range(candidate_count)]
    remapped_binary = [binary[index] for index in ordered_local]  # type: ignore[index]
    return {
        **locator,
        "binary_values": remapped_binary,
        "present_bond_ids": [
            common_candidates[index]["bond_id"]
            for index, present in enumerate(remapped_binary) if present
        ],
        "donor_acceptor_distances_angstrom": [
            distances[index] for index in ordered_local  # type: ignore[index]
        ],
        "donor_hydrogen_acceptor_angles_degrees": [
            angles[index] for index in ordered_local  # type: ignore[index]
        ],
    }


def _reduce_per_system_hbond_discovery_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    """Retain full per-system chemistry and derive one shared comparison view."""

    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in (
            "module_id", "settings", "geometry_contract", "cutoff_definitions",
            "frame_matrix_representation",
        ):
            if report.get(key) != first.get(key):
                raise HydrogenBondDiscoveryError(
                    f"replica hydrogen-bond reports disagree on {key}"
                )
    by_system: Dict[str, List[Mapping[str, object]]] = {}
    for partial in partials:
        by_system.setdefault(partial.system_id, []).append(partial.value)
    system_views = [
        _system_feature_space(system_id, by_system[system_id])
        for system_id in sorted(by_system)
    ]

    unique_maps: Dict[str, Dict[Tuple[object, ...], int]] = {}
    ambiguous_maps: Dict[str, set[Tuple[object, ...]]] = {}
    identities_by_system: Dict[str, Dict[Tuple[object, ...], tuple]] = {}
    for view in system_views:
        system_id = str(view["system_id"])
        raw: Dict[Tuple[object, ...], List[int]] = {}
        identity_rows = _candidate_identity_records(view)
        identities_by_system[system_id] = {}
        for index, identities in enumerate(identity_rows):
            key = _chemical_position_key(identities)
            raw.setdefault(key, []).append(index)
            identities_by_system[system_id][key] = tuple(dict(row) for row in identities)
        unique_maps[system_id] = {
            key: indices[0] for key, indices in raw.items() if len(indices) == 1
        }
        ambiguous_maps[system_id] = {
            key for key, indices in raw.items() if len(indices) != 1
        }
    system_ids = sorted(unique_maps)
    all_keys = set().union(*(
        set(unique_maps[system_id]) | ambiguous_maps[system_id]
        for system_id in system_ids
    ))
    common_keys = sorted(
        set.intersection(*(set(unique_maps[system_id]) for system_id in system_ids)),
        key=lambda value: json.dumps(value, separators=(",", ":")),
    )
    common_index = {key: index for index, key in enumerate(common_keys)}
    common_candidates = []
    for key in common_keys:
        identities = identities_by_system[system_ids[0]][key]
        candidate_id = _chemical_feature_id(key)
        source_candidates = {}
        for view in system_views:
            system_id = str(view["system_id"])
            local = unique_maps[system_id][key]
            candidate = view["candidate_dictionary"][local]  # type: ignore[index]
            source_candidates[system_id] = {
                "candidate_index": local,
                "bond_id": candidate["bond_id"],
                "donor_atom_index": candidate["donor_atom_index"],
                "hydrogen_atom_index": candidate["hydrogen_atom_index"],
                "acceptor_atom_index": candidate["acceptor_atom_index"],
            }
        common_candidates.append({
            "bond_id": candidate_id,
            "comparison_feature_id": candidate_id,
            "identity_policy": "chemical_role_and_topology_position_v2",
            "donor_identity": identities[0],
            "hydrogen_identity": identities[1],
            "acceptor_identity": identities[2],
            "donor_atom_index": source_candidates[system_ids[0]]["donor_atom_index"],
            "hydrogen_atom_index": source_candidates[system_ids[0]]["hydrogen_atom_index"],
            "acceptor_atom_index": source_candidates[system_ids[0]]["acceptor_atom_index"],
            "source_candidates_by_system": source_candidates,
        })
    if len(system_views) == 1:
        # A one-system report has no cross-topology ambiguity. Preserve the
        # established candidate dictionary and bond identifiers exactly.
        common_candidates = list(system_views[0]["candidate_dictionary"])  # type: ignore[arg-type]

    comparative_frames: List[Dict[str, object]] = []
    comparative_occupancies: List[Dict[str, object]] = []
    comparative_cutoff_occupancies: List[Dict[str, object]] = []
    comparative_packed_counts: List[Dict[str, object]] = []
    feature_status_rows = []
    if common_candidates:
        for view in system_views:
            system_id = str(view["system_id"])
            local_to_common = {
                unique_maps[system_id][key]: common_index[key] for key in common_keys
            }
            bond_to_common = {
                str(view["candidate_dictionary"][local]["bond_id"]): common  # type: ignore[index]
                for local, common in local_to_common.items()
            }
            comparative_frames.extend(
                _remap_comparison_frame(
                    frame, local_to_common, common_candidates,
                    first["cutoff_definitions"],  # type: ignore[arg-type]
                )
                for frame in view["frame_bond_matrix"]  # type: ignore[union-attr]
            )
            for row in view["occupancies"]:  # type: ignore[union-attr]
                if row["bond_id"] in bond_to_common:
                    common = bond_to_common[str(row["bond_id"])]
                    comparative_occupancies.append({
                        **row, "bond_id": common_candidates[common]["bond_id"]
                    })
            if view["cutoff_occupancies"] is not None:
                for row in view["cutoff_occupancies"]:  # type: ignore[union-attr]
                    if row["bond_id"] in bond_to_common:
                        common = bond_to_common[str(row["bond_id"])]
                        comparative_cutoff_occupancies.append({
                            **row, "bond_id": common_candidates[common]["bond_id"]
                        })
            if view["packed_cutoff_occupancy_segments"] is not None:
                for row in view["packed_cutoff_occupancy_segments"]:  # type: ignore[union-attr]
                    counts = unpack_sparse_cutoff_counts(row)
                    remapped = [dict() for _ in first["cutoff_definitions"]]  # type: ignore[arg-type]
                    for value in counts:
                        local = int(value["candidate_index"])
                        if local in local_to_common:
                            remapped[int(value["cutoff_index"])][
                                local_to_common[local]
                            ] = int(value["present_frame_count"])
                    comparative_packed_counts.append({
                        **{
                            key: row[key] for key in (
                                "system_id", "replica_id", "segment_id"
                            )
                        },
                        **pack_sparse_cutoff_counts(
                            remapped,
                            first["cutoff_definitions"],  # type: ignore[arg-type]
                            len(common_candidates),
                            int(row["evaluated_frame_count"]),
                        ),
                    })

    observed_by_system = {
        str(view["system_id"]): {
            str(row["bond_id"]) for row in view["occupancies"]  # type: ignore[union-attr]
            if int(row.get("present_frame_count", 0)) > 0
        }
        for view in system_views
    }
    for key in sorted(
        all_keys, key=lambda value: json.dumps(value, separators=(",", ":"))
    ):
        identity_source = next(
            identities_by_system[system_id][key]
            for system_id in system_ids if key in identities_by_system[system_id]
        )
        statuses = {}
        for view in system_views:
            system_id = str(view["system_id"])
            if key in ambiguous_maps[system_id]:
                status = "unmappable_nonunique_identity"
            elif key not in unique_maps[system_id]:
                status = "chemically_absent"
            else:
                local = unique_maps[system_id][key]
                bond_id = str(view["candidate_dictionary"][local]["bond_id"])  # type: ignore[index]
                status = (
                    "observed" if bond_id in observed_by_system[system_id]
                    else "chemically_present_never_observed"
                )
            statuses[system_id] = status
        feature_status_rows.append({
            "comparison_feature_id": _chemical_feature_id(key),
            "donor_identity": identity_source[0],
            "hydrogen_identity": identity_source[1],
            "acceptor_identity": identity_source[2],
            "status_by_system": statuses,
            "comparable_across_all_systems": key in common_index,
        })

    frame_selection = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    evaluated = sum(int(view["evaluated_frame_count"]) for view in system_views)
    feature_count = sum(int(view["feature_observation_count"]) for view in system_views)
    maximum = int(first["settings"]["maximum_feature_observations"])  # type: ignore[index]
    if feature_count > maximum:
        raise HydrogenBondDiscoveryError(
            "parallel full per-system hydrogen-bond feature count exceeds "
            f"maximum_feature_observations: {feature_count} > {maximum}"
        )
    issues = [
        issue for issue in unique_issues(reports)
        if issue.get("code") != "FRAME_SUBSAMPLING"
    ]
    issues.append({
        "severity": "info",
        "code": "HBOND_FULL_PER_SYSTEM_CANDIDATES_RETAINED",
        "location": source_context["project_manifest_path"],
        "message": (
            f"Evaluated every topology-defined candidate in {len(system_views)} systems; "
            f"the shared chemical-position view contains {len(common_candidates)} features."
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
    first.update({
        "frame_selection": frame_selection,
        "candidate_harmonization": {
            "policy": "per_system_full_with_chemical_identity_comparison_v2",
            "selection_basis": (
                "All per-system candidates are retained. Shared features are the "
                "outcome-independent intersection of unique donor, hydrogen, and "
                "acceptor chemical-role/topology-position identities; raw atom indices "
                "are never compared across systems."
            ),
            "system_candidate_counts": {
                str(view["system_id"]): int(view["candidate_count"])
                for view in system_views
            },
            "common_candidate_count": len(common_candidates),
            "union_candidate_count": len(all_keys),
            "unmappable_identity_count": len(set().union(*ambiguous_maps.values())),
            "feature_statuses": feature_status_rows,
        },
        "system_feature_spaces": system_views,
        "candidate_dictionary": common_candidates,
        "atom_dictionary": (
            system_views[0]["atom_dictionary"] if len(system_views) == 1 else None
        ),
        "candidate_count": len(common_candidates),
        "planned_feature_observation_count": feature_count,
        "evaluated_frame_count": evaluated,
        "feature_observation_count": feature_count,
        "frame_bond_matrix": comparative_frames,
        "occupancies": comparative_occupancies,
        "cutoff_occupancies": (
            None if first.get("cutoff_occupancies") is None
            else comparative_cutoff_occupancies
        ),
        "packed_cutoff_occupancy_segments": (
            comparative_packed_counts
            if first.get("packed_cutoff_occupancy_segments") is not None else None
        ),
        "chemistry_reports": [
            row for view in system_views
            for row in view["chemistry_reports"]  # type: ignore[union-attr]
        ],
        "observation_accounting": {
            "source_physical_frame_count": int(frame_selection["source_frame_count"]),
            "selected_physical_frame_count": evaluated,
            "symmetry_expanded_observation_count": evaluated,
            "candidate_frame_feature_observation_count": feature_count,
            "per_system_candidate_frame_feature_observation_counts": {
                str(view["system_id"]): int(view["feature_observation_count"])
                for view in system_views
            },
            "subsampling_triggered": evaluated < int(frame_selection["source_frame_count"]),
        },
        "issues": issues,
        "error_count": sum(issue.get("severity") == "error" for issue in issues),
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "limitations": [
            "Automatic chemistry cannot repair missing or incorrect protonation, formal charge, bond order, or tautomer state.",
            "Chemical-position mapping assumes declared chain and residue numbering represent homologous positions; explicit homolog mappings remain necessary when numbering differs.",
            "Chemically absent, chemically present but never observed, observed, and nonunique unmappable features are reported separately.",
            "Hydrogen-bond occupancy does not establish energy, affinity, causality, or mechanism.",
        ],
    })
    restore_source_provenance(first, source_context)
    return first


def _reduce_common_hbond_discovery_reports(
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
    if harmonization.get("policy") == "intersection_by_atom_index_v1":
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


def _reduce_hbond_discovery_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    policy = partials[0].value.get("settings", {}).get(  # type: ignore[union-attr]
        "candidate_harmonization"
    )
    if policy == "per_system_full_with_chemical_identity_comparison_v2":
        return _reduce_per_system_hbond_discovery_reports(partials, source_context)
    return _reduce_common_hbond_discovery_reports(partials, source_context)


def hydrogen_bond_discovery_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Discover chemistry globally, evaluate replicas in parallel, then reduce."""

    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    selection = settings.get("frame_selection")
    keys = None
    harmonization = None
    if (
        settings["mode"] == "automatic"
        and settings["candidate_harmonization"] == "intersection_by_atom_index_v1"
    ):
        context = compile_project_context_file(source, hash_content=hash_content)
        system_path = Path(context["system_manifest_path"])
        keys, harmonization = _automatic_candidate_intersection(
            load_json(system_path), system_path, settings
        )
    return execute_replica_final_module(
        project_path,
        runner_id="hydrogen_bond_discovery",
        hash_content=hash_content,
        reducer=_reduce_hbond_discovery_reports,
        worker_payload={
            "harmonized_candidate_keys": (
                [list(row) for row in sorted(keys)] if keys is not None else None
            ),
            "candidate_harmonization_report": harmonization,
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
