"""Outcome-independent conformational views inferred from topology chemistry."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomRecord
from .chemical_identity import (
    ION_RESIDUES,
    STANDARD_NUCLEIC_RESIDUES,
    WATER_RESIDUES,
)
from .oligomer_symmetry import (
    OligomerSymmetryError,
    plan_equivalent_oligomer_members,
    restrict_equivalent_member_plan,
    validate_member_plan,
)
from .selections import select_atoms


ResidueKey = Tuple[str, int, str]

def _key(atom: AtomRecord) -> ResidueKey:
    return atom.chain_id, atom.residue_number, atom.insertion_code


def _key_record(key: ResidueKey) -> Dict[str, object]:
    return {
        "chain_id": key[0],
        "residue_number": key[1],
        "insertion_code": key[2],
    }


def _heavy(atom: AtomRecord) -> bool:
    return not (
        atom.element.upper() == "H"
        or atom.atom_name.lstrip("0123456789").upper().startswith("H")
    )


def plan_conformational_views(
    atoms: Sequence[AtomRecord],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    *,
    interface_distance_angstrom: float = 6.0,
    modified_nucleotide_neighbor_span: int = 1,
) -> Dict[str, object]:
    """Classify a reference topology and declare prespecified PCA views.

    The reference structure alone determines every view. No trajectory coordinate,
    occupancy, PCA score, or condition label is used.
    """

    if len(atoms) != len(coordinates_angstrom):
        raise ValueError("topology and reference coordinate atom counts differ")
    if interface_distance_angstrom <= 0.0:
        raise ValueError("interface_distance_angstrom must be positive")
    if modified_nucleotide_neighbor_span < 0:
        raise ValueError("modified_nucleotide_neighbor_span must be nonnegative")
    residues: Dict[ResidueKey, List[AtomRecord]] = defaultdict(list)
    coordinate_by_index = {}
    for atom, coordinate in zip(atoms, coordinates_angstrom):
        residues[_key(atom)].append(atom)
        coordinate_by_index[atom.atom_index] = coordinate

    protein: List[ResidueKey] = []
    nucleic: List[ResidueKey] = []
    modified: List[ResidueKey] = []
    other_solute: List[ResidueKey] = []
    water_count = 0
    ion_count = 0
    for key, residue_atoms in residues.items():
        names = {atom.atom_name.upper() for atom in residue_atoms}
        residue_name = residue_atoms[0].residue_name.upper()
        if residue_name in WATER_RESIDUES:
            water_count += 1
        elif residue_name in ION_RESIDUES:
            ion_count += 1
        elif {"N", "CA", "C"}.issubset(names):
            protein.append(key)
        elif "C1'" in names and ({"P", "O4'", "C4'"} & names):
            nucleic.append(key)
            if residue_name not in STANDARD_NUCLEIC_RESIDUES:
                modified.append(key)
        else:
            other_solute.append(key)

    protein = sorted(protein)
    nucleic = sorted(nucleic)
    modified = sorted(modified)
    interface_definition: Dict[str, object] | None = None
    if protein and nucleic:
        macromolecular_class = "protein_nucleic_acid_complex"
        system_class = (
            macromolecular_class
            if not other_solute
            else "protein_nucleic_acid_other_solute_complex"
        )
    elif protein:
        macromolecular_class = "protein_only"
        system_class = (
            macromolecular_class
            if not other_solute
            else "protein_other_solute_complex"
        )
    elif nucleic:
        macromolecular_class = "nucleic_acid_only"
        system_class = (
            macromolecular_class
            if not other_solute
            else "nucleic_acid_other_solute_complex"
        )
    else:
        macromolecular_class = "other_or_ambiguous"
        system_class = macromolecular_class

    views: List[Dict[str, object]] = []
    global_definition = {"preset": "solute_heavy"}
    global_count = len(select_atoms(atoms, global_definition, "global_common_heavy"))
    views.append({
        "view_id": "global_common_heavy",
        "role": "primary_global",
        "selection_id": "global_common_heavy",
        "selection": global_definition,
        "alignment_selection_id": "alignment",
        "atom_count_in_reference": global_count,
        "applicability": "automatic",
        "rationale": "global shared non-hydrogen solute conformation",
    })

    if protein and nucleic:
        focus = set()
        if modified:
            by_chain: Dict[str, List[ResidueKey]] = defaultdict(list)
            for key in nucleic:
                by_chain[key[0]].append(key)
            for key in modified:
                chain_order = by_chain[key[0]]
                center = chain_order.index(key)
                lower = max(0, center - modified_nucleotide_neighbor_span)
                upper = min(len(chain_order), center + modified_nucleotide_neighbor_span + 1)
                focus.update(chain_order[lower:upper])
            focus_basis = "modified nucleotides plus sequence neighbors"
        else:
            focus.update(nucleic)
            focus_basis = "all detected nucleic-acid residues; no modified nucleotide identified"
        focus_atoms = [
            atom for key in focus for atom in residues[key] if _heavy(atom)
        ]
        cutoff2 = interface_distance_angstrom * interface_distance_angstrom
        contact_protein = set()
        for key in protein:
            for atom in residues[key]:
                if not _heavy(atom):
                    continue
                x, y, z = coordinate_by_index[atom.atom_index]
                if any(
                    (x - coordinate_by_index[target.atom_index][0]) ** 2
                    + (y - coordinate_by_index[target.atom_index][1]) ** 2
                    + (z - coordinate_by_index[target.atom_index][2]) ** 2
                    <= cutoff2
                    for target in focus_atoms
                ):
                    contact_protein.add(key)
                    break
        interface_keys = sorted(focus | contact_protein)
        interface_definition = {
            "residue_keys": [_key_record(key) for key in interface_keys],
            "heavy_only": True,
        }
        interface_count = len(
            select_atoms(atoms, interface_definition, "chemical_interface")
        )
        views.append({
            "view_id": "chemical_interface",
            "role": "primary_local",
            "selection_id": "chemical_interface",
            "selection": interface_definition,
            "alignment_selection_id": "protein_alignment",
            "atom_count_in_reference": interface_count,
            "applicability": "automatic",
            "rationale": (
                f"{focus_basis}; complete contacting protein residues selected from the "
                f"reference structure at {interface_distance_angstrom:g} angstrom"
            ),
            "modified_nucleotide_residue_keys": [
                _key_record(key) for key in modified
            ],
            "focus_nucleic_residue_keys": [_key_record(key) for key in sorted(focus)],
            "contacting_protein_residue_keys": [
                _key_record(key) for key in sorted(contact_protein)
            ],
            "bound_ion_policy": (
                "ions are excluded from Cartesian PCA; analyze coordination and ion-distance "
                "features separately unless conserved bound-ion identity is explicitly locked"
            ),
        })

    trace_definition = {"preset": "complex_trace"}
    trace_count = len(select_atoms(atoms, trace_definition, "macromolecular_trace"))
    views.append({
        "view_id": "macromolecular_trace",
        "role": "secondary_sensitivity",
        "selection_id": "macromolecular_trace",
        "selection": trace_definition,
        "alignment_selection_id": "alignment",
        "atom_count_in_reference": trace_count,
        "applicability": "automatic",
        "rationale": "coarse C-alpha and nucleic-acid C1-prime robustness view",
    })

    try:
        oligomer = plan_equivalent_oligomer_members(atoms, coordinates_angstrom)
    except OligomerSymmetryError as exc:
        oligomer = {
            "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
            "applicable": False,
            "reason": str(exc),
        }
    if oligomer["applicable"]:
        views.append({
            "view_id": "oligomer_member_common_heavy",
            "role": "symmetry_expanded_member",
            "selection_id": "oligomer_member_common_heavy",
            "alignment_selection_id": "oligomer_member_alignment",
            "atom_count_in_reference": oligomer["analysis_atom_count_per_member"],
            "applicability": "automatic_strict_equivalent_oligomer",
            "rationale": (
                "equivalent protein-centered oligomer members are independently "
                "aligned to one canonical member and pooled with member provenance"
            ),
            "symmetry_expansion": oligomer,
            "observation_contract": oligomer["observation_contract"],
            "paired_member_correlation": (
                "within-frame PCA-score cross-correlation for every unordered member pair"
            ),
        })
        if interface_definition is not None:
            interface_atoms = select_atoms(
                atoms, interface_definition, "chemical_interface"
            )
            try:
                member_interface = restrict_equivalent_member_plan(
                    oligomer,
                    [atom.atom_index for atom in interface_atoms],
                    selection_id="chemical_interface",
                )
            except OligomerSymmetryError as exc:
                member_interface = {
                    "planning_schema": "salsbury-equivalent-oligomer-plan-v1",
                    "applicable": False,
                    "reason": str(exc),
                }
            if member_interface.get("applicable") is True:
                views.append({
                    "view_id": "oligomer_member_interface_common_heavy",
                    "role": "symmetry_expanded_member_interface",
                    "selection_id": "oligomer_member_interface_common_heavy",
                    "alignment_selection_id": "oligomer_member_alignment",
                    "atom_count_in_reference": member_interface[
                        "analysis_atom_count_per_member"
                    ],
                    "applicability": "automatic_strict_equivalent_oligomer_interface",
                    "rationale": (
                        "topology-derived interface-heavy canonical positions are "
                        "symmetrized across equivalent members, independently aligned, "
                        "and pooled as observations of one system"
                    ),
                    "symmetry_expansion": member_interface,
                    "observation_contract": member_interface["observation_contract"],
                    "paired_member_correlation": (
                        "within-frame interface-PCA score cross-correlation for every "
                        "unordered member pair"
                    ),
                })

    return {
        "planning_schema": "salsbury-conformational-view-plan-v1",
        "selection_basis": (
            "reference topology, reference coordinates, residue chemistry, and declared "
            "distance only; no trajectory outcomes are inspected"
        ),
        "system_classification": system_class,
        "macromolecular_classification": macromolecular_class,
        "composition": {
            "protein_residue_count": len(protein),
            "nucleic_acid_residue_count": len(nucleic),
            "modified_nucleotide_count": len(modified),
            "other_solute_residue_count": len(other_solute),
            "water_residue_count": water_count,
            "ion_residue_count": ion_count,
        },
        "views": views,
        "equivalent_oligomer": oligomer,
        "comparison_contract": (
            "each view has its own PCA components, FES minima, and cluster labels; compare "
            "conditions within a view and cross-tabulate frame assignments across views"
        ),
    }


def _residue_classes(
    atoms: Sequence[AtomRecord],
) -> Tuple[Dict[ResidueKey, List[AtomRecord]], List[ResidueKey], List[ResidueKey]]:
    residues: Dict[ResidueKey, List[AtomRecord]] = defaultdict(list)
    for atom in atoms:
        residues[_key(atom)].append(atom)
    protein = []
    nucleic = []
    for key, residue_atoms in residues.items():
        names = {atom.atom_name.upper() for atom in residue_atoms}
        if {"N", "CA", "C"}.issubset(names):
            protein.append(key)
        elif "C1'" in names and ({"P", "O4'", "C4'"} & names):
            nucleic.append(key)
    return residues, sorted(protein), sorted(nucleic)


def _common_selected_atom_count(
    references: Sequence[Tuple[str, Sequence[AtomRecord], Sequence[Tuple[float, float, float]]]],
    selection: Mapping[str, object],
    policy: str,
) -> Tuple[int, Dict[str, int]]:
    selected_keys = []
    counts = {}
    for reference_id, atoms, _ in references:
        selected = select_atoms(atoms, selection, reference_id)
        keys = [atom.match_key(policy) for atom in selected]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"{reference_id} has duplicate selected atom identities under {policy} mapping"
            )
        selected_keys.append(set(keys))
        counts[reference_id] = len(keys)
    common = set.intersection(*selected_keys)
    if not common:
        raise ValueError("comparative conformational view has no common selected atoms")
    return len(common), counts


def plan_comparative_conformational_views(
    references: Sequence[
        Tuple[
            str,
            Sequence[AtomRecord],
            Sequence[Tuple[float, float, float]],
        ]
    ],
    *,
    common_atom_policy: str = "position",
    interface_distance_angstrom: float = 6.0,
    modified_nucleotide_neighbor_span: int = 1,
) -> Dict[str, object]:
    """Declare one prespecified set of conformational views across conditions.

    Reference topology and coordinates from every declared condition are
    inspected before trajectory outcomes. Modified-residue positions are
    harmonized across topologies, and reference-contacting protein residues are
    unioned so one condition cannot define a narrower comparison for another.
    """

    if not references:
        raise ValueError("at least one comparative reference is required")
    if common_atom_policy not in {"strict", "position"}:
        raise ValueError("common_atom_policy must be strict or position")
    ids = [reference[0] for reference in references]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("comparative reference IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("comparative reference IDs must be unique")
    if interface_distance_angstrom <= 0.0:
        raise ValueError("interface_distance_angstrom must be positive")
    if modified_nucleotide_neighbor_span < 0:
        raise ValueError("modified_nucleotide_neighbor_span must be nonnegative")

    individual = {
        reference_id: plan_conformational_views(
            atoms,
            coordinates,
            interface_distance_angstrom=interface_distance_angstrom,
            modified_nucleotide_neighbor_span=modified_nucleotide_neighbor_span,
        )
        for reference_id, atoms, coordinates in references
    }
    macromolecular_classifications = {
        str(report["macromolecular_classification"])
        for report in individual.values()
    }
    if len(macromolecular_classifications) != 1:
        raise ValueError(
            "comparative references do not share one macromolecular composition class"
        )
    macromolecular_classification = next(iter(macromolecular_classifications))
    system_classifications_by_reference = {
        reference_id: str(report["system_classification"])
        for reference_id, report in individual.items()
    }
    views = []
    global_selection = {"preset": "solute_heavy"}
    global_common, global_counts = _common_selected_atom_count(
        references, global_selection, common_atom_policy
    )
    views.append({
        "view_id": "global_common_heavy",
        "role": "primary_global",
        "selection_id": "global_common_heavy",
        "selection": global_selection,
        "alignment_selection_id": "alignment",
        "atom_count_in_reference": global_common,
        "common_atom_count": global_common,
        "atom_counts_by_reference": global_counts,
        "applicability": "automatic_comparative",
        "rationale": "global shared non-hydrogen solute conformation",
    })

    interface_selection: Dict[str, object] | None = None
    if macromolecular_classification == "protein_nucleic_acid_complex":
        class_rows = {}
        modified_positions = set()
        for reference_id, atoms, coordinates in references:
            if len(atoms) != len(coordinates):
                raise ValueError(
                    f"{reference_id} topology and reference coordinate atom counts differ"
                )
            residues, protein, nucleic = _residue_classes(atoms)
            coordinate_by_index = {
                atom.atom_index: coordinate
                for atom, coordinate in zip(atoms, coordinates)
            }
            class_rows[reference_id] = (
                residues, protein, nucleic, coordinate_by_index
            )
            for key in nucleic:
                residue_name = residues[key][0].residue_name.upper()
                if residue_name not in STANDARD_NUCLEIC_RESIDUES:
                    modified_positions.add(key)

        focus = set()
        if modified_positions:
            for _, (_, _, nucleic, _) in class_rows.items():
                by_chain: Dict[str, List[ResidueKey]] = defaultdict(list)
                for key in nucleic:
                    by_chain[key[0]].append(key)
                for chain in by_chain:
                    by_chain[chain].sort()
                for modified_key in modified_positions:
                    chain_order = by_chain.get(modified_key[0], [])
                    if modified_key not in chain_order:
                        continue
                    center = chain_order.index(modified_key)
                    lower = max(0, center - modified_nucleotide_neighbor_span)
                    upper = min(
                        len(chain_order),
                        center + modified_nucleotide_neighbor_span + 1,
                    )
                    focus.update(chain_order[lower:upper])
            focus_basis = (
                "modified-nucleotide positions detected in any declared reference plus "
                "sequence neighbors, harmonized before trajectory evaluation"
            )
        else:
            for _, (_, _, nucleic, _) in class_rows.items():
                focus.update(nucleic)
            focus_basis = (
                "all detected nucleic-acid residues; no modified nucleotide identified "
                "in any declared reference"
            )

        contact_by_reference = {}
        contact_union = set()
        cutoff2 = interface_distance_angstrom * interface_distance_angstrom
        for reference_id, (residues, protein, _, coordinate_by_index) in class_rows.items():
            focus_atoms = [
                atom
                for key in focus
                if key in residues
                for atom in residues[key]
                if _heavy(atom)
            ]
            contacts = set()
            for key in protein:
                for atom in residues[key]:
                    if not _heavy(atom):
                        continue
                    x, y, z = coordinate_by_index[atom.atom_index]
                    if any(
                        (x - coordinate_by_index[target.atom_index][0]) ** 2
                        + (y - coordinate_by_index[target.atom_index][1]) ** 2
                        + (z - coordinate_by_index[target.atom_index][2]) ** 2
                        <= cutoff2
                        for target in focus_atoms
                    ):
                        contacts.add(key)
                        break
            contact_by_reference[reference_id] = [
                _key_record(key) for key in sorted(contacts)
            ]
            contact_union.update(contacts)
        interface_keys = sorted(focus | contact_union)
        interface_selection = {
            "residue_keys": [_key_record(key) for key in interface_keys],
            "heavy_only": True,
        }
        interface_common, interface_counts = _common_selected_atom_count(
            references, interface_selection, common_atom_policy
        )
        views.append({
            "view_id": "chemical_interface",
            "role": "primary_local",
            "selection_id": "chemical_interface",
            "selection": interface_selection,
            "alignment_selection_id": "protein_alignment",
            "atom_count_in_reference": interface_common,
            "common_atom_count": interface_common,
            "atom_counts_by_reference": interface_counts,
            "applicability": "automatic_comparative",
            "rationale": (
                f"{focus_basis}; union of complete protein residues contacting that "
                f"focus within {interface_distance_angstrom:g} angstrom in any declared "
                "reference structure"
            ),
            "modified_nucleotide_residue_keys": [
                _key_record(key) for key in sorted(modified_positions)
            ],
            "focus_nucleic_residue_keys": [
                _key_record(key) for key in sorted(focus)
            ],
            "contacting_protein_residue_keys": [
                _key_record(key) for key in sorted(contact_union)
            ],
            "contacting_protein_residue_keys_by_reference": contact_by_reference,
            "bound_ion_policy": (
                "ions are excluded from Cartesian PCA; analyze coordination and ion-distance "
                "features separately unless conserved bound-ion identity is explicitly locked"
            ),
        })

    trace_selection = {"preset": "complex_trace"}
    trace_common, trace_counts = _common_selected_atom_count(
        references, trace_selection, common_atom_policy
    )
    views.append({
        "view_id": "macromolecular_trace",
        "role": "secondary_sensitivity",
        "selection_id": "macromolecular_trace",
        "selection": trace_selection,
        "alignment_selection_id": "alignment",
        "atom_count_in_reference": trace_common,
        "common_atom_count": trace_common,
        "atom_counts_by_reference": trace_counts,
        "applicability": "automatic_comparative",
        "rationale": "coarse C-alpha and nucleic-acid C1-prime robustness view",
    })
    comparative_oligomer: Dict[str, object]
    oligomer_plans = [individual[reference_id]["equivalent_oligomer"] for reference_id in ids]
    if all(isinstance(plan, dict) and plan.get("applicable") is True for plan in oligomer_plans):
        first_plan = oligomer_plans[0]
        assert isinstance(first_plan, dict)
        member_counts = {int(plan["member_count"]) for plan in oligomer_plans}  # type: ignore[index]
        member_kinds = {str(plan["member_kind"]) for plan in oligomer_plans}  # type: ignore[index]
        try:
            native_resolved = {
                reference_id: validate_member_plan(
                    atoms, first_plan, policy=common_atom_policy
                )
                for reference_id, atoms, _ in references
            }
            analysis_key_sets = [
                {tuple(key) for key in row["analysis_identity_keys"]}
                for row in native_resolved.values()
            ]
            alignment_key_sets = [
                {tuple(key) for key in row["alignment_identity_keys"]}
                for row in native_resolved.values()
            ]
            common_analysis_keys = sorted(set.intersection(*analysis_key_sets))
            common_alignment_keys = sorted(set.intersection(*alignment_key_sets))
            if not common_analysis_keys or len(common_alignment_keys) < 3:
                raise OligomerSymmetryError(
                    "comparative equivalent members have insufficient common atoms"
                )
            shared_plan = deepcopy(first_plan)
            shared_plan.pop("analysis_position_indices", None)
            shared_plan["analysis_identity_keys"] = [
                list(key) for key in common_analysis_keys
            ]
            shared_plan["alignment_identity_keys"] = [
                list(key) for key in common_alignment_keys
            ]
            first_reference_id, first_atoms, _ = references[0]
            first_resolved = validate_member_plan(
                first_atoms, shared_plan, policy=common_atom_policy
            )
            stored_members = shared_plan.get("members")
            if not isinstance(stored_members, list):
                raise OligomerSymmetryError(
                    "comparative equivalent-member plan has no members"
                )
            for stored, resolved_member in zip(
                stored_members, first_resolved["members"]
            ):
                if not isinstance(stored, dict):
                    raise OligomerSymmetryError(
                        "comparative equivalent-member record is invalid"
                    )
                stored["analysis_atom_indices"] = list(
                    resolved_member["analysis_atom_indices"]
                )
                stored["alignment_atom_indices"] = list(
                    resolved_member["alignment_atom_indices"]
                )
            shared_plan["analysis_atom_count_per_member"] = len(
                common_analysis_keys
            )
            shared_plan["alignment_atom_count_per_member"] = len(
                common_alignment_keys
            )
            resolved = {
                reference_id: validate_member_plan(
                    atoms, shared_plan, policy=common_atom_policy
                )
                for reference_id, atoms, _ in references
            }
        except OligomerSymmetryError as exc:
            comparative_oligomer = {
                "applicable": False,
                "reason": f"member mapping differs across comparative references: {exc}",
            }
        else:
            atom_counts = {
                reference_id: int(row["analysis_atom_count_per_member"])
                for reference_id, row in resolved.items()
            }
            if len(member_counts) != 1 or len(member_kinds) != 1:
                comparative_oligomer = {
                    "applicable": False,
                    "reason": "comparative references do not share one equivalent-member contract",
                    "member_counts": sorted(member_counts),
                    "analysis_atom_counts_by_reference": atom_counts,
                }
            else:
                comparative_oligomer = {
                    "applicable": True,
                    "mapping_policy": common_atom_policy,
                    "analysis_atom_counts_by_reference": atom_counts,
                    "native_analysis_atom_counts_by_reference": {
                        reference_id: int(row["analysis_atom_count_per_member"])
                        for reference_id, row in native_resolved.items()
                    },
                    "symmetry_expansion": shared_plan,
                }
                views.append({
                    "view_id": "oligomer_member_common_heavy",
                    "role": "symmetry_expanded_member",
                    "selection_id": "oligomer_member_common_heavy",
                    "alignment_selection_id": "oligomer_member_alignment",
                    "atom_count_in_reference": next(iter(atom_counts.values())),
                    "common_atom_count": next(iter(atom_counts.values())),
                    "atom_counts_by_reference": atom_counts,
                    "applicability": "automatic_comparative_strict_equivalent_oligomer",
                    "rationale": (
                        "equivalent members in every condition are independently aligned "
                        "and projected on one comparative canonical-member basis"
                    ),
                    "symmetry_expansion": first_plan,
                    "observation_contract": shared_plan["observation_contract"],
                })
                if interface_selection is not None:
                    first_reference_id, first_atoms, _ = references[0]
                    interface_atom_indices = [
                        atom.atom_index for atom in select_atoms(
                            first_atoms, interface_selection, first_reference_id
                        )
                    ]
                    try:
                        member_interface_plan = restrict_equivalent_member_plan(
                            shared_plan,
                            interface_atom_indices,
                            selection_id="chemical_interface",
                        )
                        resolved_interface = {
                            reference_id: validate_member_plan(
                                atoms,
                                member_interface_plan,
                                policy=common_atom_policy,
                            )
                            for reference_id, atoms, _ in references
                        }
                    except OligomerSymmetryError as exc:
                        comparative_oligomer["interface_applicable"] = False
                        comparative_oligomer["interface_reason"] = str(exc)
                    else:
                        interface_atom_counts = {
                            reference_id: int(row["analysis_atom_count_per_member"])
                            for reference_id, row in resolved_interface.items()
                        }
                        if len(set(interface_atom_counts.values())) != 1:
                            comparative_oligomer["interface_applicable"] = False
                            comparative_oligomer["interface_reason"] = (
                                "comparative references do not share one equivalent-"
                                "member interface atom count"
                            )
                            comparative_oligomer[
                                "interface_analysis_atom_counts_by_reference"
                            ] = interface_atom_counts
                        else:
                            comparative_oligomer.update({
                                "interface_applicable": True,
                                "interface_analysis_atom_counts_by_reference": (
                                    interface_atom_counts
                                ),
                                "interface_symmetry_expansion": member_interface_plan,
                            })
                            interface_atom_count = next(
                                iter(interface_atom_counts.values())
                            )
                            views.append({
                                "view_id": (
                                    "oligomer_member_interface_common_heavy"
                                ),
                                "role": "symmetry_expanded_member_interface",
                                "selection_id": (
                                    "oligomer_member_interface_common_heavy"
                                ),
                                "alignment_selection_id": (
                                    "oligomer_member_alignment"
                                ),
                                "atom_count_in_reference": interface_atom_count,
                                "common_atom_count": interface_atom_count,
                                "atom_counts_by_reference": interface_atom_counts,
                                "applicability": (
                                    "automatic_comparative_strict_equivalent_"
                                    "oligomer_interface"
                                ),
                                "rationale": (
                                    "the shared topology-derived interface is "
                                    "restricted to equivalent-member positions in "
                                    "every condition; members are independently "
                                    "aligned and projected on one comparative basis"
                                ),
                                "symmetry_expansion": member_interface_plan,
                                "observation_contract": member_interface_plan[
                                    "observation_contract"
                                ],
                            })
    else:
        comparative_oligomer = {
            "applicable": False,
            "reason": "not every comparative reference has a strict equivalent-oligomer plan",
        }
    return {
        "planning_schema": "salsbury-comparative-conformational-view-plan-v1",
        "selection_basis": (
            "all declared reference topologies, reference coordinates, residue chemistry, "
            "and declared distance only; no trajectory outcomes are inspected"
        ),
        "reference_ids": ids,
        "common_atom_policy": common_atom_policy,
        "system_classification": macromolecular_classification,
        "macromolecular_classification": macromolecular_classification,
        "system_classifications_by_reference": system_classifications_by_reference,
        "composition_by_reference": {
            reference_id: individual[reference_id]["composition"]
            for reference_id in ids
        },
        "views": views,
        "equivalent_oligomer": comparative_oligomer,
        "comparison_contract": (
            "each view has its own PCA components, FES minima, and cluster labels; compare "
            "conditions within a view and cross-tabulate frame assignments across views"
        ),
    }
