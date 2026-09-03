"""Automatic, topology-backed chemical perception for direct hydrogen bonds.

The module deliberately separates *chemical eligibility* from the geometric
hydrogen-bond rule.  Standard protein and nucleic-acid residues use explicit
atom-name templates.  Unknown covalently connected residues fall back to a
conservative N/O/S rule and are reported as provisional rather than silently
treated as equivalent to templated chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomRecord
from .chemical_identity import (
    ION_RESIDUES,
    NUCLEIC_RESIDUES,
    PROTEIN_RESIDUES,
    WATER_RESIDUES,
)


Bond = Tuple[int, int]

_PROTEIN_ACCEPTORS: Mapping[str, frozenset[str]] = {
    "ARG": frozenset({"O", "OXT"}),
    "ASN": frozenset({"O", "OXT", "OD1"}),
    "ASP": frozenset({"O", "OXT", "OD1", "OD2"}),
    "ASH": frozenset({"O", "OXT", "OD1", "OD2"}),
    "CYS": frozenset({"O", "OXT", "SG"}),
    "CYM": frozenset({"O", "OXT", "SG"}),
    "CYX": frozenset({"O", "OXT", "SG"}),
    "GLN": frozenset({"O", "OXT", "OE1"}),
    "GLU": frozenset({"O", "OXT", "OE1", "OE2"}),
    "GLH": frozenset({"O", "OXT", "OE1", "OE2"}),
    "HIS": frozenset({"O", "OXT"}),
    "HID": frozenset({"O", "OXT"}),
    "HIE": frozenset({"O", "OXT"}),
    "HIP": frozenset({"O", "OXT"}),
    "MET": frozenset({"O", "OXT", "SD"}),
    "SER": frozenset({"O", "OXT", "OG"}),
    "THR": frozenset({"O", "OXT", "OG1"}),
    "TYR": frozenset({"O", "OXT", "OH"}),
}
_PROTEIN_SIDECHAIN_DONORS: Mapping[str, frozenset[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "CYS": frozenset({"SG"}),
    "CYM": frozenset({"SG"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "HID": frozenset({"ND1"}),
    "HIE": frozenset({"NE2"}),
    "HIP": frozenset({"ND1", "NE2"}),
    "LYS": frozenset({"NZ"}),
    "LYN": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
}
_NUCLEIC_BASES = {
    "A": "A", "DA": "A", "RA": "A", "ADE": "A",
    "C": "C", "DC": "C", "RC": "C", "CYT": "C", "5MC": "C",
    "G": "G", "DG": "G", "RG": "G", "GUA": "G",
    "T": "T", "DT": "T", "THY": "T",
    "U": "U", "DU": "U", "RU": "U", "URA": "U", "PSU": "U",
    "8OG": "8OG", "8OX": "8OG", "OX3": "8OG", "EDU": "U",
    "FDU": "U",
}
_NUCLEIC_ACCEPTORS: Mapping[str, frozenset[str]] = {
    "A": frozenset({"N1", "N3", "N7"}),
    "C": frozenset({"N3", "O2"}),
    "G": frozenset({"N3", "N7", "O6"}),
    "T": frozenset({"O2", "O4"}),
    "U": frozenset({"O2", "O4"}),
    # 8-oxoG is represented as the usual guanine base edge plus O8.  N7 is
    # protonated in this 7,8-dihydro tautomer and is therefore not an acceptor.
    "8OG": frozenset({"N3", "O6", "O8"}),
}
_NUCLEIC_DONORS: Mapping[str, frozenset[str]] = {
    "A": frozenset({"N6"}),
    "C": frozenset({"N4"}),
    "G": frozenset({"N1", "N2"}),
    "T": frozenset({"N3"}),
    "U": frozenset({"N3"}),
    # N7-H is the lesion-specific donor; all donor assignments still require
    # an explicitly bonded hydrogen from the supplied connectivity graph.
    "8OG": frozenset({"N1", "N2", "N7"}),
}
_NUCLEIC_SUGAR_ACCEPTORS = frozenset({
    "O2'", "O2*", "O3'", "O3*", "O4'", "O4*", "O5'", "O5*",
    "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
})


@dataclass(frozen=True)
class AtomChemicalRole:
    """One atom's automatically inferred direct-hydrogen-bond eligibility."""

    entity_class: str
    donor: bool
    acceptor: bool
    confidence: str
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "entity_class": self.entity_class,
            "donor": self.donor,
            "acceptor": self.acceptor,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _normalized_name(atom: AtomRecord) -> str:
    return atom.atom_name.strip().upper().replace("`", "'")


def _entity_class(residue_name: str) -> str:
    residue = residue_name.strip().upper()
    if residue in WATER_RESIDUES:
        return "water"
    if residue in PROTEIN_RESIDUES:
        return "protein"
    if residue in NUCLEIC_RESIDUES:
        return "nucleic_acid"
    if residue in ION_RESIDUES:
        return "ion"
    return "ligand"


def _adjacency(atom_count: int, bonds: Iterable[Bond]) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {index: [] for index in range(atom_count)}
    for first, second in bonds:
        result[first].append(second)
        result[second].append(first)
    return result


def _has_attached_hydrogen(atom_index: int, atoms: Sequence[AtomRecord], adjacency: Mapping[int, Sequence[int]]) -> bool:
    return any(atoms[index].element.upper() == "H" for index in adjacency[atom_index])


def _protein_role(atom: AtomRecord, has_hydrogen: bool) -> AtomChemicalRole:
    residue = atom.residue_name.upper()
    # CHARMM names the delta-, epsilon-, and doubly protonated histidine states
    # HSD, HSE, and HSP; their chemistry is identical to HID, HIE, and HIP.
    residue = {"HSD": "HID", "HSE": "HIE", "HSP": "HIP"}.get(residue, residue)
    name = _normalized_name(atom)
    donor = False
    acceptor = name in _PROTEIN_ACCEPTORS.get(residue, frozenset({"O", "OXT"}))
    if name == "N" and residue != "PRO":
        donor = has_hydrogen
    elif name in _PROTEIN_SIDECHAIN_DONORS.get(residue, frozenset()):
        donor = has_hydrogen
    if residue in {"HIS", "HID", "HIE"} and name in {"ND1", "NE2"}:
        # Explicit bonded hydrogens distinguish neutral histidine tautomers.
        acceptor = not has_hydrogen
    if residue == "HIP" and name in {"ND1", "NE2"}:
        acceptor = False
    return AtomChemicalRole("protein", donor, acceptor, "template", "standard_protein_template_v1")


def _nucleic_role(atom: AtomRecord, has_hydrogen: bool) -> AtomChemicalRole:
    name = _normalized_name(atom)
    base = _NUCLEIC_BASES[atom.residue_name.upper()]
    donor = name in _NUCLEIC_DONORS[base] and has_hydrogen
    # Hydroxyl oxygens are donor-capable only if their explicit bonded H is present.
    if name in {"O2'", "O2*", "O3'", "O3*", "O5'", "O5*"} and has_hydrogen:
        donor = True
    acceptor = name in _NUCLEIC_ACCEPTORS[base] or name in _NUCLEIC_SUGAR_ACCEPTORS
    return AtomChemicalRole("nucleic_acid", donor, acceptor, "template", "standard_nucleic_acid_template_v1")


def infer_atom_chemical_roles(
    atoms: Sequence[AtomRecord], bonds: Sequence[Bond]
) -> Dict[int, AtomChemicalRole]:
    """Infer direct hydrogen-bond roles without user-provided atom indices.

    Explicit connectivity is required by the caller.  Protein and nucleic-acid
    templates are deterministic.  For untemplated residues, the fallback is
    intentionally conservative and its lower-confidence provenance remains in
    every candidate dictionary entry.
    """

    adjacency = _adjacency(len(atoms), bonds)
    roles: Dict[int, AtomChemicalRole] = {}
    for atom in atoms:
        entity = _entity_class(atom.residue_name)
        attached_h = _has_attached_hydrogen(atom.atom_index, atoms, adjacency)
        if entity == "protein":
            role = _protein_role(atom, attached_h)
        elif entity == "nucleic_acid":
            role = _nucleic_role(atom, attached_h)
        elif entity in {"water", "ion"}:
            role = AtomChemicalRole(entity, False, False, "excluded", f"{entity}_not_a_direct_solute_role")
        else:
            element = atom.element.upper()
            donor = element in {"N", "O", "S"} and attached_h
            # A neutral or anionic N/O/S ligand atom may accept.  Tetravalent
            # nitrogen is excluded, but bond order/formal charge remain absent
            # from PDB/GRO identities and are therefore recorded as provisional.
            acceptor = element in {"O", "S"} or (element == "N" and not attached_h and len(adjacency[atom.atom_index]) < 4)
            role = AtomChemicalRole("ligand", donor, acceptor, "provisional", "generic_element_connectivity_v1")
        roles[atom.atom_index] = role
    return roles


def scope_allows(
    donor_entity: str, acceptor_entity: str, interaction_scope: str
) -> bool:
    """Return whether an oriented direct H-bond endpoint pair is in scope."""

    pair = frozenset((donor_entity, acceptor_entity))
    if interaction_scope == "all_solute":
        return donor_entity in {"protein", "nucleic_acid", "ligand"} and acceptor_entity in {"protein", "nucleic_acid", "ligand"}
    required = {
        "protein_protein": frozenset(("protein",)),
        "protein_ligand": frozenset(("protein", "ligand")),
        "protein_nucleic_acid": frozenset(("protein", "nucleic_acid")),
        "nucleic_acid_nucleic_acid": frozenset(("nucleic_acid",)),
        "nucleic_acid_ligand": frozenset(("nucleic_acid", "ligand")),
        "ligand_ligand": frozenset(("ligand",)),
    }
    return pair == required.get(interaction_scope, frozenset())


def chemistry_summary(roles: Mapping[int, AtomChemicalRole]) -> Dict[str, object]:
    """Summarize automatic chemistry provenance for audit-friendly reports."""

    entity_counts: Dict[str, int] = {}
    confidence_counts: Dict[str, int] = {}
    donor_count = 0
    acceptor_count = 0
    for role in roles.values():
        entity_counts[role.entity_class] = entity_counts.get(role.entity_class, 0) + 1
        confidence_counts[role.confidence] = confidence_counts.get(role.confidence, 0) + 1
        donor_count += int(role.donor)
        acceptor_count += int(role.acceptor)
    return {
        "entity_atom_counts": dict(sorted(entity_counts.items())),
        "chemistry_confidence_atom_counts": dict(sorted(confidence_counts.items())),
        "donor_atom_count": donor_count,
        "acceptor_atom_count": acceptor_count,
    }
