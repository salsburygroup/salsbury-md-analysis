"""Explicit protein continuity, chirality, clash, and covalent-link checks."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .atom_mapping import AtomRecord
from .dihedrals import DihedralAnalysisError, dihedral_degrees
from .geometry import distance3


class StructuralChemistryError(ValueError):
    """Raised when a chemical-integrity snapshot cannot be evaluated."""


_VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "ZN": 1.39, "FE": 1.56, "CA": 2.31,
}
def _residues(atoms: Sequence[AtomRecord]) -> list[Tuple[Tuple[object, ...], Dict[str, int]]]:
    result = []
    key = None
    names: Dict[str, int] = {}
    for atom in atoms:
        current = (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        if current != key:
            if key is not None:
                result.append((key, names))
            key, names = current, {}
        if not atom.altloc and atom.atom_name.upper() not in names:
            names[atom.atom_name.upper()] = atom.atom_index
    if key is not None:
        result.append((key, names))
    return result


def ca_chirality_volume(
    coordinates: Sequence[Sequence[float]], n_index: int, ca_index: int, c_index: int, cb_index: int
) -> float:
    """Return the signed N-C-CB tetrahedral product around C-alpha."""

    xyz = np.asarray(coordinates, dtype=float)
    center = xyz[ca_index]
    return float(np.dot(xyz[n_index] - center, np.cross(xyz[c_index] - center, xyz[cb_index] - center)))


def reference_chirality_signs(
    atoms: Sequence[AtomRecord], coordinates: Sequence[Sequence[float]]
) -> Dict[str, int]:
    result = {}
    for identity, names in _residues(atoms):
        if {"N", "CA", "C", "CB"}.issubset(names) and str(identity[3]).upper() != "GLY":
            volume = ca_chirality_volume(coordinates, names["N"], names["CA"], names["C"], names["CB"])
            result[repr(identity)] = 1 if volume > 0 else -1 if volume < 0 else 0
    return result


def chemical_integrity_snapshot(
    atoms: Sequence[AtomRecord],
    coordinates: Sequence[Sequence[float]],
    *,
    maximum_peptide_bond_angstrom: float,
    maximum_trans_omega_deviation_degrees: float,
    minimum_ca_chirality_volume_angstrom3: float,
    steric_clash_scale: float,
    reference_chirality: Mapping[str, int],
    covalent_bonds: Optional[Sequence[Tuple[int, int]]] = None,
    allow_cis_proline: bool = True,
    declared_covalent_links: Sequence[Mapping[str, object]] = (),
    example_limit: int = 10,
) -> Dict[str, object]:
    """Evaluate one reconstructed frame using explicit, reportable definitions."""

    if len(atoms) != len(coordinates):
        raise StructuralChemistryError("atom and coordinate counts differ")
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise StructuralChemistryError(
            "chemical-integrity coordinates must be a finite atom-by-three array"
        )
    residues = _residues(atoms)
    peptide_breaks = []
    omega_outliers = []
    chirality_outliers = []
    peptide_break_count = 0
    omega_outlier_count = 0
    chirality_outlier_count = 0
    atom_to_residue = {
        atom_index: (identity, names)
        for identity, names in residues
        for atom_index in names.values()
    }
    excluded_pairs = set()
    neighbors = {index: set() for index in range(len(atoms))}
    peptide_links = []
    for raw_first, raw_second in covalent_bonds or ():
        first, second = int(raw_first), int(raw_second)
        if min(first, second) < 0 or max(first, second) >= len(atoms) or first == second:
            raise StructuralChemistryError("covalent-bond atom indices are invalid")
        pair = tuple(sorted((first, second)))
        excluded_pairs.add(pair)
        neighbors[first].add(second)
        neighbors[second].add(first)
        left = atom_to_residue.get(first)
        right = atom_to_residue.get(second)
        if left is None or right is None or left[0] == right[0]:
            continue
        first_name = atoms[first].atom_name.upper()
        second_name = atoms[second].atom_name.upper()
        if first_name == "C" and second_name == "N":
            peptide_links.append((left, right, pair))
        elif first_name == "N" and second_name == "C":
            peptide_links.append((right, left, pair))
    # Bonded (1-2) and angle-related (1-3) pairs are not steric clashes.
    for center, bonded_neighbors in neighbors.items():
        ordered = sorted(bonded_neighbors)
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                excluded_pairs.add((ordered[left_index], ordered[right_index]))

    for (identity, current), (next_identity, following), pair in peptide_links:
        if not {"CA", "C"}.issubset(current) or not {"N", "CA"}.issubset(following):
            continue
        distance = distance3(coordinates[pair[0]], coordinates[pair[1]])
        if distance > maximum_peptide_bond_angstrom:
            peptide_break_count += 1
            if len(peptide_breaks) < example_limit:
                peptide_breaks.append({"residue": repr(identity), "next_residue": repr(next_identity), "distance_angstrom": distance})
        try:
            omega = dihedral_degrees(
                coordinates[current["CA"]], coordinates[current["C"]],
                coordinates[following["N"]], coordinates[following["CA"]],
            )
            deviation = min(abs(omega - 180.0), abs(omega + 180.0))
            cis_proline = (
                allow_cis_proline
                and str(next_identity[3]).upper() == "PRO"
                and abs(omega) <= maximum_trans_omega_deviation_degrees
            )
            if deviation > maximum_trans_omega_deviation_degrees and not cis_proline:
                omega_outlier_count += 1
                if len(omega_outliers) < example_limit:
                    omega_outliers.append({"residue": repr(identity), "next_residue": repr(next_identity), "omega_degrees": omega, "trans_deviation_degrees": deviation})
        except DihedralAnalysisError:
            pass
    for identity, names in residues:
        if not {"N", "CA", "C", "CB"}.issubset(names) or str(identity[3]).upper() == "GLY":
            continue
        volume = ca_chirality_volume(coordinates, names["N"], names["CA"], names["C"], names["CB"])
        reference_sign = int(reference_chirality.get(repr(identity), 0))
        inverted = reference_sign and volume * reference_sign < 0
        if abs(volume) < minimum_ca_chirality_volume_angstrom3 or inverted:
            chirality_outlier_count += 1
            if len(chirality_outliers) < example_limit:
                chirality_outliers.append({"residue": repr(identity), "signed_volume_angstrom3": volume, "reference_sign": reference_sign, "inverted": bool(inverted)})
    link_outliers = []
    for link in declared_covalent_links:
        first, second = map(int, link["atom_indices"])
        if min(first, second) < 0 or max(first, second) >= len(atoms) or first == second:
            raise StructuralChemistryError("declared covalent-link atom indices are invalid")
        excluded_pairs.add(tuple(sorted((first, second))))
        distance = distance3(coordinates[first], coordinates[second])
        if not float(link["minimum_distance_angstrom"]) <= distance <= float(link["maximum_distance_angstrom"]):
            link_outliers.append({"link_id": str(link["link_id"]), "atom_indices": [first, second], "distance_angstrom": distance})
    clashes = []
    clash_count = 0
    if covalent_bonds is not None:
        maximum_radius = max(_VDW_RADII.get(atom.element.upper(), 1.70) for atom in atoms)
        maximum_cutoff = max(0.1, 2.0 * maximum_radius * steric_clash_scale)
        residue_keys = [(atom.chain_id, atom.residue_number, atom.insertion_code) for atom in atoms]
        candidate_pairs = cKDTree(xyz).query_pairs(
            r=maximum_cutoff, output_type="ndarray"
        )
        if candidate_pairs.size:
            candidate_pairs = candidate_pairs[
                np.lexsort((candidate_pairs[:, 1], candidate_pairs[:, 0]))
            ]
        for raw_left, raw_right in candidate_pairs:
            left, right = int(raw_left), int(raw_right)
            pair = (left, right)
            if pair in excluded_pairs or residue_keys[left] == residue_keys[right]:
                continue
            cutoff = steric_clash_scale * (
                _VDW_RADII.get(atoms[left].element.upper(), 1.70)
                + _VDW_RADII.get(atoms[right].element.upper(), 1.70)
            )
            distance = distance3(coordinates[left], coordinates[right])
            if distance < cutoff:
                clash_count += 1
                if len(clashes) < example_limit:
                    clashes.append({
                        "atom_indices": [left, right],
                        "distance_angstrom": distance,
                        "cutoff_angstrom": cutoff,
                    })
    return {
        "peptide_break_count": peptide_break_count,
        "peptide_break_examples": peptide_breaks,
        "omega_outlier_count": omega_outlier_count,
        "omega_outlier_examples": omega_outliers,
        "chirality_outlier_count": chirality_outlier_count,
        "chirality_outlier_examples": chirality_outliers,
        "declared_covalent_link_outlier_count": len(link_outliers),
        "declared_covalent_link_outliers": link_outliers[:example_limit],
        "steric_clash_count": clash_count,
        "steric_clash_examples": clashes,
        "explicit_covalent_bond_count": (
            len({tuple(sorted(map(int, pair))) for pair in covalent_bonds})
            if covalent_bonds is not None
            else None
        ),
        "peptide_link_count": len(peptide_links),
        "steric_clash_status": (
            "evaluated" if covalent_bonds is not None else "not_evaluated_connectivity_required"
        ),
        "steric_exclusion_policy": (
            "same-residue, explicit 1-2 and 1-3 topology pairs, and declared-link pairs excluded"
            if covalent_bonds is not None
            else "not evaluated because explicit covalent connectivity was unavailable"
        ),
    }
