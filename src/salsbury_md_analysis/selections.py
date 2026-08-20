"""Portable named atom selections and common-basis topology correspondence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, MAPPING_POLICIES
from .chemical_identity import SOLVENT_AND_ION_RESIDUES, WATER_RESIDUES


_BACKBONE_NAMES = {"N", "CA", "C", "O"}
_MACROMOLECULAR_BACKBONE_NAMES = _BACKBONE_NAMES | {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'", "C2'", "O2'",
    "C3'", "O3'",
}
_COMPLEX_TRACE_NAMES = {"CA", "C1'"}
_PRESETS = {
    "all", "backbone", "complex_trace", "heavy", "macromolecular_backbone",
    "solute_heavy", "molecular_payload",
}


def _is_hydrogen(atom: AtomRecord) -> bool:
    return (
        atom.element.upper() == "H"
        or atom.atom_name.lstrip("0123456789").upper().startswith("H")
    )


@dataclass(frozen=True)
class AtomCorrespondence:
    selection_id: str
    policy: str
    reference_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    reference_atoms: Tuple[AtomRecord, ...]
    target_atoms: Tuple[AtomRecord, ...]
    reference_selected_count: int
    target_selected_count: int
    reference_coverage: float
    residue_name_mismatch_count: int
    mapping_signature_sha256: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "policy": self.policy,
            "mapped_atom_count": len(self.reference_indices),
            "reference_selected_atom_count": self.reference_selected_count,
            "target_selected_atom_count": self.target_selected_count,
            "reference_coverage": self.reference_coverage,
            "reference_indices": list(self.reference_indices),
            "target_indices": list(self.target_indices),
            "residue_name_mismatch_count": self.residue_name_mismatch_count,
            "mapping_signature_sha256": self.mapping_signature_sha256,
        }


def select_atoms(
    atoms: Sequence[AtomRecord], definition: Mapping[str, object], selection_id: str
) -> Tuple[AtomRecord, ...]:
    """Resolve a portable preset or exact atom-name list against topology atoms."""

    if set(definition) == {"preset"}:
        preset = definition["preset"]
        if preset not in _PRESETS:
            raise AtomMappingError(
                f"selection {selection_id!r} preset must be one of: "
                + ", ".join(sorted(_PRESETS))
            )
        if preset == "all":
            selected = list(atoms)
        elif preset == "backbone":
            selected = [
                atom for atom in atoms
                if atom.atom_name.upper() in _BACKBONE_NAMES
                and atom.residue_name.upper() not in SOLVENT_AND_ION_RESIDUES
            ]
        elif preset in {"heavy", "solute_heavy"}:
            selected = [
                atom for atom in atoms
                if not _is_hydrogen(atom)
                and (
                    preset == "heavy"
                    or atom.residue_name.upper() not in SOLVENT_AND_ION_RESIDUES
                )
            ]
        elif preset == "molecular_payload":
            selected = [
                atom for atom in atoms
                if atom.residue_name.upper() not in WATER_RESIDUES
            ]
        elif preset == "macromolecular_backbone":
            selected = [
                atom for atom in atoms
                if atom.atom_name.upper() in _MACROMOLECULAR_BACKBONE_NAMES
                and atom.residue_name.upper() not in SOLVENT_AND_ION_RESIDUES
            ]
        else:
            selected = [
                atom for atom in atoms
                if atom.atom_name.upper() in _COMPLEX_TRACE_NAMES
                and atom.residue_name.upper() not in SOLVENT_AND_ION_RESIDUES
            ]
    elif set(definition) == {"atom_names"}:
        names = definition["atom_names"]
        if not isinstance(names, list) or not names:
            raise AtomMappingError(
                f"selection {selection_id!r} atom_names must be a nonempty array"
            )
        normalized = {str(name).strip().upper() for name in names}
        selected = [atom for atom in atoms if atom.atom_name.upper() in normalized]
    elif set(definition) == {"residue_keys", "heavy_only"}:
        raw_keys = definition["residue_keys"]
        heavy_only = definition["heavy_only"]
        if not isinstance(raw_keys, list) or not raw_keys:
            raise AtomMappingError(
                f"selection {selection_id!r} residue_keys must be a nonempty array"
            )
        if not isinstance(heavy_only, bool):
            raise AtomMappingError(
                f"selection {selection_id!r} heavy_only must be boolean"
            )
        keys = set()
        for index, raw in enumerate(raw_keys):
            if not isinstance(raw, dict) or set(raw) != {
                "chain_id", "residue_number", "insertion_code",
            }:
                raise AtomMappingError(
                    f"selection {selection_id!r} residue_keys[{index}] must contain "
                    "chain_id, residue_number, and insertion_code"
                )
            chain_id = raw["chain_id"]
            residue_number = raw["residue_number"]
            insertion_code = raw["insertion_code"]
            if (
                not isinstance(chain_id, str)
                or isinstance(residue_number, bool)
                or not isinstance(residue_number, int)
                or not isinstance(insertion_code, str)
            ):
                raise AtomMappingError(
                    f"selection {selection_id!r} residue_keys[{index}] has invalid field types"
                )
            key = (chain_id, residue_number, insertion_code)
            if key in keys:
                raise AtomMappingError(
                    f"selection {selection_id!r} contains duplicate residue key {key!r}"
                )
            keys.add(key)
        selected = [
            atom for atom in atoms
            if (atom.chain_id, atom.residue_number, atom.insertion_code) in keys
            and (not heavy_only or not _is_hydrogen(atom))
        ]
    else:
        raise AtomMappingError(
            f"selection {selection_id!r} must declare preset, atom_names, or "
            "residue_keys with heavy_only"
        )
    if not selected:
        raise AtomMappingError(f"selection {selection_id!r} produced no atoms")
    return tuple(selected)


def _unique_index(
    atoms: Sequence[AtomRecord], policy: str, label: str
) -> Dict[Tuple[object, ...], AtomRecord]:
    indexed: Dict[Tuple[object, ...], AtomRecord] = {}
    for atom in atoms:
        key = atom.match_key(policy)
        if key in indexed:
            raise AtomMappingError(
                f"{label} has duplicate atom identity {key!r} under {policy!r} policy"
            )
        indexed[key] = atom
    return indexed


def build_correspondence(
    reference_atoms: Sequence[AtomRecord],
    target_atoms: Sequence[AtomRecord],
    definition: Mapping[str, object],
    selection_id: str,
    policy: str,
    minimum_reference_coverage: float,
) -> AtomCorrespondence:
    """Create one ordered common basis for a named selection and topology pair."""

    return build_common_correspondences(
        reference_atoms,
        (target_atoms,),
        definition,
        selection_id,
        policy,
        minimum_reference_coverage,
    )[0]


def build_common_correspondences(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    definition: Mapping[str, object],
    selection_id: str,
    policy: str,
    minimum_reference_coverage: float,
) -> Tuple[AtomCorrespondence, ...]:
    """Create one reference-ordered atom basis shared by every target topology."""

    if policy not in MAPPING_POLICIES:
        raise AtomMappingError(f"common_atom_policy must be one of: {', '.join(MAPPING_POLICIES)}")
    if not 0.0 <= minimum_reference_coverage <= 1.0:
        raise AtomMappingError("minimum_reference_coverage must be between 0 and 1")
    if not target_atom_sets:
        raise AtomMappingError("at least one target topology is required")
    selected_reference = select_atoms(reference_atoms, definition, selection_id)
    reference_index = _unique_index(selected_reference, policy, "reference topology")
    selected_targets = tuple(
        select_atoms(atoms, definition, selection_id) for atoms in target_atom_sets
    )
    target_indexes = tuple(
        _unique_index(atoms, policy, f"target topology {index + 1}")
        for index, atoms in enumerate(selected_targets)
    )
    keys = [
        key for key, atom in sorted(
            reference_index.items(), key=lambda item: item[1].atom_index
        )
        if all(key in target_index for target_index in target_indexes)
    ]
    if not keys:
        raise AtomMappingError(
            f"selection {selection_id!r} has no atom identity common to every topology "
            f"under {policy!r} policy"
        )
    coverage = len(keys) / len(selected_reference)
    if coverage < minimum_reference_coverage:
        raise AtomMappingError(
            f"selection {selection_id!r} all-topology reference coverage is {coverage:.6f}; "
            f"required minimum is {minimum_reference_coverage:.6f}"
        )
    mapped_reference = tuple(reference_index[key] for key in keys)
    common_signature_payload = {
        "selection_id": selection_id,
        "policy": policy,
        "reference_basis": [
            {
                "identity": list(key),
                "reference_index": left.atom_index,
            }
            for key, left in zip(keys, mapped_reference)
        ],
    }
    results = []
    for selected_target, target_index in zip(selected_targets, target_indexes):
        mapped_target = tuple(target_index[key] for key in keys)
        mismatch_count = sum(
            left.residue_name != right.residue_name
            for left, right in zip(mapped_reference, mapped_target)
        )
        signature_payload = {
            **common_signature_payload,
            "target_indices": [atom.atom_index for atom in mapped_target],
        }
        signature = hashlib.sha256(
            json.dumps(
                signature_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        results.append(AtomCorrespondence(
            selection_id=selection_id,
            policy=policy,
            reference_indices=tuple(atom.atom_index for atom in mapped_reference),
            target_indices=tuple(atom.atom_index for atom in mapped_target),
            reference_atoms=mapped_reference,
            target_atoms=mapped_target,
            reference_selected_count=len(selected_reference),
            target_selected_count=len(selected_target),
            reference_coverage=coverage,
            residue_name_mismatch_count=mismatch_count,
            mapping_signature_sha256=signature,
        ))
    return tuple(results)
