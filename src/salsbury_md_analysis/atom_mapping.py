"""Deterministic, fail-closed common-atom mapping for validated topologies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .manifests import sha256_file
from .preflight import FileProbeError, probe_gro, probe_pdb
from .reporting import DataclassRecordMixin


MAPPING_POLICIES = ("strict", "position")
ATOM_SELECTIONS = ("all", "backbone", "ca", "heavy")
_BACKBONE_NAMES = {"N", "CA", "C", "O"}
_BASE36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_BASE36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


class AtomMappingError(ValueError):
    """Raised when atom identities cannot be mapped unambiguously."""


@dataclass(frozen=True)
class AtomRecord(DataclassRecordMixin):
    atom_index: int
    serial: int
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    element: str

    def match_key(self, policy: str) -> Tuple[object, ...]:
        base = (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.atom_name,
            self.altloc,
        )
        if policy == "strict":
            return base[:3] + (self.residue_name,) + base[3:]
        if policy == "position":
            return base
        raise ValueError(f"unknown mapping policy: {policy}")

def _infer_element(atom_name: str, declared: str = "") -> str:
    if declared.strip():
        return declared.strip().upper()
    letters = "".join(character for character in atom_name if character.isalpha())
    return letters[:1].upper() if letters else ""


def decode_pdb_hybrid36(field: str, label: str) -> int:
    """Decode decimal, wwPDB hybrid-36, or VMD hexadecimal PDB fields.

    VMD writes large atom serials and residue numbers as fixed-width lowercase
    hexadecimal values such as ``186a0``.  They are distinguishable from
    formal hybrid-36 because hybrid-36 encoded fields begin with a letter.
    """

    width = len(field)
    text = field.strip()
    if not text:
        raise AtomMappingError(f"PDB {label} field is empty")
    if text[0] in "-+0123456789":
        try:
            return int(text)
        except ValueError as exc:
            if (
                text[0].isdigit()
                and any(character in "abcdef" for character in text)
                and all(character in "0123456789abcdef" for character in text)
            ):
                return int(text, 16)
            raise AtomMappingError(f"PDB {label} field is malformed: {field!r}") from exc
    alphabet = _BASE36_UPPER if text[0].isupper() else _BASE36_LOWER
    if text[0].isupper():
        offset = -(10 * (36 ** (width - 1))) + (10 ** width)
    elif text[0].islower():
        offset = (16 * (36 ** (width - 1))) + (10 ** width)
    else:
        raise AtomMappingError(f"PDB {label} field is malformed: {field!r}")
    lookup = {character: index for index, character in enumerate(alphabet)}
    if len(text) != width or any(character not in lookup for character in text):
        raise AtomMappingError(f"PDB {label} field is malformed: {field!r}")
    value = 0
    for character in text:
        value = value * 36 + lookup[character]
    return value + offset


def read_pdb_atoms(path: Path) -> List[AtomRecord]:
    """Read atom identities from the first PDB model after structural probing."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        probe_pdb(source, "topology")
    except FileProbeError as exc:
        raise AtomMappingError(str(exc)) from exc
    atoms: List[AtomRecord] = []
    saw_model = False
    in_first_model = False
    first_model_complete = False
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = line[:6].strip().upper()
                if record == "MODEL":
                    if not saw_model:
                        saw_model = True
                        in_first_model = True
                    elif first_model_complete:
                        break
                    continue
                if record == "ENDMDL" and in_first_model:
                    in_first_model = False
                    first_model_complete = True
                    continue
                if record not in {"ATOM", "HETATM"}:
                    continue
                if saw_model and not in_first_model:
                    continue
                if len(line) < 27:
                    raise AtomMappingError(f"PDB atom record is too short at line {line_number}")
                try:
                    serial = decode_pdb_hybrid36(line[6:11], "serial")
                    residue_number = decode_pdb_hybrid36(
                        line[22:26], "residue number"
                    )
                except AtomMappingError as exc:
                    raise AtomMappingError(
                        f"PDB serial or residue number is malformed at line {line_number}"
                    ) from exc
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                if not atom_name or not residue_name:
                    raise AtomMappingError(
                        f"PDB atom or residue name is empty at line {line_number}"
                    )
                declared_element = line[76:78] if len(line) >= 78 else ""
                atoms.append(
                    AtomRecord(
                        atom_index=len(atoms),
                        serial=serial,
                        atom_name=atom_name,
                        altloc=line[16:17].strip(),
                        residue_name=residue_name,
                        chain_id=line[21:22].strip(),
                        residue_number=residue_number,
                        insertion_code=line[26:27].strip(),
                        element=_infer_element(atom_name, declared_element),
                    )
                )
    except (OSError, UnicodeError) as exc:
        raise AtomMappingError(str(exc)) from exc
    if not atoms:
        raise AtomMappingError("PDB produced no atoms for mapping")
    return atoms


def read_gro_atoms(path: Path) -> List[AtomRecord]:
    """Read fixed-column atom identities from one validated GRO frame."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        metadata = probe_gro(source, "topology")
    except FileProbeError as exc:
        raise AtomMappingError(str(exc)) from exc
    atom_count = int(metadata["atom_count"])
    try:
        lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AtomMappingError(str(exc)) from exc
    atoms: List[AtomRecord] = []
    for offset, line in enumerate(lines[2 : 2 + atom_count], start=3):
        if len(line) < 20:
            raise AtomMappingError(f"GRO atom record is too short at line {offset}")
        try:
            residue_number = int(line[0:5].strip())
            serial = int(line[15:20].strip())
        except ValueError as exc:
            raise AtomMappingError(
                f"GRO residue or atom number is malformed at line {offset}"
            ) from exc
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        if not residue_name or not atom_name:
            raise AtomMappingError(f"GRO atom or residue name is empty at line {offset}")
        atoms.append(
            AtomRecord(
                atom_index=len(atoms),
                serial=serial,
                atom_name=atom_name,
                altloc="",
                residue_name=residue_name,
                chain_id="",
                residue_number=residue_number,
                insertion_code="",
                element=_infer_element(atom_name),
            )
        )
    return atoms


def read_topology_atoms(path: Path) -> Tuple[str, List[AtomRecord]]:
    source = Path(path).expanduser().resolve(strict=False)
    suffix = source.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb", read_pdb_atoms(source)
    if suffix == ".gro":
        return "gro", read_gro_atoms(source)
    raise AtomMappingError("common-atom mapping currently supports PDB and GRO topologies only")


def _selected(atoms: Iterable[AtomRecord], selection: str) -> List[AtomRecord]:
    if selection not in ATOM_SELECTIONS:
        raise ValueError(f"unknown atom selection: {selection}")
    if selection == "all":
        result = list(atoms)
    elif selection == "backbone":
        result = [atom for atom in atoms if atom.atom_name.upper() in _BACKBONE_NAMES]
    elif selection == "ca":
        result = [atom for atom in atoms if atom.atom_name.upper() == "CA"]
    else:
        result = [
            atom for atom in atoms
            if atom.element.upper() != "H" and not atom.atom_name.lstrip("0123456789").upper().startswith("H")
        ]
    if not result:
        raise AtomMappingError(f"atom selection {selection!r} produced no atoms")
    return result


def _identity_dict(atom: AtomRecord, policy: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "chain_id": atom.chain_id,
        "residue_number": atom.residue_number,
        "insertion_code": atom.insertion_code,
        "atom_name": atom.atom_name,
        "altloc": atom.altloc,
    }
    if policy == "strict":
        result["residue_name"] = atom.residue_name
    return result


def _index_atoms(
    atoms: Sequence[AtomRecord], policy: str, source_label: str
) -> Dict[Tuple[object, ...], AtomRecord]:
    indexed: Dict[Tuple[object, ...], AtomRecord] = {}
    duplicates: List[Tuple[object, ...]] = []
    for atom in atoms:
        key = atom.match_key(policy)
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = atom
    if duplicates:
        preview = "; ".join(repr(value) for value in duplicates[:5])
        extra = f" and {len(duplicates) - 5} more" if len(duplicates) > 5 else ""
        raise AtomMappingError(
            f"{source_label} has duplicate identities under {policy!r}: {preview}{extra}"
        )
    return indexed


def _excluded(
    atoms: Sequence[AtomRecord], common: set, policy: str
) -> List[Dict[str, object]]:
    return [atom.as_dict() for atom in atoms if atom.match_key(policy) not in common]


def map_common_atoms(
    reference_path: Path,
    target_paths: Sequence[Path],
    policy: str,
    selection: str,
    minimum_reference_coverage: float,
    hash_content: bool = False,
) -> Dict[str, object]:
    """Map one explicit atom identity basis across every supplied topology."""

    if policy not in MAPPING_POLICIES:
        raise AtomMappingError(f"policy must be one of: {', '.join(MAPPING_POLICIES)}")
    if selection not in ATOM_SELECTIONS:
        raise AtomMappingError(f"selection must be one of: {', '.join(ATOM_SELECTIONS)}")
    if not target_paths:
        raise AtomMappingError("at least one target topology is required")
    if not 0.0 <= minimum_reference_coverage <= 1.0:
        raise AtomMappingError("minimum_reference_coverage must be between 0 and 1")

    reference = Path(reference_path).expanduser().resolve(strict=False)
    reference_format, all_reference_atoms = read_topology_atoms(reference)
    reference_atoms = _selected(all_reference_atoms, selection)
    reference_index = _index_atoms(reference_atoms, policy, "reference topology")

    target_records: List[Dict[str, object]] = []
    target_indexes: List[Dict[Tuple[object, ...], AtomRecord]] = []
    target_atom_lists: List[List[AtomRecord]] = []
    for number, raw_target in enumerate(target_paths, start=1):
        target_id = f"target_{number}"
        target = Path(raw_target).expanduser().resolve(strict=False)
        format_name, all_atoms = read_topology_atoms(target)
        selected_atoms = _selected(all_atoms, selection)
        indexed = _index_atoms(selected_atoms, policy, target_id)
        target_indexes.append(indexed)
        target_atom_lists.append(selected_atoms)
        target_records.append(
            {
                "target_id": target_id,
                "path": str(target),
                "format": format_name,
                "total_atom_count": len(all_atoms),
                "selected_atom_count": len(selected_atoms),
                "sha256": sha256_file(target) if hash_content else None,
            }
        )

    common = set(reference_index)
    for indexed in target_indexes:
        common.intersection_update(indexed)
    ordered_common = sorted(common, key=lambda key: reference_index[key].atom_index)

    mapping_rows: List[Dict[str, object]] = []
    mismatch_counts = {record["target_id"]: 0 for record in target_records}
    for common_index, key in enumerate(ordered_common):
        reference_atom = reference_index[key]
        targets: Dict[str, Dict[str, object]] = {}
        for record, indexed in zip(target_records, target_indexes):
            target_id = str(record["target_id"])
            target_atom = indexed[key]
            if target_atom.residue_name != reference_atom.residue_name:
                mismatch_counts[target_id] += 1
            targets[target_id] = {
                "atom_index": target_atom.atom_index,
                "serial": target_atom.serial,
                "residue_name": target_atom.residue_name,
            }
        mapping_rows.append(
            {
                "common_index": common_index,
                "match_identity": _identity_dict(reference_atom, policy),
                "reference": {
                    "atom_index": reference_atom.atom_index,
                    "serial": reference_atom.serial,
                    "residue_name": reference_atom.residue_name,
                },
                "targets": targets,
            }
        )

    reference_coverage = len(common) / len(reference_atoms)
    issues: List[Dict[str, object]] = []
    if not common:
        issues.append(
            {
                "severity": "error",
                "code": "NO_COMMON_ATOMS",
                "message": "no selected atom identity occurs in every topology",
            }
        )
    if reference_coverage < minimum_reference_coverage:
        issues.append(
            {
                "severity": "error",
                "code": "MINIMUM_REFERENCE_COVERAGE_NOT_MET",
                "message": (
                    f"reference coverage is {reference_coverage:.6f}; "
                    f"required minimum is {minimum_reference_coverage:.6f}"
                ),
            }
        )
    if policy == "position":
        for target_id, mismatch_count in mismatch_counts.items():
            if mismatch_count:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "RESIDUE_NAME_MISMATCH",
                        "target_id": target_id,
                        "count": mismatch_count,
                        "message": (
                            f"{mismatch_count} mapped atoms have a target residue name different "
                            "from the reference because the position policy ignores residue name"
                        ),
                    }
                )

    excluded_targets: Dict[str, List[Dict[str, object]]] = {}
    for record, atoms in zip(target_records, target_atom_lists):
        excluded_targets[str(record["target_id"])] = _excluded(atoms, common, policy)

    signature_payload = {
        "policy": policy,
        "selection": selection,
        "mapping": mapping_rows,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "policy": policy,
        "selection": selection,
        "minimum_reference_coverage": minimum_reference_coverage,
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "reference": {
            "path": str(reference),
            "format": reference_format,
            "total_atom_count": len(all_reference_atoms),
            "selected_atom_count": len(reference_atoms),
            "sha256": sha256_file(reference) if hash_content else None,
        },
        "targets": target_records,
        "common_atom_count": len(common),
        "reference_coverage": reference_coverage,
        "target_coverage": {
            record["target_id"]: len(common) / int(record["selected_atom_count"])
            for record in target_records
        },
        "mapping_signature_sha256": signature,
        "mapping": mapping_rows,
        "excluded_reference_atoms": _excluded(reference_atoms, common, policy),
        "excluded_target_atoms": excluded_targets,
        "indexing": "zero_based",
        "limitations": [
            "Identity mapping does not perform sequence alignment or structural alignment.",
            "The position policy assumes prevalidated chain, residue-number, insertion-code, and atom-name correspondence and ignores residue-name substitutions.",
            "GRO files have no chain or insertion-code fields and residue numbers can wrap; ambiguous identities fail closed.",
            "Inferred elements are sufficient only for the heavy-atom selection heuristic, not chemical typing.",
            "A technically complete map does not establish that a whole-protein common basis is scientifically appropriate.",
        ],
    }
