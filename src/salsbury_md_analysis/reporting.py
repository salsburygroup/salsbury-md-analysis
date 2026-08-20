"""Shared machine-report record builders with stable field ordering."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Protocol


class DataclassRecordMixin:
    """Provide the standard dictionary representation for report dataclasses."""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)  # type: ignore[arg-type]


class AtomIdentity(Protocol):
    """Structural atom interface required by machine-report serialization."""

    atom_index: int
    serial: int
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    element: str


def issue_record(
    severity: str,
    code: str,
    location: str,
    message: str,
    **details: object,
) -> Dict[str, object]:
    """Build one issue record and append optional module-specific details."""

    result: Dict[str, object] = {
        "severity": severity,
        "code": code,
        "location": location,
        "message": message,
    }
    result.update(details)
    return result


def atom_identity_record(
    atom: AtomIdentity, common_index: Optional[int] = None
) -> Dict[str, object]:
    """Serialize a topology atom identity used by analysis reports."""

    result: Dict[str, object] = {}
    if common_index is not None:
        result["common_atom_index"] = common_index
    result.update({
        "reference_atom_index": atom.atom_index,
        "serial": atom.serial,
        "atom_name": atom.atom_name,
        "element": atom.element,
        "residue_name": atom.residue_name,
        "chain_id": atom.chain_id,
        "residue_number": atom.residue_number,
        "insertion_code": atom.insertion_code,
        "altloc": atom.altloc,
    })
    return result
