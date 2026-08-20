"""Optional, fail-closed OpenMM PDB connectivity export.

The analysis suite consumes covalent connectivity, not force-field parameters.
OpenMM is therefore used only as an optional preparation aid when a simulation
topology was not retained.  Standard residue templates and explicit PDB CONECT
records are accepted; arbitrary distance-based bond guessing is never used.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence


class OpenMMConnectivityError(ValueError):
    """Raised when OpenMM cannot produce complete, auditable connectivity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_pdb_connectivity(
    source_pdb: Path,
    *,
    additional_bond_definitions: Sequence[Path] = (),
) -> Dict[str, object]:
    """Return portable connectivity derived from OpenMM's explicit topology.

    Additional bond-definition XML files are loaded before the PDB.  This is
    OpenMM's documented extension route for nonstandard residue templates.  A
    multi-atom residue containing any isolated atom is rejected rather than
    silently accepted as chemically complete.
    """

    try:
        import openmm
        from openmm.app import PDBFile, Topology
    except ImportError as exc:
        raise OpenMMConnectivityError(
            "OpenMM connectivity generation was requested but OpenMM is unavailable; "
            "install the openmm-connectivity optional dependency or supply PSF, "
            "PRMTOP/PARM7, or salsbury-bonds-v1 JSON connectivity"
        ) from exc

    source = source_pdb.expanduser().resolve(strict=True)
    definition_paths = [
        path.expanduser().resolve(strict=True)
        for path in additional_bond_definitions
    ]
    for path in definition_paths:
        if path.suffix.lower() != ".xml":
            raise OpenMMConnectivityError(
                f"OpenMM bond definition must be XML: {path}"
            )
        Topology.loadBondDefinitions(str(path))

    pdb = PDBFile(str(source))
    topology = pdb.topology
    atoms = list(topology.atoms())
    bonds = sorted({
        (min(first.index, second.index), max(first.index, second.index))
        for first, second in topology.bonds()
    })
    if not atoms:
        raise OpenMMConnectivityError("OpenMM PDB topology contains no atoms")
    if not bonds:
        raise OpenMMConnectivityError(
            "OpenMM PDB topology contains no bonds; supply explicit connectivity "
            "or reviewed residue bond definitions"
        )

    degrees = [0] * len(atoms)
    for first, second in bonds:
        if first == second or first < 0 or second >= len(atoms):
            raise OpenMMConnectivityError(
                "OpenMM PDB topology contains invalid bond indices"
            )
        degrees[first] += 1
        degrees[second] += 1

    unsafe = []
    for residue in topology.residues():
        residue_atoms = list(residue.atoms())
        if len(residue_atoms) <= 1:
            continue
        for atom in residue_atoms:
            if degrees[atom.index] == 0:
                unsafe.append({
                    "atom_index": atom.index,
                    "atom_name": atom.name,
                    "residue_name": residue.name,
                    "residue_id": residue.id,
                    "chain_id": residue.chain.id,
                })
    if unsafe:
        preview = ", ".join(
            f"{item['atom_index']}:{item['residue_name']}/{item['atom_name']}"
            for item in unsafe[:10]
        )
        raise OpenMMConnectivityError(
            f"{len(unsafe)} atoms in multi-atom residues have no OpenMM bond; "
            f"unsupported or incomplete residue templates are likely ({preview})"
        )

    return {
        "format": "salsbury-bonds-v1",
        "atom_count": len(atoms),
        "index_base": 0,
        "bonds": [list(bond) for bond in bonds],
        "provenance": {
            "generator": "salsbury_md_analysis.openmm_connectivity",
            "openmm_version": openmm.__version__,
            "source_pdb": str(source),
            "source_pdb_sha256": _sha256(source),
            "bond_model": (
                "OpenMM PDBFile standard-residue topology, explicit PDB connectivity, "
                "and any declared additional bond definitions; no distance guessing"
            ),
            "additional_bond_definitions": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in definition_paths
            ],
            "multi_atom_residue_isolated_atom_count": 0,
            "scientific_status": "requires topology-owner review",
        },
    }
