#!/usr/bin/env python3
"""Write an atom-order-matched PDB from a serialized OpenMM ``State``.

The supplied PDB contributes topology and atom order. Coordinates and periodic
box vectors come from the reviewed checkpoint state. No input file is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology_pdb", type=Path)
    parser.add_argument("state_xml", type=Path)
    parser.add_argument("output_pdb", type=Path)
    arguments = parser.parse_args()
    topology_pdb = arguments.topology_pdb.expanduser().resolve(strict=True)
    state_xml = arguments.state_xml.expanduser().resolve(strict=True)
    output_pdb = arguments.output_pdb.expanduser().resolve(strict=False)
    if output_pdb.exists():
        parser.error(f"refusing to replace existing output: {output_pdb}")
    if output_pdb in {topology_pdb, state_xml}:
        parser.error("output must differ from both inputs")
    try:
        import openmm
        from openmm.app import PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM is required; install the openmm-connectivity optional dependency"
        ) from exc

    topology = PDBFile(str(topology_pdb)).topology
    state = openmm.XmlSerializer.deserialize(state_xml.read_text(encoding="utf-8"))
    positions = state.getPositions()
    if positions is None:
        raise RuntimeError("serialized OpenMM State contains no positions")
    atom_count = sum(1 for _ in topology.atoms())
    if len(positions) != atom_count:
        raise RuntimeError(
            f"PDB/State particle mismatch: topology has {atom_count} atoms but "
            f"State has {len(positions)} positions"
        )
    box_vectors = state.getPeriodicBoxVectors()
    if box_vectors is not None:
        topology.setPeriodicBoxVectors(box_vectors)
    with output_pdb.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, positions, handle, keepIds=True)
    print(json.dumps({
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "topology_pdb": str(topology_pdb),
        "topology_pdb_sha256": sha256(topology_pdb),
        "state_xml": str(state_xml),
        "state_xml_sha256": sha256(state_xml),
        "output_pdb": str(output_pdb),
        "output_pdb_sha256": sha256(output_pdb),
        "atom_count": atom_count,
        "openmm_version": openmm.__version__,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
