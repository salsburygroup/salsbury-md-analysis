#!/usr/bin/env python3
"""Export OpenMM's explicit PDB topology as portable salsbury-bonds-v1 JSON.

OpenMM constructs standard-residue bonds from atom/residue names and retains
explicit PDB connectivity.  This converter never guesses general bonds from a
distance cutoff.  It fails if a multi-atom residue contains an isolated atom,
which is a useful guard against unsupported residue templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from salsbury_md_analysis.openmm_connectivity import export_pdb_connectivity
from salsbury_md_analysis.manifests import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdb", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--bond-definitions", type=Path, action="append", default=[],
        help="Optional OpenMM residue bond-definition XML; repeat as needed.",
    )
    arguments = parser.parse_args()
    source = arguments.source_pdb.expanduser().resolve(strict=True)
    output = arguments.output_json.expanduser().resolve(strict=False)
    if output == source:
        parser.error("output must differ from source")
    payload = export_pdb_connectivity(
        source, additional_bond_definitions=arguments.bond_definitions
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "source_pdb": str(source),
        "output_json": str(output),
        "atom_count": payload["atom_count"],
        "bond_count": len(payload["bonds"]),
        "output_sha256": sha256_file(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
