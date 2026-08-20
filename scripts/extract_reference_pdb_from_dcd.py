#!/usr/bin/env python3
"""Write one immutable PDB reference from a declared DCD frame and PDB atom order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from salsbury_md_analysis.atom_mapping import read_pdb_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.state_coordinate_exports import _write_pdb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-pdb", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    topology = args.topology_pdb.expanduser().resolve(strict=True)
    trajectory = args.trajectory.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if args.frame_index < 0:
        raise ValueError("frame index must be nonnegative")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atoms = read_pdb_atoms(topology)
    frames = iter_coordinate_frames(
        trajectory, "angstrom", {args.frame_index}
    )
    try:
        frame = next(frames)
    except StopIteration as exc:
        raise ValueError("requested DCD frame was not found") from exc
    _write_pdb(
        output,
        atoms,
        [({"source_frame_index": args.frame_index}, frame.coordinates_angstrom)],
        multi_model=False,
    )
    print(json.dumps({
        "output": str(output),
        "source_topology": str(topology),
        "source_trajectory": str(trajectory),
        "source_frame_index": args.frame_index,
        "atom_count": len(atoms),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
