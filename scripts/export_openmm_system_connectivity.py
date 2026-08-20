#!/usr/bin/env python3
"""Export an OpenMM ``System`` bond graph as portable connectivity JSON.

Unlike :mod:`export_openmm_connectivity`, this converter uses the bonds already
declared by an atom-order-matched OpenMM PDB topology. It inventories
``HarmonicBondForce``, ``CustomBondForce``, and constraint pairs from the
serialized ``System`` but excludes System-only pairs by default: they may
represent a mismatched atom order, a restraint, an angle, or rigid geometry
rather than a covalent bond.

The output is suitable for the connectivity-aware periodic reconstruction
used by the direct and water-mediated hydrogen-bond modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Tuple


Bond = Tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(first: int, second: int) -> Bond:
    if first == second:
        raise RuntimeError("OpenMM system contains a self-bond")
    return (first, second) if first < second else (second, first)


def _force_bonds(
    system: object, *, include_custom_bonds: bool = False
) -> Iterator[Bond]:
    """Yield pairs from selected OpenMM bond-force classes.

    ``CustomBondForce`` is excluded unless explicitly requested because it may
    encode restraints or other pairwise terms instead of connectivity.
    """
    for force in system.getForces():
        force_name = force.__class__.__name__
        if force_name != "HarmonicBondForce" and not (
            include_custom_bonds and force_name == "CustomBondForce"
        ):
            continue
        for index in range(force.getNumBonds()):
            parameters = force.getBondParameters(index)
            yield _canonical(int(parameters[0]), int(parameters[1]))


def _topology_bonds(topology: object) -> Iterator[Bond]:
    """Yield bonds declared by the atom-order-matched OpenMM topology."""
    for first, second in topology.bonds():
        yield _canonical(int(first.index), int(second.index))


def _constraint_bonds(system: object) -> Iterator[Bond]:
    """Yield unique particle pairs carrying any OpenMM constraint."""
    for index in range(system.getNumConstraints()):
        first, second, _distance = system.getConstraintParameters(index)
        yield _canonical(int(first), int(second))


def _select_bonds(
    topology_bonds: Iterable[Bond],
    harmonic_bonds: Iterable[Bond],
    custom_bonds: Iterable[Bond],
    constraint_bonds: Iterable[Bond],
    *,
    include_harmonic_force_only_pairs: bool = False,
    include_custom_bonds: bool = False,
    include_constraint_only_pairs: bool = False,
) -> Tuple[set[Bond], dict[str, set[Bond]]]:
    """Return selected bonds and the ambiguous System-only categories.

    The PDB topology is the default covalent graph. System-only harmonic,
    custom, and constraint pairs are recorded separately and included only by
    their corresponding explicit review switch.
    """
    topology = set(topology_bonds)
    harmonic = set(harmonic_bonds)
    custom = set(custom_bonds)
    constraints = set(constraint_bonds)
    harmonic_only = harmonic - topology
    custom_only = custom - topology
    constraint_only = constraints - topology - harmonic - custom
    selected = set(topology)
    if include_harmonic_force_only_pairs:
        selected.update(harmonic_only)
    if include_custom_bonds:
        selected.update(custom_only)
    if include_constraint_only_pairs:
        selected.update(constraint_only)
    return selected, {
        "harmonic_force_only": harmonic_only,
        "custom_force_only": custom_only,
        "constraint_only": constraint_only,
    }


def export_connectivity(
    topology_pdb: Path,
    system_xml: Path,
    *,
    include_harmonic_force_only_pairs: bool = False,
    include_custom_bonds: bool = False,
    include_constraint_only_pairs: bool = False,
) -> dict:
    try:
        import openmm
        from openmm.app import PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "OpenMM is required; install the openmm-connectivity optional dependency"
        ) from exc

    pdb = PDBFile(str(topology_pdb))
    atom_count = sum(1 for _ in pdb.topology.atoms())
    system = openmm.XmlSerializer.deserialize(system_xml.read_text(encoding="utf-8"))
    particle_count = int(system.getNumParticles())
    if atom_count != particle_count:
        raise RuntimeError(
            "PDB/System particle mismatch: "
            f"topology has {atom_count} atoms but System has {particle_count} particles"
        )

    topology_bond_set = set(_topology_bonds(pdb.topology))
    harmonic_bond_set = set(_force_bonds(system))
    all_requested_force_bonds = set(
        _force_bonds(system, include_custom_bonds=True)
    )
    custom_bond_set = all_requested_force_bonds - harmonic_bond_set
    constraint_bond_set = set(_constraint_bonds(system))
    bonds, excluded_categories = _select_bonds(
        topology_bond_set,
        harmonic_bond_set,
        custom_bond_set,
        constraint_bond_set,
        include_harmonic_force_only_pairs=(
            include_harmonic_force_only_pairs
        ),
        include_custom_bonds=include_custom_bonds,
        include_constraint_only_pairs=include_constraint_only_pairs,
    )
    if not bonds:
        raise RuntimeError("OpenMM PDB topology contains no selected bonds")
    if any(first < 0 or second >= atom_count for first, second in bonds):
        raise RuntimeError("OpenMM System connectivity includes an out-of-range particle index")

    return {
        "format": "salsbury-bonds-v1",
        "atom_count": atom_count,
        "index_base": 0,
        "bonds": [list(bond) for bond in sorted(bonds)],
        "provenance": {
            "generator": "scripts/export_openmm_system_connectivity.py",
            "openmm_version": openmm.__version__,
            "source_topology_pdb_name": topology_pdb.name,
            "source_topology_pdb_sha256": _sha256(topology_pdb),
            "source_system_xml_name": system_xml.name,
            "source_system_xml_sha256": _sha256(system_xml),
            "bond_model": (
                "OpenMM PDB topology bonds"
                + (
                    " plus explicitly requested System-only "
                    "HarmonicBondForce pairs"
                    if include_harmonic_force_only_pairs else ""
                )
                + (
                    " and explicitly requested CustomBondForce pairs"
                    if include_custom_bonds else ""
                )
                + (
                    " and explicitly requested constraint-only pairs"
                    if include_constraint_only_pairs else ""
                )
                + "; no coordinate-distance bond inference"
            ),
            "topology_bond_count": len(topology_bond_set),
            "harmonic_force_pair_count": len(harmonic_bond_set),
            "harmonic_force_only_pair_count": len(
                excluded_categories["harmonic_force_only"]
            ),
            "harmonic_force_only_pair_policy": (
                "included_by_explicit_request"
                if include_harmonic_force_only_pairs
                else "excluded_by_default"
            ),
            "custom_force_pair_count": len(custom_bond_set),
            "custom_force_only_pair_count": len(
                excluded_categories["custom_force_only"]
            ),
            "constraint_count": int(system.getNumConstraints()),
            "unique_constraint_pair_count": len(constraint_bond_set),
            "constraint_only_pair_count": len(
                excluded_categories["constraint_only"]
            ),
            "constraint_only_pair_policy": (
                "included_by_explicit_request"
                if include_constraint_only_pairs else "excluded_by_default"
            ),
            "custom_bond_force_policy": (
                "included_by_explicit_request"
                if include_custom_bonds else "excluded_by_default"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology_pdb", type=Path)
    parser.add_argument("system_xml", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--include-harmonic-force-only-pairs", action="store_true",
        help=(
            "Treat HarmonicBondForce pairs absent from the atom-order-matched "
            "PDB topology as connectivity. This is off by default because a "
            "System/PDB order mismatch or specialized force construction can "
            "make System-only pairs unsafe."
        ),
    )
    parser.add_argument(
        "--include-custom-bonds", action="store_true",
        help=(
            "Treat every CustomBondForce pair as connectivity. This is off by "
            "default because custom pairs may be restraints rather than bonds."
        ),
    )
    parser.add_argument(
        "--include-constraint-only-pairs", action="store_true",
        help=(
            "Treat every System constraint pair that is absent from both the "
            "PDB topology and HarmonicBondForce as connectivity. This is off "
            "by default because angle and rigid-geometry constraints are not "
            "necessarily covalent bonds."
        ),
    )
    arguments = parser.parse_args()
    topology_pdb = arguments.topology_pdb.expanduser().resolve(strict=True)
    system_xml = arguments.system_xml.expanduser().resolve(strict=True)
    output = arguments.output_json.expanduser().resolve(strict=False)
    if output in {topology_pdb, system_xml}:
        parser.error("output must differ from both inputs")
    payload = export_connectivity(
        topology_pdb, system_xml,
        include_harmonic_force_only_pairs=(
            arguments.include_harmonic_force_only_pairs
        ),
        include_custom_bonds=arguments.include_custom_bonds,
        include_constraint_only_pairs=arguments.include_constraint_only_pairs,
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "topology_pdb": str(topology_pdb),
        "system_xml": str(system_xml),
        "output_json": str(output),
        "atom_count": payload["atom_count"],
        "bond_count": len(payload["bonds"]),
        "output_sha256": _sha256(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
