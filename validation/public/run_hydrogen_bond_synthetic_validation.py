#!/usr/bin/env python3
"""Public dependency-light validation of hydrogen-bond discovery and geometry."""

from __future__ import annotations

import json
import math

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.hydrogen_bond_chemistry import infer_atom_chemical_roles
from salsbury_md_analysis.hydrogen_bond_discovery import (
    discover_automatic_candidate_bonds,
)
from salsbury_md_analysis.hydrogen_bonds import hydrogen_bond_present


MAXIMUM_DISTANCE_ANGSTROM = 3.5
MINIMUM_ANGLE_DEGREES = 150.0


def _vector(left, right):
    return tuple(float(right[index]) - float(left[index]) for index in range(3))


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def _independent_geometry(donor, hydrogen, acceptor):
    donor_acceptor = _vector(donor, acceptor)
    hydrogen_donor = _vector(hydrogen, donor)
    hydrogen_acceptor = _vector(hydrogen, acceptor)
    distance = _norm(donor_acceptor)
    denominator = _norm(hydrogen_donor) * _norm(hydrogen_acceptor)
    if denominator == 0.0:
        raise ValueError("synthetic validation geometry contains coincident atoms")
    cosine = sum(
        left * right for left, right in zip(hydrogen_donor, hydrogen_acceptor)
    ) / denominator
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return distance, angle


def run_validation():
    atoms = [
        AtomRecord(0, 1, "N", "", "ALA", "A", 1, "", "N"),
        AtomRecord(1, 2, "H", "", "ALA", "A", 1, "", "H"),
        AtomRecord(2, 3, "OD1", "", "ASP", "A", 2, "", "O"),
    ]
    bonds = [(0, 1)]
    roles = infer_atom_chemical_roles(atoms, bonds)
    candidates, summary = discover_automatic_candidate_bonds(
        atoms,
        bonds,
        interaction_scope="protein_protein",
        exclude_same_residue=True,
    )
    chemistry_pass = (
        roles[0].donor
        and not roles[0].acceptor
        and roles[2].acceptor
        and not roles[2].donor
        and len(candidates) == 1
        and int(candidates[0]["donor_atom_index"]) == 0
        and int(candidates[0]["hydrogen_atom_index"]) == 1
        and int(candidates[0]["acceptor_atom_index"]) == 2
    )

    cases = [
        {
            "case_id": "linear_present",
            "coordinates": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.8, 0.0, 0.0)),
            "expected_present": True,
        },
        {
            "case_id": "bent_absent",
            "coordinates": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
            "expected_present": False,
        },
        {
            "case_id": "distant_absent",
            "coordinates": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (4.2, 0.0, 0.0)),
            "expected_present": False,
        },
    ]
    geometry_rows = []
    geometry_pass = True
    for case in cases:
        donor, hydrogen, acceptor = case["coordinates"]
        reference_distance, reference_angle = _independent_geometry(
            donor, hydrogen, acceptor
        )
        reference_present = (
            reference_distance <= MAXIMUM_DISTANCE_ANGSTROM
            and reference_angle >= MINIMUM_ANGLE_DEGREES
        )
        observed_present, observed_distance, observed_angle = hydrogen_bond_present(
            donor,
            hydrogen,
            acceptor,
            maximum_distance=MAXIMUM_DISTANCE_ANGSTROM,
            minimum_angle=MINIMUM_ANGLE_DEGREES,
        )
        case_pass = (
            reference_present == bool(case["expected_present"])
            and observed_present == reference_present
            and abs(observed_distance - reference_distance) <= 1.0e-12
            and abs(observed_angle - reference_angle) <= 1.0e-12
        )
        geometry_pass = geometry_pass and case_pass
        geometry_rows.append({
            "case_id": case["case_id"],
            "passed": case_pass,
            "present": observed_present,
            "distance_angstrom": observed_distance,
            "angle_degrees": observed_angle,
        })

    passed = chemistry_pass and geometry_pass
    return {
        "schema_version": "salsbury-public-hydrogen-bond-validation-v1",
        "module_id": "hydrogen_bond_discovery",
        "fixture_kind": "public_synthetic",
        "dependencies": ["Python", "salsbury-md-analysis base runtime"],
        "chemistry": {
            "passed": chemistry_pass,
            "candidate_count": len(candidates),
            "summary": summary,
        },
        "geometry": {
            "passed": geometry_pass,
            "cases": geometry_rows,
        },
        "technical_status": "complete" if passed else "failed",
        "scientific_status": "bounded synthetic implementation check",
        "interpretation": (
            "Checks public synthetic chemistry and geometry only; it does not "
            "establish real-trajectory sampling, energetic importance, or a "
            "biological conclusion."
        ),
    }


def main():
    report = run_validation()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
