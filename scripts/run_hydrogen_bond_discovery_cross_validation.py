#!/usr/bin/env python3
"""Cross-validate automatic hydrogen-bond discovery on retained real frames.

This validation deliberately uses independent topology and geometry engines.
It writes only bounded, path-redacted evidence; the private trajectories and
their absolute storage locations remain outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import mdtraj as md
import numpy as np
from openmm import NonbondedForce
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile
from openmm.unit import elementary_charge

from salsbury_md_analysis.atom_mapping import AtomRecord, read_topology_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.hydrogen_bond_chemistry import (
    chemistry_summary,
    infer_atom_chemical_roles,
)
from salsbury_md_analysis.hydrogen_bond_discovery import (
    discover_automatic_candidate_bonds,
    hydrogen_bond_discovery_project,
)
from salsbury_md_analysis.hydrogen_bonds import hydrogen_bond_present
from salsbury_md_analysis.periodic import load_connectivity


EXPECTED_TOPOLOGY_SHA256 = "3cdcef023de68a2c4fb5bed23fbcb9623a3ceb430f540c3fba008131ebff78ce"
EXPECTED_CONNECTIVITY_SHA256 = "85970f0f38df01c0378c5191cd99c71e62de9ce180f71e7249f9cc6a00eedd36"
EXPECTED_SOURCE_DCD_SHA256 = {
    "430001": "e810b2d7f0f5a98a47372018b6fefeda866b7270e5d0c2c93f176ed8df63af75",
    "431001": "69664eb11141074d64d72cc4c437fe6fdf583d41890df26e07df5abace338cc1",
    "432001": "5782de91be81e2be6ccd613bb02ca30e2859b11a580081de2e4b6de5150b091f",
}
EXPECTED_DERIVED_XYZ_SHA256 = {
    "430001": "f15784e51f08e2ede5dd483bc94b2968d7df0ab28bc99b1403049e82af5309b2",
    "431001": "9bcddc2ca2e29d1903b2989ff948b5020245ad383fb816f13592ffd364c326b1",
    "432001": "26304ead6b19302f0662964514a310970a08014295cd2a8af2761a5e97cb93f9",
}

DISTANCE_TOLERANCE_ANGSTROM = 3.0e-5
ANGLE_TOLERANCE_DEGREES = 1.0e-3
BOUNDARY_DISTANCE_MARGIN_ANGSTROM = 1.0e-4
BOUNDARY_ANGLE_MARGIN_DEGREES = 1.0e-2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def adjacency(atom_count: int, bonds: Iterable[Tuple[int, int]]) -> Dict[int, List[int]]:
    result = {index: [] for index in range(atom_count)}
    for left, right in bonds:
        result[left].append(right)
        result[right].append(left)
    return result


def atom_pair_labels(
    pairs: Iterable[Tuple[int, int]], atoms: Sequence[AtomRecord]
) -> List[str]:
    labels = []
    for heavy, hydrogen in sorted(pairs):
        donor = atoms[heavy]
        attached = atoms[hydrogen]
        labels.append(
            f"{donor.chain_id}:{donor.residue_name}{donor.residue_number}:"
            f"{donor.atom_name}-{attached.atom_name}"
        )
    return labels


def independent_forcefield_audit(
    topology_path: Path,
    atoms: Sequence[AtomRecord],
    acceptor_indices: Sequence[int],
    donor_hydrogen_pairs: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    pdb = PDBFile(str(topology_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.delete([residue for residue in modeller.topology.residues() if residue.name == "MG"])
    openmm_atoms = list(modeller.topology.atoms())
    nonion_atoms = [atom for atom in atoms if atom.residue_name.upper() != "MG"]
    identity_matches = len(openmm_atoms) == len(nonion_atoms) and all(
        reference.name == observed.atom_name
        and reference.residue.name == observed.residue_name
        for reference, observed in zip(openmm_atoms, nonion_atoms)
    )
    if not identity_matches:
        raise ValueError("OpenMM and toolkit atom identities do not map one-to-one")

    forcefield = ForceField("charmm36_2024.xml")
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=None
    )
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, NonbondedForce)
    )
    charges = [
        nonbonded.getParticleParameters(index)[0].value_in_unit(elementary_charge)
        for index in range(system.getNumParticles())
    ]
    polar_acceptors = [
        index for index in acceptor_indices
        if atoms[index].element.upper() in {"N", "O"}
        and atoms[index].residue_name.upper() != "MG"
    ]
    acceptor_charges = [charges[index] for index in polar_acceptors]
    openmm_bonds = {
        tuple(sorted((left.index, right.index))) for left, right in modeller.topology.bonds()
    }
    donor_bonds_present = sum(
        tuple(sorted(pair)) in openmm_bonds for pair in donor_hydrogen_pairs
    )
    return {
        "forcefield": "OpenMM charmm36_2024.xml",
        "parameterized_atom_count": system.getNumParticles(),
        "topology_bond_count_after_ion_exclusion": len(openmm_bonds),
        "atom_identity_exact_match": identity_matches,
        "polar_acceptor_count": len(polar_acceptors),
        "polar_acceptor_minimum_partial_charge_e": min(acceptor_charges),
        "polar_acceptor_maximum_partial_charge_e": max(acceptor_charges),
        "polar_acceptor_charge_at_most_minus_0p4_fraction": sum(
            charge <= -0.4 for charge in acceptor_charges
        ) / len(acceptor_charges),
        "donor_hydrogen_bonds_present": donor_bonds_present,
        "donor_hydrogen_bond_count": len(donor_hydrogen_pairs),
    }


def read_fixture_frames(
    records: Sequence[Mapping[str, object]], atom_count: int
) -> Tuple[np.ndarray, List[str], List[Path], List[int]]:
    coordinates = []
    replica_ids = []
    paths = []
    counts = []
    for record in records:
        replica_id = str(record["replica_id"])
        path = Path(str(record["output_path"])).resolve(strict=True)
        paths.append(path)
        count = 0
        for frame in iter_coordinate_frames(path, "angstrom"):
            if len(frame.coordinates_angstrom) != atom_count:
                raise ValueError("derived trajectory atom count does not match topology")
            coordinates.append(frame.coordinates_angstrom)
            replica_ids.append(replica_id)
            count += 1
        counts.append(count)
    return np.asarray(coordinates, dtype=float), replica_ids, paths, counts


def candidate_triplets(candidates: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            [
                int(candidate["donor_atom_index"]),
                int(candidate["hydrogen_atom_index"]),
                int(candidate["acceptor_atom_index"]),
            ]
            for candidate in candidates
        ],
        dtype=int,
    )


def mdtraj_geometry(
    trajectory: md.Trajectory, triplets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    distances = md.compute_distances(
        trajectory, triplets[:, [0, 2]], periodic=False, opt=True
    ) * 10.0
    angles = np.degrees(
        md.compute_angles(trajectory, triplets, periodic=False, opt=True)
    )
    return np.asarray(distances, dtype=float), np.asarray(angles, dtype=float)


def sampled_suite_geometry(
    coordinates: np.ndarray,
    candidates: Sequence[Mapping[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.empty((len(coordinates), len(candidates)), dtype=float)
    angles = np.empty_like(distances)
    for frame_index, frame in enumerate(coordinates):
        for candidate_index, candidate in enumerate(candidates):
            _, distance, angle = hydrogen_bond_present(
                frame[int(candidate["donor_atom_index"])],
                frame[int(candidate["hydrogen_atom_index"])],
                frame[int(candidate["acceptor_atom_index"])],
                math.inf,
                1.0e-12,
            )
            distances[frame_index, candidate_index] = distance
            angles[frame_index, candidate_index] = angle
    return distances, angles


def cutoff_agreement(
    suite_distances: np.ndarray,
    suite_angles: np.ndarray,
    reference_distances: np.ndarray,
    reference_angles: np.ndarray,
    cutoff_definitions: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    total = 0
    exact = 0
    boundary_equivalent = 0
    nonboundary_mismatches = 0
    by_cutoff = []
    for cutoff in cutoff_definitions:
        distance_cutoff = float(cutoff["maximum_donor_acceptor_distance_angstrom"])
        angle_cutoff = float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
        suite_present = (suite_distances <= distance_cutoff) & (suite_angles >= angle_cutoff)
        reference_present = (
            (reference_distances <= distance_cutoff) & (reference_angles >= angle_cutoff)
        )
        mismatches = suite_present != reference_present
        mismatch_count = int(np.count_nonzero(mismatches))
        boundary = mismatches & (
            (np.abs(reference_distances - distance_cutoff) <= BOUNDARY_DISTANCE_MARGIN_ANGSTROM)
            | (np.abs(reference_angles - angle_cutoff) <= BOUNDARY_ANGLE_MARGIN_DEGREES)
        )
        boundary_count = int(np.count_nonzero(boundary))
        observations = int(suite_present.size)
        total += observations
        exact += observations - mismatch_count
        boundary_equivalent += boundary_count
        nonboundary_mismatches += mismatch_count - boundary_count
        by_cutoff.append({
            "cutoff_id": cutoff["cutoff_id"],
            "observation_count": observations,
            "exact_match_count": observations - mismatch_count,
            "boundary_equivalent_mismatch_count": boundary_count,
            "nonboundary_mismatch_count": mismatch_count - boundary_count,
        })
    return {
        "comparison_count": total,
        "exact_match_count": exact,
        "exact_match_fraction": exact / total,
        "boundary_equivalent_mismatch_count": boundary_equivalent,
        "nonboundary_mismatch_count": nonboundary_mismatches,
        "by_cutoff": by_cutoff,
    }


def run_integrated_nucleic_project(
    topology: Path,
    connectivity: Path,
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="salsbury-hbond-validation-") as temporary:
        root = Path(temporary)
        system = {
            "systems": [{
                "system_id": "trex_control",
                "replicas": [
                    {
                        "replica_id": str(record["replica_id"]),
                        "topology": str(topology),
                        "connectivity": str(connectivity),
                        "segments": [{
                            "segment_id": "3p01_to_243p01ns_stride100",
                            "trajectory": str(Path(str(record["output_path"])).resolve()),
                            "timing": {
                                "first_frame_time": 3.01,
                                "frame_interval": 10.0,
                                "unit": "ns",
                            },
                        }],
                    }
                    for record in records
                ],
            }],
        }
        project = {
            "project_id": "trex-hbond-discovery-cross-validation",
            "analysis_profile": "standard_md_v1",
            "system_manifest": "system.json",
            "analysis_output_root": "outputs",
            "sampling_mode": "UNBIASED_MD",
            "coordinate_unit": "angstrom",
            "time_unit": "ns",
            "periodic_coordinate_policy": "make_whole",
            "periodic_reconstruction": {
                "maximum_bond_length_angstrom": 2.5,
                "cycle_closure_tolerance_angstrom": 0.001,
            },
            "reference_connectivity": str(connectivity),
            "reference_structure": str(topology),
            "common_atom_policy": "strict",
            "selections": {
                "alignment": {"preset": "all"},
                "analysis": {"preset": "all"},
            },
            "definitions": {"hydrogen_bond_discovery": {
                "chemistry_policy": "automatic_topology_templates_v1",
                "interaction_scope": "nucleic_acid_nucleic_acid",
                "exclude_same_residue": True,
                "water_policy": "exclude",
                "frame_stride": 1,
                "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
                "maximum_reference_donor_hydrogen_bond_angstrom": 1.3,
                "maximum_candidate_bonds": 2000,
                "maximum_feature_observations": 100000,
            }},
            "requested_modules": ["hydrogen_bond_discovery"],
            "protected_locations": ["/protected/example"],
        }
        (root / "system.json").write_text(
            json.dumps(system, indent=2) + "\n", encoding="utf-8"
        )
        project_path = root / "project.json"
        project_path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
        return hydrogen_bond_discovery_project(project_path, hash_content=True)


def redacted_input_record(path: Path, kind: str, replica_id: str | None = None) -> Dict[str, object]:
    record: Dict[str, object] = {
        "kind": kind,
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if replica_id is not None:
        record["replica_id"] = replica_id
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    fixture = arguments.fixture_root.expanduser().resolve(strict=True)
    topology = fixture / "topology" / "control_no_water_topology.pdb"
    connectivity = fixture / "topology" / "control_no_water_topology.bonds.json"
    fixture_manifest_path = fixture / "derived_stride100" / "fixture_manifest.json"
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    records = sorted(fixture_manifest["records"], key=lambda row: str(row["replica_id"]))

    lineage_pass = (
        sha256(topology) == EXPECTED_TOPOLOGY_SHA256
        and sha256(connectivity) == EXPECTED_CONNECTIVITY_SHA256
        and all(
            str(record["source_sha256"]) == EXPECTED_SOURCE_DCD_SHA256[str(record["replica_id"])]
            and str(record["output_sha256"]) == EXPECTED_DERIVED_XYZ_SHA256[str(record["replica_id"])]
            and sha256(Path(str(record["output_path"]))) == EXPECTED_DERIVED_XYZ_SHA256[str(record["replica_id"])]
            for record in records
        )
    )

    _, atoms = read_topology_atoms(topology)
    bonds, connectivity_provenance = load_connectivity(connectivity, len(atoms))
    roles = infer_atom_chemical_roles(atoms, bonds)
    role_summary = chemistry_summary(roles)
    adjacency_by_atom = adjacency(len(atoms), bonds)
    donor_hydrogen_pairs = sorted({
        (index, neighbor)
        for index, role in roles.items() if role.donor
        for neighbor in adjacency_by_atom[index]
        if atoms[neighbor].element.upper() == "H"
    })
    acceptor_indices = sorted(index for index, role in roles.items() if role.acceptor)

    md_topology = md.load_pdb(str(topology)).topology
    md_donor_pairs = set()
    for left, right in md_topology.bonds:
        elements = {left.element.symbol, right.element.symbol}
        if elements not in ({"N", "H"}, {"O", "H"}):
            continue
        md_donor_pairs.add(
            (left.index, right.index)
            if left.element.symbol != "H"
            else (right.index, left.index)
        )
    suite_no_donor_pairs = {
        pair for pair in donor_hydrogen_pairs
        if atoms[pair[0]].element.upper() in {"N", "O"}
    }
    suite_only_no_pairs = suite_no_donor_pairs - md_donor_pairs
    mdtraj_only_no_pairs = md_donor_pairs - suite_no_donor_pairs
    allowed_terminal_names = {
        ("O5'", "HO5'"), ("O3'", "HO3'"), ("O2'", "HO2'")
    }
    suite_only_pairs_are_terminal_nucleic = all(
        roles[heavy].entity_class == "nucleic_acid"
        and (atoms[heavy].atom_name, atoms[hydrogen].atom_name) in allowed_terminal_names
        for heavy, hydrogen in suite_only_no_pairs
    )

    forcefield_audit = independent_forcefield_audit(
        topology, atoms, acceptor_indices, donor_hydrogen_pairs
    )
    chemistry_pass = (
        role_summary["chemistry_confidence_atom_counts"].get("provisional", 0) == 0
        and len(mdtraj_only_no_pairs) == 0
        and suite_only_pairs_are_terminal_nucleic
        and forcefield_audit["polar_acceptor_charge_at_most_minus_0p4_fraction"] == 1.0
        and forcefield_audit["donor_hydrogen_bonds_present"] == len(donor_hydrogen_pairs)
    )

    protein_nucleic_candidates, _ = discover_automatic_candidate_bonds(
        atoms, bonds,
        interaction_scope="protein_nucleic_acid",
        exclude_same_residue=True,
    )
    nucleic_candidates, _ = discover_automatic_candidate_bonds(
        atoms, bonds,
        interaction_scope="nucleic_acid_nucleic_acid",
        exclude_same_residue=True,
    )
    coordinates, frame_replica_ids, trajectory_paths, frame_counts = read_fixture_frames(
        records, len(atoms)
    )
    reference_trajectory = md.Trajectory(
        xyz=np.asarray(coordinates / 10.0, dtype=np.float32),
        topology=md_topology,
    )

    integrated = run_integrated_nucleic_project(topology, connectivity, records)
    direct_nucleic_triplets = candidate_triplets(nucleic_candidates)
    integrated_triplets = candidate_triplets(integrated["candidate_dictionary"])
    candidate_dictionary_exact = np.array_equal(
        direct_nucleic_triplets, integrated_triplets
    )
    integrated_distances = np.asarray([
        row["donor_acceptor_distances_angstrom"]
        for row in integrated["frame_bond_matrix"]
    ], dtype=float)
    integrated_angles = np.asarray([
        row["donor_hydrogen_acceptor_angles_degrees"]
        for row in integrated["frame_bond_matrix"]
    ], dtype=float)
    reference_nucleic_distances, reference_nucleic_angles = mdtraj_geometry(
        reference_trajectory, direct_nucleic_triplets
    )
    nucleic_distance_difference = float(np.max(np.abs(
        integrated_distances - reference_nucleic_distances
    )))
    nucleic_angle_difference = float(np.max(np.abs(
        integrated_angles - reference_nucleic_angles
    )))
    nucleic_cutoff_agreement = cutoff_agreement(
        integrated_distances,
        integrated_angles,
        reference_nucleic_distances,
        reference_nucleic_angles,
        integrated["cutoff_definitions"],
    )

    expected_cutoff_counts = {}
    for cutoff in integrated["cutoff_definitions"]:
        present = (
            reference_nucleic_distances
            <= float(cutoff["maximum_donor_acceptor_distance_angstrom"])
        ) & (
            reference_nucleic_angles
            >= float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
        )
        for replica_id in sorted(set(frame_replica_ids)):
            frame_mask = np.asarray([
                observed == replica_id for observed in frame_replica_ids
            ], dtype=bool)
            counts = np.sum(present[frame_mask], axis=0)
            for candidate, count in zip(nucleic_candidates, counts):
                expected_cutoff_counts[
                    (replica_id, str(candidate["bond_id"]), str(cutoff["cutoff_id"]))
                ] = int(count)
    occupancy_count_mismatches = 0
    for row in integrated["cutoff_occupancies"]:
        key = (str(row["replica_id"]), str(row["bond_id"]), str(row["cutoff_id"]))
        if int(row["present_frame_count"]) != expected_cutoff_counts[key]:
            occupancy_count_mismatches += 1

    sample_count = min(1024, len(protein_nucleic_candidates))
    sampled_indices = np.unique(np.linspace(
        0, len(protein_nucleic_candidates) - 1, sample_count, dtype=int
    ))
    sampled_protein_nucleic = [
        protein_nucleic_candidates[int(index)] for index in sampled_indices
    ]
    suite_sample_distances, suite_sample_angles = sampled_suite_geometry(
        coordinates, sampled_protein_nucleic
    )
    reference_sample_distances, reference_sample_angles = mdtraj_geometry(
        reference_trajectory, candidate_triplets(sampled_protein_nucleic)
    )
    sample_distance_difference = float(np.max(np.abs(
        suite_sample_distances - reference_sample_distances
    )))
    sample_angle_difference = float(np.max(np.abs(
        suite_sample_angles - reference_sample_angles
    )))
    sample_cutoff_agreement = cutoff_agreement(
        suite_sample_distances,
        suite_sample_angles,
        reference_sample_distances,
        reference_sample_angles,
        integrated["cutoff_definitions"],
    )
    full_protein_nucleic_reference_distances, full_protein_nucleic_reference_angles = (
        mdtraj_geometry(
            reference_trajectory, candidate_triplets(protein_nucleic_candidates)
        )
    )
    protein_nucleic_population_sensitivity = []
    event_counts_by_rule = {}
    for cutoff in integrated["cutoff_definitions"]:
        distance_cutoff = float(cutoff["maximum_donor_acceptor_distance_angstrom"])
        angle_cutoff = float(cutoff["minimum_donor_hydrogen_acceptor_angle_degrees"])
        present = (
            full_protein_nucleic_reference_distances <= distance_cutoff
        ) & (
            full_protein_nucleic_reference_angles >= angle_cutoff
        )
        event_count = int(np.count_nonzero(present))
        event_counts_by_rule[(distance_cutoff, angle_cutoff)] = event_count
        protein_nucleic_population_sensitivity.append({
            "cutoff_id": cutoff["cutoff_id"],
            "maximum_donor_acceptor_distance_angstrom": distance_cutoff,
            "minimum_donor_hydrogen_acceptor_angle_degrees": angle_cutoff,
            "present_frame_candidate_event_count": event_count,
            "candidate_with_at_least_one_present_frame_count": int(
                np.count_nonzero(np.any(present, axis=0))
            ),
            "per_replica_event_count": {
                replica_id: int(np.count_nonzero(present[np.asarray([
                    observed == replica_id for observed in frame_replica_ids
                ], dtype=bool)]))
                for replica_id in sorted(set(frame_replica_ids))
            },
        })
    sensitivity_monotonic = all(
        event_counts_by_rule[(left_distance, angle)]
        <= event_counts_by_rule[(right_distance, angle)]
        for angle in (120.0, 135.0, 150.0)
        for left_distance, right_distance in ((3.0, 3.2), (3.2, 3.5))
    ) and all(
        event_counts_by_rule[(distance, lower_angle)]
        >= event_counts_by_rule[(distance, higher_angle)]
        for distance in (3.0, 3.2, 3.5)
        for lower_angle, higher_angle in ((120.0, 135.0), (135.0, 150.0))
    )

    integrated_geometry_pass = (
        candidate_dictionary_exact
        and integrated["technical_status"] == "complete"
        and int(integrated["evaluated_frame_count"]) == len(coordinates)
        and nucleic_distance_difference <= DISTANCE_TOLERANCE_ANGSTROM
        and nucleic_angle_difference <= ANGLE_TOLERANCE_DEGREES
        and nucleic_cutoff_agreement["nonboundary_mismatch_count"] == 0
        and occupancy_count_mismatches == 0
    )
    sampled_protein_nucleic_pass = (
        sample_distance_difference <= DISTANCE_TOLERANCE_ANGSTROM
        and sample_angle_difference <= ANGLE_TOLERANCE_DEGREES
        and sample_cutoff_agreement["nonboundary_mismatch_count"] == 0
        and sensitivity_monotonic
    )
    geometry_pass = integrated_geometry_pass and sampled_protein_nucleic_pass
    overall_pass = lineage_pass and chemistry_pass and geometry_pass

    input_records = [
        redacted_input_record(topology, "topology"),
        redacted_input_record(connectivity, "connectivity"),
    ]
    input_records.extend(
        redacted_input_record(path, "derived_make_whole_trajectory", replica_id)
        for path, replica_id in zip(trajectory_paths, sorted(EXPECTED_DERIVED_XYZ_SHA256))
    )
    report = {
        "schema_version": "1.0",
        "validation_date": "2026-08-11",
        "module_id": "hydrogen_bond_discovery",
        "technical_status": "complete" if overall_pass else "failed",
        "scientific_status": (
            "scoped real-trajectory cross-validation passed"
            if overall_pass else "scoped real-trajectory cross-validation failed"
        ),
        "release_boundary": (
            "Validates standard protein/nucleic-acid automatic chemistry and direct-H-bond geometry on a retained TREX control fixture. "
            "It does not validate arbitrary ligand protonation, water-mediated networks, sampling adequacy, or a biological conclusion."
        ),
        "dataset": {
            "description": "Retained private three-replica TREX control; public evidence contains only hashes and bounded metrics.",
            "replica_count": len(records),
            "source_frame_count_per_replica": [int(row["observed_frame_count"]) for row in records],
            "selected_frame_count_per_replica": frame_counts,
            "selected_frame_count": len(coordinates),
            "selected_time_range_ns": [3.01, 243.01],
            "atom_count": len(atoms),
            "explicit_connectivity_bond_count": connectivity_provenance["bond_count"],
            "input_records": input_records,
            "fixture_manifest_sha256": sha256(fixture_manifest_path),
        },
        "reference_environment_versions": {
            "MDTraj": compact_version("mdtraj"),
            "OpenMM": compact_version("OpenMM"),
            "NumPy": compact_version("numpy"),
        },
        "cases": [
            {
                "case_id": "authoritative_trex_hbond_fixture_lineage",
                "passed": lineage_pass,
                "reference": "recorded topology, connectivity, source-DCD, and derived-frame SHA-256 values",
                "metrics": {
                    "replica_count": len(records),
                    "selected_frame_count": len(coordinates),
                    "all_hashes_matched": lineage_pass,
                },
                "tolerances": {"all_hashes_must_match": True},
            },
            {
                "case_id": "automatic_chemistry_independent_topology_and_forcefield_crosscheck",
                "passed": chemistry_pass,
                "reference": "MDTraj PDB bonding and OpenMM CHARMM36 2024 residue templates/partial charges",
                "metrics": {
                    "chemistry_summary": role_summary,
                    "donor_hydrogen_pair_count": len(donor_hydrogen_pairs),
                    "mdtraj_common_n_o_donor_pair_count": len(
                        suite_no_donor_pairs & md_donor_pairs
                    ),
                    "mdtraj_only_n_o_donor_pair_count": len(mdtraj_only_no_pairs),
                    "suite_only_terminal_nucleic_donor_pair_count": len(suite_only_no_pairs),
                    "suite_only_terminal_nucleic_donor_pairs": atom_pair_labels(
                        suite_only_no_pairs, atoms
                    ),
                    "forcefield_audit": forcefield_audit,
                },
                "tolerances": {
                    "provisional_atom_count": 0,
                    "mdtraj_only_n_o_donor_pair_count": 0,
                    "suite_only_pairs": "explicitly bonded terminal nucleic hydroxyls only",
                    "polar_acceptor_partial_charge_e_maximum": -0.4,
                },
            },
            {
                "case_id": "integrated_nucleic_hbond_discovery_mdtraj_crosscheck",
                "passed": integrated_geometry_pass,
                "reference": "MDTraj independent distance and angle kernels on all retained frames",
                "metrics": {
                    "nucleic_candidate_count": len(nucleic_candidates),
                    "integrated_feature_observation_count": integrated["feature_observation_count"],
                    "candidate_dictionary_exact_match": candidate_dictionary_exact,
                    "maximum_distance_difference_angstrom": nucleic_distance_difference,
                    "maximum_angle_difference_degrees": nucleic_angle_difference,
                    "cutoff_agreement": nucleic_cutoff_agreement,
                    "cutoff_occupancy_count_mismatch_count": occupancy_count_mismatches,
                },
                "tolerances": {
                    "distance_angstrom": DISTANCE_TOLERANCE_ANGSTROM,
                    "angle_degrees": ANGLE_TOLERANCE_DEGREES,
                    "nonboundary_cutoff_mismatch_count": 0,
                    "cutoff_occupancy_count_mismatch_count": 0,
                },
            },
            {
                "case_id": "protein_nucleic_scope_deterministic_sample_mdtraj_crosscheck",
                "passed": sampled_protein_nucleic_pass,
                "reference": "MDTraj independent geometry on an outcome-independent evenly spaced candidate sample",
                "metrics": {
                    "full_protein_nucleic_candidate_count": len(protein_nucleic_candidates),
                    "sampled_candidate_count": len(sampled_protein_nucleic),
                    "sampled_frame_count": len(coordinates),
                    "maximum_distance_difference_angstrom": sample_distance_difference,
                    "maximum_angle_difference_degrees": sample_angle_difference,
                    "cutoff_agreement": sample_cutoff_agreement,
                    "full_candidate_reference_population_sensitivity": protein_nucleic_population_sensitivity,
                    "population_sensitivity_is_monotonic": sensitivity_monotonic,
                },
                "tolerances": {
                    "distance_angstrom": DISTANCE_TOLERANCE_ANGSTROM,
                    "angle_degrees": ANGLE_TOLERANCE_DEGREES,
                    "nonboundary_cutoff_mismatch_count": 0,
                    "population_sensitivity_is_monotonic": True,
                },
            },
        ],
        "overall_pass": overall_pass,
        "limitations": [
            "The real fixture contains standard protein, DNA, and Mg ions but no ligand or water; provisional generic-ligand chemistry and water-mediated networks are outside this case.",
            "Protein-nucleic geometry uses an outcome-independent 1,024-candidate sample because the full 63,228-candidate frame matrix would create a disproportionate validation artifact; all 1,220 nucleic-nucleic candidates are checked through the integrated project runner.",
            "Automatic discovery retains every chemically eligible donor-hydrogen-acceptor triple, so candidate enumeration and report volume scale with donor-hydrogen pairs times acceptors times frames times cutoff rules; production projects must set and review the existing resource gates.",
            "The two MDTraj donor-bond omissions are terminal DNA O5'-HO5' pairs confirmed by both explicit connectivity and OpenMM, so they are retained as expected reference differences.",
            "Agreement establishes implementation fidelity for the scoped definitions, not energetic importance, adequate sampling, convergence, or a biological mechanism.",
        ],
    }
    output = arguments.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "technical_status": report["technical_status"],
        "scientific_status": report["scientific_status"],
        "case_count": len(report["cases"]),
        "passed_case_count": sum(case["passed"] for case in report["cases"]),
        "output": str(output),
        "output_sha256": sha256(output),
    }, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
