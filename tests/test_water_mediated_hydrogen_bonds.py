import json
import random
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.hydrogen_bond_discovery import _cutoff_definitions
from salsbury_md_analysis.hydrogen_bonds import distance_angstrom
from salsbury_md_analysis.water_mediated_hydrogen_bonds import (
    WaterMediatedHydrogenBondError,
    discover_solute_endpoints,
    discover_waters,
    _frame_selection_plan,
    evaluate_water_bridge_frame,
    neighbor_pairs_within,
    water_mediated_hydrogen_bond_networks_project,
)


def _pdb_atom(serial, name, residue, chain, residue_number, xyz, element, record="ATOM"):
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {name:^4} {residue:>3} {chain:1}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


class WaterMediatedHydrogenBondTests(unittest.TestCase):
    def test_connectivity_keeps_wrapped_duplicate_water_residue_ids_separate(self):
        atoms = [
            AtomRecord(0, 1, "OH2", "", "TIP", "W", 2, "", "O"),
            AtomRecord(1, 2, "H1", "", "TIP", "W", 2, "", "H"),
            AtomRecord(2, 3, "H2", "", "TIP", "W", 2, "", "H"),
            AtomRecord(3, 4, "OH2", "", "TIP", "W", 2, "", "O"),
            AtomRecord(4, 5, "H1", "", "TIP", "W", 2, "", "H"),
            AtomRecord(5, 6, "H2", "", "TIP", "W", 2, "", "H"),
        ]
        waters = discover_waters(atoms, [(0, 1), (0, 2), (3, 4), (3, 5)])
        self.assertEqual(len(waters), 2)
        self.assertEqual(
            [water["oxygen_atom_index"] for water in waters], [0, 3]
        )
        self.assertEqual(len({water["water_id"] for water in waters}), 2)

    def test_uniform_per_replica_budget_covers_all_replicas_and_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xyz = (
                "1\nf0\nC 0 0 0\n"
                "1\nf1\nC 1 0 0\n"
                "1\nf2\nC 2 0 0\n"
            )
            for name in ("r1a.xyz", "r1b.xyz", "r2a.xyz", "r2b.xyz"):
                (root / name).write_text(xyz, encoding="ascii")
            system = {"systems": [{"system_id": "test", "replicas": [
                {"replica_id": "r1", "segments": [
                    {"segment_id": "a", "trajectory": "r1a.xyz"},
                    {"segment_id": "b", "trajectory": "r1b.xyz"},
                ]},
                {"replica_id": "r2", "segments": [
                    {"segment_id": "a", "trajectory": "r2a.xyz"},
                    {"segment_id": "b", "trajectory": "r2b.xyz"},
                ]},
            ]}]}
            settings = {
                "frame_selection": {
                    "mode": "uniform_per_replica_budget_v1",
                    "maximum_frames_per_replica": 2,
                },
                "maximum_evaluated_frames": 4,
                "frame_stride": 1,
            }
            plan, report = _frame_selection_plan(
                system, root / "system.json", "angstrom", settings
            )
        self.assertEqual(plan[("test", "r1", "a")], {0})
        self.assertEqual(plan[("test", "r1", "b")], {2})
        self.assertEqual(plan[("test", "r2", "a")], {0})
        self.assertEqual(plan[("test", "r2", "b")], {2})
        self.assertEqual(report["selected_frame_count"], 4)

    def test_direct_and_water_mediated_coincidence_is_explicit(self):
        atoms = [
            AtomRecord(0, 1, "N", "", "ALA", "A", 1, "", "N"),
            AtomRecord(1, 2, "H", "", "ALA", "A", 1, "", "H"),
            AtomRecord(2, 3, "O", "", "GLY", "A", 2, "", "O"),
            AtomRecord(3, 4, "O", "", "HOH", "W", 10, "", "O"),
            AtomRecord(4, 5, "H1", "", "HOH", "W", 10, "", "H"),
            AtomRecord(5, 6, "H2", "", "HOH", "W", 10, "", "H"),
        ]
        bonds = [(0, 1), (3, 4), (3, 5)]
        endpoints, roles, donor_hydrogens = discover_solute_endpoints(atoms, bonds)
        waters = discover_waters(atoms, bonds)
        result = evaluate_water_bridge_frame(
            [(0, 0, 0), (1, 0, 0), (2.8, 0, 0),
             (1.4, 0, 0), (2.0, 0, 0), (1.4, 1, 0)],
            atoms, endpoints, roles, donor_hydrogens, waters,
            _cutoff_definitions({"preset": "mdanalysis_compatible_v1"}),
            {
                "interaction_scope": "protein_protein",
                "exclude_same_residue_endpoints": True,
                "maximum_neighbor_pairs_per_frame": 100,
                "maximum_bridge_paths_per_frame": 1000,
            },
            None,
        )
        primary_paths = result["paths_by_cutoff"][0]
        self.assertEqual(len(primary_paths), 1)
        self.assertEqual(primary_paths[0]["relation"], "donor_acceptor")
        self.assertTrue(primary_paths[0]["direct_hydrogen_bond_present"])

    def test_periodic_cell_list_matches_exact_minimum_image(self):
        coordinates = [
            (0.2, 0.2, 0.2),
            (9.8, 0.2, 0.2),
            (5.0, 5.0, 5.0),
        ]
        cell = ((10.0, 0.0, 0.0), (2.0, 9.0, 0.0), (1.0, 1.0, 8.0))
        pairs = neighbor_pairs_within(
            coordinates, [0], [1, 2], 1.0, cell, maximum_pairs=10
        )
        expected = [
            (0, right, distance_angstrom(coordinates[0], coordinates[right], cell))
            for right in (1, 2)
            if distance_angstrom(coordinates[0], coordinates[right], cell) <= 1.0
        ]
        self.assertEqual(pairs, expected)

    def test_neighbor_pair_resource_gate_fails_closed(self):
        with self.assertRaises(WaterMediatedHydrogenBondError):
            neighbor_pairs_within(
                [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], [0], [1], 1.0,
                None, maximum_pairs=0,
            )

    def test_scope_is_applied_before_within_water_pair_expansion(self):
        atoms = [
            AtomRecord(0, 1, "N", "", "ALA", "A", 1, "", "N"),
            AtomRecord(1, 2, "H", "", "ALA", "A", 1, "", "H"),
            AtomRecord(2, 3, "O", "", "GLY", "A", 2, "", "O"),
            AtomRecord(3, 4, "N3", "", "DG", "B", 4, "", "N"),
            AtomRecord(4, 5, "O", "", "HOH", "W", 10, "", "O"),
            AtomRecord(5, 6, "H1", "", "HOH", "W", 10, "", "H"),
            AtomRecord(6, 7, "H2", "", "HOH", "W", 10, "", "H"),
        ]
        bonds = [(0, 1), (4, 5), (4, 6)]
        endpoints, roles, donor_hydrogens = discover_solute_endpoints(atoms, bonds)
        waters = discover_waters(atoms, bonds)
        result = evaluate_water_bridge_frame(
            [(0, 0, 0), (1, 0, 0), (5.6, 0, 0), (5.6, 0, 0),
             (2.8, 0, 0), (3.8, 0, 0), (2.8, 1, 0)],
            atoms, endpoints, roles, donor_hydrogens, waters,
            _cutoff_definitions({"preset": "mdanalysis_compatible_v1"}),
            {
                "interaction_scope": "protein_nucleic_acid",
                "exclude_same_residue_endpoints": True,
                "maximum_neighbor_pairs_per_frame": 100,
                "maximum_bridge_paths_per_frame": 1000,
            },
            None,
        )
        paths = result["paths_by_cutoff"][0]
        self.assertEqual(len(paths), 2)
        for path in paths:
            identities = {
                atoms[int(edge["endpoint_atom_index"])].residue_name
                for edge in path["edges"]
            }
            self.assertIn("DG", identities)
            self.assertTrue(identities.intersection({"ALA", "GLY"}))

    def test_triclinic_cell_list_matches_bruteforce_for_seeded_points(self):
        generator = random.Random(20260811)
        cell = ((18.0, 0.0, 0.0), (4.0, 16.0, 0.0), (2.0, 3.0, 14.0))
        fractional = [tuple(generator.random() for _ in range(3)) for _ in range(30)]
        coordinates = [
            tuple(sum(point[index] * cell[index][axis] for index in range(3))
                  for axis in range(3))
            for point in fractional
        ]
        left, right, cutoff = list(range(10)), list(range(10, 30)), 4.2
        observed = neighbor_pairs_within(
            coordinates, left, right, cutoff, cell, maximum_pairs=1000
        )
        expected = sorted(
            (first, second, distance_angstrom(coordinates[first], coordinates[second], cell))
            for first in left for second in right
            if distance_angstrom(coordinates[first], coordinates[second], cell) <= cutoff
        )
        self.assertEqual(observed, expected)

    def test_make_whole_stride_skips_unselected_frames_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def model(hydrogen_x):
                return "".join([
                    _pdb_atom(1, "N", "ALA", "A", 1, (0, 0, 0), "N"),
                    _pdb_atom(2, "H", "ALA", "A", 1, (hydrogen_x, 0, 0), "H"),
                    _pdb_atom(3, "O", "GLY", "A", 2, (5.6, 0, 0), "O"),
                    _pdb_atom(4, "O", "HOH", "W", 10, (2.8, 0, 0), "O", "HETATM"),
                    _pdb_atom(5, "H1", "HOH", "W", 10, (3.8, 0, 0), "H", "HETATM"),
                    _pdb_atom(6, "H2", "HOH", "W", 10, (2.8, 1, 0), "H", "HETATM"),
                ])

            crystal = "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n"
            (root / "reference.pdb").write_text(
                crystal + model(1.0) + "END\n", encoding="ascii"
            )
            (root / "trajectory.pdb").write_text(
                crystal + "MODEL        1\n" + model(1.0) + "ENDMDL\n"
                + "MODEL        2\n" + model(10.0) + "ENDMDL\nEND\n",
                encoding="ascii",
            )
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 6, "index_base": 0,
                "bonds": [[0, 1], [3, 4], [3, 5]],
            }), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({
                "systems": [{"system_id": "test", "replicas": [{
                    "replica_id": "r1", "topology": "reference.pdb",
                    "connectivity": "bonds.json", "segments": [{
                        "segment_id": "s1", "trajectory": "trajectory.pdb",
                        "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                    }],
                }]}],
            }), encoding="utf-8")
            project = {
                "project_id": "water-stride-test", "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json", "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
                "time_unit": "ps", "periodic_coordinate_policy": "make_whole",
                "periodic_reconstruction": {
                    "maximum_bond_length_angstrom": 2.5,
                    "cycle_closure_tolerance_angstrom": 0.01,
                },
                "reference_structure": "reference.pdb", "reference_connectivity": "bonds.json",
                "common_atom_policy": "strict",
                "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
                "definitions": {"water_mediated_hydrogen_bond_networks": {
                    "chemistry_policy": "automatic_topology_templates_v1",
                    "interaction_scope": "protein_protein",
                    "water_identity_policy": "standard_residue_names_connectivity_v1",
                    "maximum_bridge_length": 1, "exclude_same_residue_endpoints": True,
                    "frame_stride": 2, "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
                    "frame_selection": {"mode": "fixed_stride_v1"},
                    "maximum_reference_donor_hydrogen_bond_angstrom": 1.2,
                    "neighbor_search": "cell_list_v1", "maximum_solute_endpoints": 10,
                    "maximum_waters": 10, "maximum_evaluated_frames": 10,
                    "maximum_neighbor_pairs_per_frame": 100,
                    "maximum_bridge_paths_per_frame": 1000, "maximum_sparse_records": 10000,
                }},
                "requested_modules": ["water_mediated_hydrogen_bond_networks"],
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = water_mediated_hydrogen_bond_networks_project(project_path)
        self.assertEqual(report["evaluated_frame_count"], 1)

    def test_one_water_network_tracks_exchange_residence_and_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = "".join([
                _pdb_atom(1, "N", "ALA", "A", 1, (0.0, 0.0, 0.0), "N"),
                _pdb_atom(2, "H", "ALA", "A", 1, (1.0, 0.0, 0.0), "H"),
                _pdb_atom(3, "O", "GLY", "A", 2, (5.6, 0.0, 0.0), "O"),
                _pdb_atom(4, "O", "HOH", "W", 10, (2.8, 0.0, 0.0), "O", "HETATM"),
                _pdb_atom(5, "H1", "HOH", "W", 10, (3.8, 0.0, 0.0), "H", "HETATM"),
                _pdb_atom(6, "H2", "HOH", "W", 10, (2.8, 1.0, 0.0), "H", "HETATM"),
                _pdb_atom(7, "O", "HOH", "W", 11, (20.0, 0.0, 0.0), "O", "HETATM"),
                _pdb_atom(8, "H1", "HOH", "W", 11, (21.0, 0.0, 0.0), "H", "HETATM"),
                _pdb_atom(9, "H2", "HOH", "W", 11, (20.0, 1.0, 0.0), "H", "HETATM"),
                "END\n",
            ])
            (root / "reference.pdb").write_text(topology, encoding="ascii")

            def xyz_frame(label, water1, water2):
                coordinates = [
                    ("N", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)),
                    ("O", (5.6, 0.0, 0.0)),
                    ("O", water1), ("H", (water1[0] + 1.0, water1[1], water1[2])),
                    ("H", (water1[0], water1[1] + 1.0, water1[2])),
                    ("O", water2), ("H", (water2[0] + 1.0, water2[1], water2[2])),
                    ("H", (water2[0], water2[1] + 1.0, water2[2])),
                ]
                return "9\n" + label + "\n" + "".join(
                    f"{element} {point[0]} {point[1]} {point[2]}\n"
                    for element, point in coordinates
                )

            (root / "trajectory.xyz").write_text(
                xyz_frame("water-1", (2.8, 0.0, 0.0), (20.0, 0.0, 0.0))
                + xyz_frame("water-2", (20.0, 0.0, 0.0), (2.8, 0.0, 0.0))
                + xyz_frame("absent", (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)),
                encoding="ascii",
            )
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 9, "index_base": 0,
                "bonds": [[0, 1], [3, 4], [3, 5], [6, 7], [6, 8]],
            }), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({
                "systems": [{"system_id": "test", "replicas": [{
                    "replica_id": "r1", "topology": "reference.pdb",
                    "connectivity": "bonds.json", "segments": [{
                        "segment_id": "s1", "trajectory": "trajectory.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                    }],
                }]}],
            }), encoding="utf-8")
            project = {
                "project_id": "water-hbond-test", "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json", "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
                "time_unit": "ps", "periodic_coordinate_policy": "reject",
                "reference_structure": "reference.pdb", "common_atom_policy": "strict",
                "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
                "definitions": {"water_mediated_hydrogen_bond_networks": {
                    "chemistry_policy": "automatic_topology_templates_v1",
                    "interaction_scope": "protein_protein",
                    "water_identity_policy": "standard_residue_names_connectivity_v1",
                    "maximum_bridge_length": 1,
                    "exclude_same_residue_endpoints": True,
                    "frame_stride": 1,
                    "frame_selection": {"mode": "fixed_stride_v1"},
                    "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
                    "maximum_reference_donor_hydrogen_bond_angstrom": 1.2,
                    "neighbor_search": "cell_list_v1", "maximum_solute_endpoints": 10,
                    "maximum_waters": 10, "maximum_evaluated_frames": 10,
                    "maximum_neighbor_pairs_per_frame": 100,
                    "maximum_bridge_paths_per_frame": 1000,
                    "maximum_sparse_records": 10000,
                }},
                "requested_modules": ["water_mediated_hydrogen_bond_networks"],
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = water_mediated_hydrogen_bond_networks_project(project_path)

        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["evaluated_frame_count"], 3)
        self.assertLess(report["evaluated_neighbor_pair_count"], 12)
        self.assertEqual(len(report["water_dictionary"]), 2)
        self.assertEqual(len(report["observed_bridge_dictionary"]), 1)
        bridge_id = report["observed_bridge_dictionary"][0]["bridge_id"]
        primary = [
            row for row in report["bridge_occupancies"]
            if row["cutoff_id"] == "primary" and row["bridge_id"] == bridge_id
        ][0]
        self.assertAlmostEqual(primary["occupancy_fraction"], 2.0 / 3.0)
        self.assertEqual(primary["direct_coincident_frame_count"], 0)
        any_run = report["any_water_bridge_residence_runs"][0]
        self.assertEqual(any_run["sampled_frame_count"], 2)
        self.assertTrue(any_run["left_censored"])
        self.assertFalse(any_run["right_censored"])
        self.assertEqual(
            sorted(row["sampled_frame_count"] for row in report["same_water_bridge_residence_runs"]),
            [1, 1],
        )
        self.assertEqual(report["representative_frames"][0]["source_frame_index"], 0)
        self.assertTrue(any(
            row["cutoff_id"].startswith("sensitivity_")
            for row in report["bridge_occupancies"]
        ))


if __name__ == "__main__":
    unittest.main()
