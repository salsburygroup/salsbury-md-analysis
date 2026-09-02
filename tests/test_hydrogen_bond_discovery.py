import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.hydrogen_bond_discovery import (
    HydrogenBondDiscoveryError,
    _chemical_position_key,
    _conceptual_endpoint_candidate_stratum_counts,
    discover_automatic_candidate_bonds,
    discover_candidate_bonds,
    hydrogen_bond_discovery_project,
)
from salsbury_md_analysis.hydrogen_bond_chemistry import infer_atom_chemical_roles
from salsbury_md_analysis.hydrogen_bond_sparse import (
    CompiledSparseHydrogenBondEvaluator,
    LazySpatialHydrogenBondEvaluator,
    SparseHydrogenBondError,
    dense_primary_values,
    evaluate_sparse_frame,
    pack_sparse_cutoff_counts,
    pack_sparse_present_geometry,
    packed_present_indices,
    unpack_sparse_cutoff_counts,
    unpack_sparse_present_events,
)


def _atom(index, name, element, residue):
    return AtomRecord(
        atom_index=index, serial=index + 1, atom_name=name, altloc="",
        residue_name="ALA", chain_id="A", residue_number=residue,
        insertion_code="", element=element,
    )


class HydrogenBondDiscoveryTests(unittest.TestCase):
    def test_comparison_identity_ignores_atom_indices_and_residue_names(self):
        control = (
            _atom(0, "N", "N", 1).as_dict(),
            _atom(1, "H", "H", 1).as_dict(),
            _atom(2, "O6", "O", 2).as_dict(),
        )
        variant = [dict(row) for row in control]
        for index, row in enumerate(variant):
            row["atom_index"] = 100 + index
            row["serial"] = 900 + index
        variant[2]["residue_name"] = "8OG"
        self.assertEqual(
            _chemical_position_key(control), _chemical_position_key(variant)
        )

    def test_conceptual_strata_are_counted_without_cartesian_materialization(self):
        atom = lambda chain, residue, name: (chain, residue, "", name, "")
        donors = [
            (atom("A", 1, "N"), atom("A", 1, "H"), "protein"),
            (atom("A", 2, "N"), atom("A", 2, "H"), "protein"),
            (atom("B", 1, "N"), atom("B", 1, "H"), "nucleic_acid"),
        ]
        acceptors = [
            (atom("A", 1, "O"), "protein"),
            (atom("A", 3, "O"), "protein"),
            (atom("B", 1, "O"), "nucleic_acid"),
            (atom("B", 2, "O"), "nucleic_acid"),
        ]
        self.assertEqual(
            _conceptual_endpoint_candidate_stratum_counts(
                donors,
                acceptors,
                interaction_scope="all_solute",
                exclude_same_residue=True,
            ),
            {
                "nucleic_acid_to_nucleic_acid": 1,
                "nucleic_acid_to_protein": 2,
                "protein_to_nucleic_acid": 4,
                "protein_to_protein": 3,
            },
        )

    def test_packed_sparse_frame_round_trip_preserves_cutoffs_and_geometry(self):
        cutoffs = [
            {"cutoff_id": "primary"},
            {"cutoff_id": "sensitivity"},
        ]
        frame = {
            "candidate_count": 9,
            **pack_sparse_present_geometry([
                {
                    "candidate_index": 2,
                    "donor_acceptor_distance_angstrom": 2.875,
                    "donor_hydrogen_acceptor_angle_degrees": 171.25,
                    "present_cutoff_ids": ["primary", "sensitivity"],
                },
                {
                    "candidate_index": 7,
                    "donor_acceptor_distance_angstrom": 3.25,
                    "donor_hydrogen_acceptor_angle_degrees": 145.5,
                    "present_cutoff_ids": ["sensitivity"],
                },
            ], cutoffs, 9),
        }
        events = unpack_sparse_present_events(frame)
        self.assertEqual(packed_present_indices(frame, "primary"), [2])
        self.assertEqual(packed_present_indices(frame, "sensitivity"), [2, 7])
        self.assertAlmostEqual(
            float(events["donor_acceptor_distance_angstrom"][0]), 2.875, places=6
        )
        self.assertEqual(dense_primary_values(frame, 9), [0, 0, 1, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(SparseHydrogenBondError):
            dense_primary_values(frame, 8)
        packed_counts = pack_sparse_cutoff_counts(
            [{2: 4}, {2: 4, 7: 1}], cutoffs, 9, 5
        )
        count_rows = unpack_sparse_cutoff_counts(packed_counts)
        self.assertEqual(
            [tuple(int(value) for value in row) for row in count_rows.tolist()],
            [(2, 0, 4), (2, 1, 4), (7, 1, 1)],
        )

    def test_three_character_charmm_tip_is_classified_as_water(self):
        atoms = [
            AtomRecord(0, 1, "OH2", "", "TIP", "W", 1, "", "O"),
            AtomRecord(1, 2, "H1", "", "TIP", "W", 1, "", "H"),
            AtomRecord(2, 3, "H2", "", "TIP", "W", 1, "", "H"),
        ]
        roles = infer_atom_chemical_roles(atoms, [(0, 1), (0, 2)])
        self.assertEqual(roles[0].entity_class, "water")
        self.assertFalse(roles[0].donor)
        self.assertFalse(roles[0].acceptor)
        self.assertEqual(roles[0].confidence, "excluded")

    def test_compiled_sparse_geometry_matches_scalar_for_periodic_candidates(self):
        candidates = [
            {
                "bond_id": "near-boundary",
                "donor_atom_index": 0,
                "hydrogen_atom_index": 1,
                "acceptor_atom_index": 2,
            },
            {
                "bond_id": "far",
                "donor_atom_index": 0,
                "hydrogen_atom_index": 1,
                "acceptor_atom_index": 3,
            },
        ]
        cutoffs = [
            {
                "cutoff_id": "primary",
                "maximum_donor_acceptor_distance_angstrom": 3.0,
                "minimum_donor_hydrogen_acceptor_angle_degrees": 150.0,
            },
            {
                "cutoff_id": "sensitive",
                "maximum_donor_acceptor_distance_angstrom": 3.5,
                "minimum_donor_hydrogen_acceptor_angle_degrees": 120.0,
            },
        ]
        coordinates = [
            (9.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (2.3, 0.0, 0.0),
            (5.0, 5.0, 5.0),
        ]
        cell = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
        scalar = evaluate_sparse_frame(
            coordinates, candidates, cutoffs, cell=cell, chunk_size=1
        )
        compiled = CompiledSparseHydrogenBondEvaluator.compile(
            candidates, cutoffs, chunk_size=1
        ).evaluate(coordinates, cell=cell)
        self.assertEqual(compiled["geometry_engine"], "spatial_cell_list_exact_periodic_v1")
        self.assertEqual(compiled["evaluated_candidate_count"], 2)
        self.assertEqual(compiled["explicit_geometry_evaluation_count"], 1)
        self.assertEqual(
            compiled["present_candidate_indices_by_cutoff"],
            scalar["present_candidate_indices_by_cutoff"],
        )
        self.assertEqual(
            [row["candidate_index"] for row in compiled["present_geometry"]],
            [row["candidate_index"] for row in scalar["present_geometry"]],
        )
        for compiled_row, scalar_row in zip(
            compiled["present_geometry"], scalar["present_geometry"]
        ):
            self.assertAlmostEqual(
                compiled_row["donor_acceptor_distance_angstrom"],
                scalar_row["donor_acceptor_distance_angstrom"],
                places=12,
            )
            self.assertAlmostEqual(
                compiled_row["donor_hydrogen_acceptor_angle_degrees"],
                scalar_row["donor_hydrogen_acceptor_angle_degrees"],
                places=12,
            )

    def test_lazy_spatial_evaluator_never_materializes_far_cartesian_pairs(self):
        donor_rows = [{
            "donor_atom_index": 0,
            "hydrogen_atom_index": 1,
            "donor_identity_key": ("A", 1, "", "N", ""),
            "hydrogen_identity_key": ("A", 1, "", "H", ""),
            "entity_class": "protein",
            "residue_key": ("A", 1, ""),
        }]
        acceptors = [
            {
                "acceptor_atom_index": 2,
                "acceptor_identity_key": ("A", 2, "", "O", ""),
                "entity_class": "protein",
                "residue_key": ("A", 2, ""),
            },
            {
                "acceptor_atom_index": 3,
                "acceptor_identity_key": ("A", 3, "", "O", ""),
                "entity_class": "protein",
                "residue_key": ("A", 3, ""),
            },
        ]
        report = LazySpatialHydrogenBondEvaluator(
            donor_rows,
            acceptors,
            [{
                "cutoff_id": "primary",
                "maximum_donor_acceptor_distance_angstrom": 3.5,
                "minimum_donor_hydrogen_acceptor_angle_degrees": 150.0,
            }],
            "all_solute",
            True,
            10,
        ).evaluate(
            [(9.5, 0, 0), (0.5, 0, 0), (2.3, 0, 0), (5, 5, 5)],
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        )
        self.assertEqual(report["spatial_neighbor_pair_count"], 1)
        self.assertEqual(report["explicit_geometry_evaluation_count"], 1)
        self.assertEqual(len(report["present_events"]), 1)
        self.assertEqual(
            report["present_events"][0]["interaction_stratum"],
            "protein_to_protein",
        )

    def test_standard_templates_and_generic_ligands_are_auditable(self):
        atoms = [
            _atom(0, "N", "N", 1),
            _atom(1, "H", "H", 1),
            AtomRecord(2, 3, "O", "", "ASP", "A", 2, "", "O"),
            AtomRecord(3, 4, "N1", "", "LIG", "B", 3, "", "N"),
            AtomRecord(4, 5, "H1", "", "LIG", "B", 3, "", "H"),
            AtomRecord(5, 6, "O1", "", "LIG", "B", 3, "", "O"),
        ]
        roles = infer_atom_chemical_roles(atoms, [(0, 1), (3, 4)])
        self.assertTrue(roles[0].donor)
        self.assertFalse(roles[0].acceptor)  # backbone amide N is not an acceptor
        self.assertTrue(roles[2].acceptor)
        self.assertEqual(roles[3].confidence, "provisional")
        self.assertTrue(roles[3].donor)
        self.assertTrue(roles[5].acceptor)

    def test_8oxog_is_templated_nucleic_acid_not_provisional_ligand(self):
        atoms = [
            AtomRecord(0, 1, "N7", "", "8OG", "C", 4, "", "N"),
            AtomRecord(1, 2, "H7", "", "8OG", "C", 4, "", "H"),
            AtomRecord(2, 3, "O8", "", "8OG", "C", 4, "", "O"),
            AtomRecord(3, 4, "N3", "", "8OG", "C", 4, "", "N"),
            AtomRecord(4, 5, "N1", "", "8OG", "C", 4, "", "N"),
            AtomRecord(5, 6, "H1", "", "8OG", "C", 4, "", "H"),
        ]
        roles = infer_atom_chemical_roles(atoms, [(0, 1), (4, 5)])
        self.assertEqual(roles[0].entity_class, "nucleic_acid")
        self.assertEqual(roles[0].confidence, "template")
        self.assertTrue(roles[0].donor)
        self.assertFalse(roles[0].acceptor)
        self.assertTrue(roles[2].acceptor)
        self.assertTrue(roles[3].acceptor)
        self.assertTrue(roles[4].donor)

    def test_edu_is_templated_as_uracil_like_nucleic_acid(self):
        atoms = [
            AtomRecord(0, 1, "O2", "", "EDU", "C", 4, "", "O"),
            AtomRecord(1, 2, "O4", "", "EDU", "C", 4, "", "O"),
            AtomRecord(2, 3, "N3", "", "EDU", "C", 4, "", "N"),
            AtomRecord(3, 4, "H3", "", "EDU", "C", 4, "", "H"),
        ]
        roles = infer_atom_chemical_roles(atoms, [(2, 3)])
        self.assertEqual(roles[0].entity_class, "nucleic_acid")
        self.assertEqual(roles[0].confidence, "template")
        self.assertTrue(roles[0].acceptor)
        self.assertTrue(roles[1].acceptor)
        self.assertTrue(roles[2].donor)

    def test_charmm_histidine_aliases_use_protein_templates(self):
        atoms = [
            AtomRecord(0, 1, "ND1", "", "HSD", "P", 10, "", "N"),
            AtomRecord(1, 2, "HD1", "", "HSD", "P", 10, "", "H"),
            AtomRecord(2, 3, "NE2", "", "HSD", "P", 10, "", "N"),
        ]
        roles = infer_atom_chemical_roles(atoms, [(0, 1)])
        self.assertEqual(roles[0].entity_class, "protein")
        self.assertTrue(roles[0].donor)
        self.assertFalse(roles[0].acceptor)
        self.assertEqual(roles[2].entity_class, "protein")
        self.assertTrue(roles[2].acceptor)

    def test_charmm_ion_aliases_are_excluded_not_provisional_ligands(self):
        atoms = [
            AtomRecord(0, 1, "SOD", "", "SOD", "I", 1, "", "NA"),
            AtomRecord(1, 2, "POT", "", "POT", "I", 2, "", "K"),
            AtomRecord(2, 3, "CLA", "", "CLA", "I", 3, "", "CL"),
        ]
        roles = infer_atom_chemical_roles(atoms, [])
        for index in range(3):
            self.assertEqual(roles[index].entity_class, "ion")
            self.assertEqual(roles[index].confidence, "excluded")
            self.assertFalse(roles[index].donor)
            self.assertFalse(roles[index].acceptor)

    def test_automatic_discovery_uses_scope_not_atom_indices(self):
        atoms = [
            _atom(0, "N", "N", 1), _atom(1, "H", "H", 1),
            AtomRecord(2, 3, "O1", "", "LIG", "B", 2, "", "O"),
        ]
        candidates, summary = discover_automatic_candidate_bonds(
            atoms, [(0, 1)], interaction_scope="protein_ligand",
            exclude_same_residue=True,
        )
        self.assertEqual([row["bond_id"] for row in candidates], ["D0-H1-A2"])
        self.assertEqual(candidates[0]["donor_chemistry"]["confidence"], "template")
        self.assertEqual(candidates[0]["acceptor_chemistry"]["confidence"], "provisional")
        self.assertEqual(summary["donor_atom_count"], 1)

    def test_connectivity_discovers_attached_hydrogens_and_excludes_same_residue(self):
        atoms = [
            _atom(0, "N", "N", 1),
            _atom(1, "H", "H", 1),
            _atom(2, "O", "O", 1),
            _atom(3, "O", "O", 2),
        ]
        candidates = discover_candidate_bonds(
            atoms, [(0, 1)], [0], [2, 3],
            allowed_donor_elements=["N", "O"],
            allowed_acceptor_elements=["N", "O"],
            exclude_same_residue=True,
        )
        self.assertEqual([row["bond_id"] for row in candidates], ["D0-H1-A3"])
        self.assertEqual(candidates[0]["acceptor_identity"]["residue_number"], 2)

    def test_missing_connectivity_declared_hydrogen_fails_closed(self):
        atoms = [_atom(0, "N", "N", 1), _atom(1, "O", "O", 2)]
        with self.assertRaises(HydrogenBondDiscoveryError):
            discover_candidate_bonds(
                atoms, [], [0], [1],
                allowed_donor_elements=["N"],
                allowed_acceptor_elements=["O"],
                exclude_same_residue=False,
            )

    def test_project_emits_frozen_frame_by_bond_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = "".join([
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
                "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H\n",
                "ATOM      3  O   GLY A   2       2.800   0.000   0.000  1.00  0.00           O\n",
                "END\n",
            ])
            (root / "reference.pdb").write_text(topology, encoding="ascii")
            (root / "trajectory.xyz").write_text(
                "3\npresent\nN 0 0 0\nH 1 0 0\nO 2.8 0 0\n"
                "3\nabsent\nN 0 0 0\nH 1 0 0\nO 1 2 0\n",
                encoding="ascii",
            )
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 3,
                "index_base": 0, "bonds": [[0, 1]],
            }), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({
                "systems": [{"system_id": "test", "replicas": [{
                    "replica_id": "r1", "topology": "reference.pdb",
                    "connectivity": "bonds.json",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "trajectory.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                    }],
                }]}],
            }), encoding="utf-8")
            project = {
                "project_id": "hbond-discovery-test",
                "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json",
                "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom", "time_unit": "ps",
                "periodic_coordinate_policy": "reject",
                "reference_structure": "reference.pdb",
                "common_atom_policy": "strict",
                "selections": {
                    "alignment": {"preset": "all"},
                    "analysis": {"preset": "all"},
                },
                "definitions": {"hydrogen_bond_discovery": {
                    "chemistry_policy": "explicit_atoms_connectivity_v1",
                    "donor_atom_indices": [0], "acceptor_atom_indices": [2],
                    "allowed_donor_elements": ["N", "O"],
                    "allowed_acceptor_elements": ["N", "O"],
                    "exclude_same_residue": True, "water_policy": "exclude",
                    "frame_stride": 1,
                    "maximum_donor_acceptor_distance_angstrom": 3.5,
                    "minimum_donor_hydrogen_acceptor_angle_degrees": 150.0,
                    "maximum_reference_donor_hydrogen_bond_angstrom": 1.2,
                    "maximum_candidate_bonds": 10,
                    "maximum_feature_observations": 100,
                }},
                "requested_modules": ["hydrogen_bond_discovery"],
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = hydrogen_bond_discovery_project(project_path)
            project["definitions"]["hydrogen_bond_discovery"].update({
                "output_mode": "sparse_implicit_zero_v1", "candidate_chunk_size": 1,
            })
            project_path.write_text(json.dumps(project), encoding="utf-8")
            sparse_report = hydrogen_bond_discovery_project(project_path)
            project["definitions"]["hydrogen_bond_discovery"]["output_mode"] = (
                "sparse_packed_v2"
            )
            project_path.write_text(json.dumps(project), encoding="utf-8")
            packed_report = hydrogen_bond_discovery_project(project_path)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            [row["binary_values"] for row in report["frame_bond_matrix"]],
            [[1], [0]],
        )
        self.assertEqual(report["occupancies"][0]["occupancy_fraction"], 0.5)
        self.assertEqual(sparse_report["frame_matrix_representation"], "sparse_implicit_zero_v1")
        self.assertEqual(sparse_report["planned_feature_observation_count"], 2)
        self.assertNotIn("binary_values", sparse_report["frame_bond_matrix"][0])
        self.assertEqual(len(sparse_report["atom_dictionary"]), 3)
        self.assertEqual(
            [dense_primary_values(row, 1) for row in sparse_report["frame_bond_matrix"]],
            [[1], [0]],
        )
        self.assertEqual(sparse_report["occupancies"][0]["occupancy_fraction"], 0.5)
        self.assertEqual(packed_report["frame_matrix_representation"], "sparse_packed_v2")
        self.assertEqual(
            packed_report["observation_accounting"]["selected_physical_frame_count"],
            2,
        )
        self.assertEqual(
            packed_report["observation_accounting"]
            ["candidate_frame_feature_observation_count"],
            2,
        )
        self.assertEqual(
            [dense_primary_values(row, 1) for row in packed_report["frame_bond_matrix"]],
            [[1], [0]],
        )
        self.assertNotIn("present_geometry", packed_report["frame_bond_matrix"][0])
        self.assertNotIn("present_bond_ids", packed_report["frame_bond_matrix"][0])
        self.assertEqual(packed_report["occupancies"], sparse_report["occupancies"])
        self.assertIsNone(packed_report["cutoff_occupancies"])
        packed_cutoff_rows = unpack_sparse_cutoff_counts(
            packed_report["packed_cutoff_occupancy_segments"][0]
        )
        self.assertEqual(
            packed_cutoff_rows["present_frame_count"].tolist(), [1]
        )

    def test_automatic_multi_system_keeps_full_system_dictionaries_and_shared_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
                "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H\n",
                "ATOM      3  O6   DG B   2       2.800   0.000   0.000  1.00  0.00           O\n",
            ]
            (root / "variant.pdb").write_text("".join(common + [
                "END\n",
            ]).replace(
                common[2],
                "ATOM      3  C8   DG B   2       3.200   0.000   0.000  1.00  0.00           C\n"
                "ATOM      4  O6   DG B   2       2.800   0.000   0.000  1.00  0.00           O\n",
            ), encoding="ascii")
            # The shared O6 moves from atom index 2 to 3. Raw-index
            # intersection would falsely align control N7 with variant O6.
            (root / "control.pdb").write_text("".join(common + [
                "ATOM      4  N7   DG B   2       3.200   0.000   0.000  1.00  0.00           N\n",
                "END\n",
            ]), encoding="ascii")
            (root / "control.xyz").write_text(
                "4\nframe\nN 0 0 0\nH 1 0 0\nO 2.8 0 0\nN 3.2 0 0\n",
                encoding="ascii",
            )
            (root / "variant.xyz").write_text(
                "4\nframe\nN 0 0 0\nH 1 0 0\nC 3.2 0 0\nO 2.8 0 0\n",
                encoding="ascii",
            )
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 4,
                "index_base": 0, "bonds": [[0, 1]],
            }), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({
                "systems": [
                    {"system_id": "control", "replicas": [{
                        "replica_id": "r1", "topology": "control.pdb",
                        "connectivity": "bonds.json", "segments": [{
                            "segment_id": "s1", "trajectory": "control.xyz",
                            "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                        }],
                    }]},
                    {"system_id": "variant", "replicas": [{
                        "replica_id": "r1", "topology": "variant.pdb",
                        "connectivity": "bonds.json", "segments": [{
                            "segment_id": "s1", "trajectory": "variant.xyz",
                            "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                        }],
                    }]},
                ],
            }), encoding="utf-8")
            project = {
                "project_id": "automatic-harmonization-test",
                "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json",
                "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom", "time_unit": "ps",
                "periodic_coordinate_policy": "reject",
                "reference_structure": "control.pdb",
                "reference_system": "control",
                "common_atom_policy": "position",
                "selections": {
                    "alignment": {"preset": "all"},
                    "analysis": {"preset": "all"},
                },
                "definitions": {"hydrogen_bond_discovery": {
                    "chemistry_policy": "automatic_topology_templates_v1",
                    "interaction_scope": "protein_nucleic_acid",
                    "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
                    "exclude_same_residue": True, "water_policy": "exclude",
                    "frame_stride": 1,
                    "maximum_reference_donor_hydrogen_bond_angstrom": 1.2,
                    "maximum_candidate_bonds": 10,
                    "maximum_feature_observations": 100,
                    "output_mode": "sparse_spatial_observed_union_v3",
                    "candidate_chunk_size": 2,
                }},
                "requested_modules": ["hydrogen_bond_discovery"],
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = hydrogen_bond_discovery_project(project_path)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(
            report["candidate_harmonization"]["policy"],
            "intersection_by_endpoint_identity_lazy_v3",
        )
        self.assertIn(":O6:", report["candidate_dictionary"][0]["bond_id"])
        self.assertEqual(
            report["candidate_harmonization"]
            ["materialized_precoordinate_candidate_count"],
            0,
        )
        self.assertEqual(report["conceptual_candidate_count"], 1)
        self.assertEqual(
            report["conceptual_candidate_stratum_counts"],
            {"protein_to_nucleic_acid": 1},
        )
        self.assertEqual(
            report["candidate_harmonization"][
                "common_candidate_stratum_counts"
            ],
            {"protein_to_nucleic_acid": 1},
        )
        self.assertEqual(report["materialized_observed_candidate_count"], 1)
        self.assertEqual(report["evaluated_frame_count"], 2)
        self.assertEqual(report["conceptual_candidate_frame_count"], 2)
        self.assertEqual(
            report["geometry_contract"]["coordinate_reconstruction"],
            "none_selected_frames_evaluated_raw_wrapped",
        )
        self.assertEqual(
            report["geometry_contract"]["project_periodic_coordinate_policy"],
            "reject",
        )
        self.assertEqual(
            report["geometry_contract"]["hydrogen_bond_coordinate_path"],
            "raw_wrapped_frame_with_exact_minimum_image_vectors_v1",
        )
        self.assertEqual(
            [row["candidate_count"] for row in report["frame_bond_matrix"]], [1, 1]
        )
        self.assertEqual(report["error_count"], 0)

    def test_automatic_project_has_default_cutoffs_and_sensitivity_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = "".join([
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
                "ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H\n",
                "ATOM      3  O   GLY A   2       2.800   0.000   0.000  1.00  0.00           O\n",
                "END\n",
            ])
            (root / "reference.pdb").write_text(topology, encoding="ascii")
            (root / "trajectory.xyz").write_text(
                "3\nprimary\nN 0 0 0\nH 1 0 0\nO 2.8 0 0\n"
                "3\nsensitivity-only\nN 0 0 0\nH 1 0 0\nO 3.3 0 0\n",
                encoding="ascii",
            )
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 3,
                "index_base": 0, "bonds": [[0, 1]],
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
                "project_id": "automatic-hbond-test", "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json", "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
                "time_unit": "ps", "periodic_coordinate_policy": "reject",
                "reference_structure": "reference.pdb", "common_atom_policy": "strict",
                "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
                "definitions": {"hydrogen_bond_discovery": {
                    "chemistry_policy": "automatic_topology_templates_v1",
                    "interaction_scope": "all_solute", "exclude_same_residue": True,
                    "water_policy": "exclude", "frame_stride": 1,
                    "cutoff_policy": {"preset": "mdanalysis_compatible_v1"},
                    "maximum_reference_donor_hydrogen_bond_angstrom": 1.2,
                    "maximum_candidate_bonds": 10, "maximum_feature_observations": 100,
                }},
                "requested_modules": ["hydrogen_bond_discovery"],
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = hydrogen_bond_discovery_project(project_path)
            project["definitions"]["hydrogen_bond_discovery"].update({
                "output_mode": "sparse_implicit_zero_v1", "candidate_chunk_size": 1,
            })
            project_path.write_text(json.dumps(project), encoding="utf-8")
            sparse_report = hydrogen_bond_discovery_project(project_path)
        self.assertEqual(report["settings"]["mode"], "automatic")
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual([row["binary_values"] for row in report["frame_bond_matrix"]], [[1], [0]])
        primary = [row for row in report["cutoff_occupancies"] if row["cutoff_id"] == "primary"]
        expanded = [row for row in report["cutoff_occupancies"] if row["cutoff_id"] == "sensitivity_da3.5_angle150"]
        self.assertEqual(primary[0]["occupancy_fraction"], 0.5)
        self.assertEqual(expanded[0]["occupancy_fraction"], 1.0)
        dense_counts = {
            row["cutoff_id"]: row["present_frame_count"]
            for row in report["cutoff_occupancies"]
        }
        sparse_counts = {
            row["cutoff_id"]: row["present_frame_count"]
            for row in sparse_report["cutoff_occupancies"]
        }
        self.assertEqual(sparse_counts, dense_counts)
        self.assertEqual(
            sparse_report["candidate_dictionary"],
            [
                {key: candidate[key] for key in (
                    "bond_id", "donor_atom_index", "hydrogen_atom_index",
                    "acceptor_atom_index", "interaction_stratum",
                )}
                for candidate in report["candidate_dictionary"]
            ],
        )


if __name__ == "__main__":
    unittest.main()
