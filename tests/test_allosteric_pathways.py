import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.allosteric_pathways import (
    allosteric_pathway_network,
    allosteric_pathways_project,
    allosteric_pathways_project_safe,
)


class AllostericPathwayTests(unittest.TestCase):
    def test_chain_path_and_weighted_betweenness_are_exact(self):
        occupancy = [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.9, 0.0],
            [0.0, 0.9, 0.0, 0.9],
            [0.0, 0.0, 0.9, 0.0],
        ]
        report = allosteric_pathway_network(occupancy, [0], [3])
        path = report["source_sink_paths"][0]
        self.assertEqual(path["canonical_shortest_path"], [0, 1, 2, 3])
        self.assertEqual(path["equal_shortest_path_count"], 1)
        np.testing.assert_allclose(
            report["internal_node_shortest_path_participation"],
            [0.0, 1.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(report["weighted_betweenness_centrality"][1], 2 / 3)
        self.assertAlmostEqual(report["weighted_betweenness_centrality"][2], 2 / 3)

    def test_equal_shortest_paths_are_fractionally_counted(self):
        occupancy = [
            [0.0, 0.8, 0.8, 0.0],
            [0.8, 0.0, 0.0, 0.8],
            [0.8, 0.0, 0.0, 0.8],
            [0.0, 0.8, 0.8, 0.0],
        ]
        report = allosteric_pathway_network(occupancy, [0], [3])
        path = report["source_sink_paths"][0]
        self.assertEqual(path["equal_shortest_path_count"], 2)
        self.assertEqual(path["canonical_shortest_path"], [0, 1, 3])
        np.testing.assert_allclose(
            report["internal_node_shortest_path_participation"],
            [0.0, 0.5, 0.5, 0.0],
        )
        self.assertTrue(all(
            abs(row["participation"] - 0.5) < 1.0e-12
            for row in report["edge_shortest_path_participation"]
        ))

    def test_negative_log_distance_selects_highest_product_route(self):
        occupancy = [
            [0.0, 0.9, 0.5],
            [0.9, 0.0, 0.9],
            [0.5, 0.9, 0.0],
        ]
        report = allosteric_pathway_network(
            occupancy, [0], [2], minimum_contact_occupancy=0.4
        )
        self.assertEqual(
            report["source_sink_paths"][0]["canonical_shortest_path"],
            [0, 1, 2],
        )

    def test_optional_neighbor_factor_and_combined_score_are_explicit(self):
        occupancy = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
        dependency = [
            [1.0, 0.2, 0.9],
            [0.2, 1.0, 0.8],
            [0.9, 0.8, 1.0],
        ]
        report = allosteric_pathway_network(
            occupancy, [0], [2], dependency_matrix=dependency
        )
        self.assertAlmostEqual(report["neighbor_correlation_factor"][0], 0.2)
        self.assertAlmostEqual(
            report["neighbor_correlation_factor"][1],
            (1.0 * 0.2 + 0.5 * 0.8) / 1.5,
        )
        self.assertEqual(len(report["combined_allosteric_score"]), 3)
        self.assertIn("min-max", report["combined_score_contract"])

    def test_project_preserves_nodes_and_flags_disconnected_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "network.json").write_text(json.dumps({
                "network_schema": "salsbury-residue-contact-network-v1",
                "nodes": [
                    {"node_id": "A:1"}, {"node_id": "A:2"}, {"node_id": "A:3"}
                ],
                "contact_occupancy_matrix": [
                    [0.0, 0.9, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]
                ],
            }), encoding="utf-8")
            path = root / "project.json"
            path.write_text(json.dumps({
                "definitions": {
                    "allosteric_pathways": {
                        "network_source": "external_json",
                        "network_path": "network.json",
                        "node_selection": "analysis",
                        "alignment_selection": "alignment",
                        "minimum_reference_coverage": 1.0,
                        "frame_stride": 1,
                        "frame_selection": {"mode": "fixed_stride_v1"},
                        "contact_cutoff_angstrom": 8.0,
                        "minimum_sequence_separation": 2,
                        "minimum_evaluated_frames_per_system": 2,
                        "minimum_variance_angstrom2": 1.0e-12,
                        "source_node_indices": [0],
                        "sink_node_indices": [2],
                        "minimum_contact_occupancy": 0.5,
                        "distance_epsilon": 1.0e-12,
                        "shortest_path_equality_tolerance": 1.0e-12,
                        "maximum_nodes": 20,
                        "neighbor_correlation_factor_enabled": False,
                    }
                }
            }), encoding="utf-8")
            report = allosteric_pathways_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["pathway_validity_status"], "failed")
        self.assertEqual(report["nodes"][2]["node_id"], "A:3")
        self.assertEqual(report["issues"][0]["code"], "SOURCE_SINK_PATHS_DISCONNECTED")
        self.assertEqual(
            report["observation_accounting"],
            {
                "selected_physical_frame_count": 0,
                "symmetry_expanded_observation_count": 0,
                "accounting_basis": "external aggregate network; no trajectory frames are read by this module",
            },
        )

    def test_ncf_requires_a_dependency_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "network.json").write_text(json.dumps({
                "network_schema": "salsbury-residue-contact-network-v1",
                "nodes": [{"node_id": "A:1"}, {"node_id": "A:2"}],
                "contact_occupancy_matrix": [[0.0, 1.0], [1.0, 0.0]],
            }), encoding="utf-8")
            path = root / "project.json"
            path.write_text(json.dumps({
                "definitions": {
                    "allosteric_pathways": {
                        "network_source": "external_json",
                        "network_path": "network.json",
                        "node_selection": "analysis",
                        "alignment_selection": "alignment",
                        "minimum_reference_coverage": 1.0,
                        "frame_stride": 1,
                        "frame_selection": {"mode": "fixed_stride_v1"},
                        "contact_cutoff_angstrom": 8.0,
                        "minimum_sequence_separation": 2,
                        "minimum_evaluated_frames_per_system": 2,
                        "minimum_variance_angstrom2": 1.0e-12,
                        "source_node_indices": [0],
                        "sink_node_indices": [1],
                        "minimum_contact_occupancy": 0.5,
                        "distance_epsilon": 1.0e-12,
                        "shortest_path_equality_tolerance": 1.0e-12,
                        "maximum_nodes": 20,
                        "neighbor_correlation_factor_enabled": True,
                    }
                }
            }), encoding="utf-8")
            report = allosteric_pathways_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("requires dependency_matrix", report["issues"][0]["message"])

    def test_project_derives_residue_contacts_and_dependencies_from_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb_rows = []
            reference = [
                ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                ((4.0, 0.0, 0.0), (4.0, 1.0, 0.0)),
                ((4.0, 4.0, 0.0), (4.0, 5.0, 0.0)),
                ((0.0, 4.0, 1.0), (0.0, 5.0, 1.0)),
            ]
            serial = 1
            for residue, (n_position, ca_position) in enumerate(reference, start=1):
                for name, position, element in (
                    ("N", n_position, "N"), ("CA", ca_position, "C")
                ):
                    x, y, z = position
                    pdb_rows.append(
                        f"ATOM  {serial:5d} {name:^4s} ALA A{residue:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                    )
                    serial += 1
            (root / "reference.pdb").write_text(
                "".join(pdb_rows) + "END\n", encoding="utf-8"
            )
            ca_frames = [
                [(0.0, 1.0, 0.0), (4.0, 1.0, 0.0), (4.0, 5.0, 0.0), (0.0, 5.0, 1.0)],
                [(0.2, 1.1, 0.1), (4.1, 1.2, 0.0), (4.2, 5.1, -0.1), (0.1, 5.2, 1.1)],
                [(-0.1, 0.9, -0.1), (3.8, 0.8, 0.1), (3.9, 4.8, 0.2), (-0.2, 4.9, 0.8)],
            ]
            frames = []
            for frame_index, ca_positions in enumerate(ca_frames):
                coordinates = []
                for residue_index, (n_position, _) in enumerate(reference):
                    coordinates.extend((n_position, ca_positions[residue_index]))
                frames.extend([str(len(coordinates)), f"frame {frame_index}"])
                frames.extend(
                    f"C {x} {y} {z}" for x, y, z in coordinates
                )
            (root / "trajectory.xyz").write_text(
                "\n".join(frames) + "\n", encoding="utf-8"
            )
            (root / "system.json").write_text(json.dumps({
                "systems": [{
                    "system_id": "test",
                    "replicas": [{
                        "replica_id": "r1",
                        "topology": "reference.pdb",
                        "segments": [{
                            "segment_id": "s1",
                            "trajectory": "trajectory.xyz",
                            "timing": {
                                "first_frame_time": 0,
                                "frame_interval": 1,
                                "unit": "ps",
                            },
                        }],
                    }],
                }],
            }), encoding="utf-8")
            project = {
                "project_id": "trajectory-pathway-test",
                "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json",
                "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "reject",
                "reference_structure": "reference.pdb",
                "reference_system": "test",
                "common_atom_policy": "strict",
                "selections": {
                    "alignment": {"atom_names": ["N"]},
                    "analysis": {"atom_names": ["CA"]},
                },
                "definitions": {
                    "allosteric_pathways": {
                        "network_source": "trajectory",
                        "network_path": "",
                        "node_selection": "analysis",
                        "alignment_selection": "alignment",
                        "minimum_reference_coverage": 1.0,
                        "frame_stride": 1,
                        "frame_selection": {"mode": "fixed_stride_v1"},
                        "contact_cutoff_angstrom": 6.0,
                        "minimum_sequence_separation": 1,
                        "minimum_evaluated_frames_per_system": 3,
                        "minimum_variance_angstrom2": 1.0e-12,
                        "source_node_indices": [0],
                        "sink_node_indices": [3],
                        "minimum_contact_occupancy": 0.5,
                        "distance_epsilon": 1.0e-12,
                        "shortest_path_equality_tolerance": 1.0e-12,
                        "maximum_nodes": 20,
                        "neighbor_correlation_factor_enabled": True,
                    },
                },
                "requested_modules": ["allosteric_pathways"],
                "protected_locations": ["/protected/example"],
            }
            path = root / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            report = allosteric_pathways_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["network_derivation"]["network_source"], "trajectory")
        self.assertEqual(report["observation_accounting"]["selected_physical_frame_count"], 3)
        self.assertEqual(len(report["nodes"]), 4)
        self.assertEqual(report["nodes"][0]["node_id"], "A:1:ALA")
        self.assertEqual(report["systems"][0]["network"]["node_count"], 4)
        self.assertIn(
            "neighbor_correlation_factor", report["systems"][0]["network"]
        )


if __name__ == "__main__":
    unittest.main()
