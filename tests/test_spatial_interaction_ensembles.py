import unittest

from salsbury_md_analysis.spatial_interaction_ensembles import (
    build_spatial_interaction_ensembles,
    compile_superfeatures,
)


class SpatialInteractionEnsembleTests(unittest.TestCase):
    def settings(self, **updates):
        values = {
            "minimum_point_observations": 8,
            "minimum_distinct_frames": 8,
            "time_block_count": 4,
            "mode_k_values": (2, 3),
            "minimum_mode_observations": 4,
            "minimum_mode_fraction": 0.20,
            "minimum_mode_silhouette": 0.50,
            "minimum_mode_centroid_separation_angstrom": 1.0,
            "minimum_mode_time_blocks": 3,
            "minimum_mode_replicas": 1,
            "maximum_point_observations": 10_000,
            "maximum_exact_mode_points": 1_000,
            "maximum_mode_iterations": 100,
            "mode_center_tolerance_angstrom": 1.0e-8,
        }
        values.update(updates)
        return values

    def dictionary(self):
        return [{
            "feature_id": "hbond-1",
            "source_module": "hydrogen_bond_discovery",
            "interaction_type": "direct_hydrogen_bond",
            "definition": {
                "donor_atom_index": 1,
                "hydrogen_atom_index": 2,
                "acceptor_atom_index": 7,
            },
        }, {
            "feature_id": "ion-1",
            "source_module": "ion_coordination_geometry",
            "interaction_type": "ion_ligand_coordination",
            "definition": {
                "site_id": "zinc", "ion_atom_index": 20,
                "ligand_atom_index": 9,
            },
        }, {
            "feature_id": "water-bridge-1",
            "source_module": "water_mediated_hydrogen_bond_networks",
            "interaction_type": "one_water_hydrogen_bond_bridge",
            "definition": {"bridge_id": "bridge-1"},
        }]

    def observations(self, superfeature_id, system_id="A", shift=0.0):
        rows = []
        for frame in range(8):
            center = 0.0 if frame % 2 == 0 else 5.0
            rows.append({
                "system_id": system_id, "replica_id": "replica-1",
                "segment_id": "segment-1", "source_frame_index": frame,
                "source_feature_id": "hbond-1",
                "superfeature_id": superfeature_id,
                "point_atom_index": 7,
                "coordinate_angstrom": [center + shift, 0.05 * frame, 0.0],
            })
        return rows

    def test_compiles_only_exact_partner_atom_superfeatures(self):
        rows, mapping, unsupported = compile_superfeatures(self.dictionary(), 10)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(mapping["hbond-1"]), 2)
        self.assertEqual(len(mapping["ion-1"]), 1)
        self.assertNotIn("water-bridge-1", mapping)
        self.assertEqual(unsupported[0]["feature_id"], "water-bridge-1")
        self.assertEqual(
            unsupported[0]["reason"],
            "no_exact_dynamic_partner_atom_in_fingerprint_definition",
        )

    def test_recurrent_separated_clouds_pass_mode_gates(self):
        dictionary, mapping, unsupported = compile_superfeatures(
            self.dictionary(), 10
        )
        superfeature_id = mapping["hbond-1"][0]["superfeature_id"]
        result = build_spatial_interaction_ensembles(
            self.observations(superfeature_id), dictionary, unsupported,
            self.settings(),
        )
        summary = result["spatial_ensemble_summaries"][0]
        self.assertEqual(summary["spatial_summary_gate"], "passed")
        self.assertEqual(
            summary["mode_inference_status"], "gated_multimodal_candidate"
        )
        selected = result["selected_spatial_mode_candidates"][0]
        self.assertEqual(selected["k"], 2)
        self.assertGreater(selected["silhouette"], 0.9)
        self.assertTrue(all(selected["gates"].values()))
        self.assertTrue(
            all(mode["time_block_count"] == 4 for mode in selected["modes"])
        )

    def test_exact_mode_cap_withholds_clustering_but_keeps_shape(self):
        dictionary, mapping, unsupported = compile_superfeatures(
            self.dictionary(), 10
        )
        superfeature_id = mapping["hbond-1"][0]["superfeature_id"]
        result = build_spatial_interaction_ensembles(
            self.observations(superfeature_id), dictionary, unsupported,
            self.settings(maximum_exact_mode_points=7),
        )
        summary = result["spatial_ensemble_summaries"][0]
        self.assertEqual(summary["spatial_summary_gate"], "passed")
        self.assertEqual(
            summary["mode_inference_status"],
            "withheld_by_exact_mode_resource_gate",
        )
        self.assertIn("centroid_angstrom", summary)
        self.assertEqual(result["mode_candidates"], [])

    def test_same_superfeature_is_compared_across_systems(self):
        dictionary, mapping, unsupported = compile_superfeatures(
            self.dictionary(), 10
        )
        superfeature_id = mapping["hbond-1"][0]["superfeature_id"]
        observations = self.observations(superfeature_id, "A")
        observations.extend(self.observations(superfeature_id, "B", shift=2.0))
        result = build_spatial_interaction_ensembles(
            observations, dictionary, unsupported, self.settings()
        )
        comparison = result["pairwise_system_spatial_differences"][0]
        self.assertEqual((comparison["system_i"], comparison["system_j"]), ("A", "B"))
        self.assertAlmostEqual(comparison["centroid_displacement_angstrom"], 2.0)
        self.assertEqual(comparison["evidence_level"], "descriptive")


if __name__ == "__main__":
    unittest.main()
