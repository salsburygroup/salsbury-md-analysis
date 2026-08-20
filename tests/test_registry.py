import unittest

from salsbury_md_analysis.registry import MODULES, get_module


class RegistryTests(unittest.TestCase):
    def test_module_ids_are_unique(self):
        self.assertEqual(len(MODULES), len({module.module_id for module in MODULES}))

    def test_public_registry_is_md_only(self):
        self.assertEqual(len(MODULES), 45)
        self.assertTrue({module.category for module in MODULES} <= {"core", "md", "reporting"})
        self.assertFalse(any("dock" in module.module_id for module in MODULES))

    def test_scaffold_statuses_are_honest(self):
        self.assertTrue(MODULES)
        by_status = {status: {module.module_id for module in MODULES if module.status == status}
                     for status in {module.status for module in MODULES}}
        self.assertEqual(
            by_status.get("experimental"),
            {
                "provenance_manifest",
                "preflight_inventory",
                "common_atom_mapping",
                "structural_integrity_qc",
                "replica_rmsd_rg",
                "pooled_rmsf",
                "dccm",
                "generalized_correlation_and_information",
                "information_dynamics",
                "correlation_networks",
                "individual_pca",
                "common_pca",
                "time_lagged_independent_component_analysis",
                "pca_fes_basins",
                "clustering_kmeans",
                "clustering_hdbscan",
                "clustering_imwkmeans",
                "alternative_clustering",
                "pald_community_analysis",
                "representative_frames",
                "state_coordinate_exports",
                "representative_structures",
                "markov_state_models",
                "dihedral_distributions",
                "hydrogen_bonds",
                "hydrogen_bond_discovery",
                "hydrogen_bond_comparison",
                "water_mediated_hydrogen_bond_networks",
                "hydrogen_bond_patterns",
                "trajectory_features",
                "scalar_feature_distributions",
                "scalar_threshold_states",
                "optional_observables",
                "convergence_uncertainty",
                "rmsf_permutation_inference",
                "grouped_ml",
                "grouped_regularized_classification",
                "integrated_comparison",
                "secondary_structure",
                "nucleic_acid_structure",
                "nucleic_acid_geometry",
                "ion_coordination_geometry",
                "ion_atmosphere",
                "solvent_accessible_surface_area",
                "radial_distribution_functions",
            },
        )
        self.assertNotIn("supported", by_status)
        self.assertEqual(by_status.get("planned", set()), set())

    def test_full_scope_is_present(self):
        required = {
            "preflight_inventory", "structural_integrity_qc", "replica_rmsd_rg",
            "pooled_rmsf", "dccm", "generalized_correlation_and_information",
            "information_dynamics", "correlation_networks", "trajectory_features",
            "scalar_feature_distributions", "scalar_threshold_states",
            "individual_pca", "common_pca",
            "time_lagged_independent_component_analysis",
            "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
            "clustering_imwkmeans", "representative_frames",
            "state_coordinate_exports",
            "alternative_clustering", "representative_structures",
            "pald_community_analysis",
            "markov_state_models", "dihedral_distributions",
            "hydrogen_bonds", "hydrogen_bond_discovery", "hydrogen_bond_comparison",
            "water_mediated_hydrogen_bond_networks", "hydrogen_bond_patterns",
            "grouped_ml", "grouped_regularized_classification", "secondary_structure",
            "nucleic_acid_structure",
            "nucleic_acid_geometry", "ion_coordination_geometry",
            "ion_atmosphere",
            "solvent_accessible_surface_area", "optional_observables",
            "radial_distribution_functions",
            "convergence_uncertainty", "rmsf_permutation_inference", "integrated_comparison",
        }
        self.assertTrue(required.issubset({module.module_id for module in MODULES}))

    def test_unknown_module_fails_closed(self):
        with self.assertRaises(KeyError):
            get_module("not-a-real-analysis")


if __name__ == "__main__":
    unittest.main()
