import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.analysis_config import (
    AnalysisConfigError,
    apply_module_configuration,
    default_analysis_config,
    load_analysis_config,
    make_memory_fit_config,
    make_resource_fit_config,
)


class AnalysisConfigTests(unittest.TestCase):
    def test_default_enables_every_module_view_and_final_report(self):
        config = default_analysis_config(
            ["common_pca", "pca_fes_basins", "solvent_accessible_surface_area"],
            ["complex_heavy", "protein_trace"],
        )
        self.assertEqual(config["default_module_policy"], "all_applicable")
        self.assertTrue(all(row["enabled"] for row in config["modules"].values()))
        self.assertTrue(all(row["enabled"] for row in config["views"].values()))
        self.assertTrue(config["reporting"]["resource_table_enabled"])
        self.assertTrue(config["reporting"]["finding_picker_enabled"])
        self.assertEqual(config["reporting"]["minimum_headline_findings"], 10)
        self.assertEqual(config["reporting"]["headline_findings"], 12)
        self.assertEqual(config["reporting"]["maximum_findings"], 50)
        self.assertEqual(config["comparisons"]["mode"], "all_pairs")
        self.assertTrue(config["inference"]["automatic_chemical_context"])
        self.assertTrue(config["inference"]["ion_site_classification_enabled"])
        self.assertEqual(config["execution"]["coordinate_cache"], "auto")
        self.assertEqual(config["execution"]["submission_adapter"], "local")
        self.assertIsNone(config["execution"]["slurm_profile"])
        self.assertEqual(config["execution"]["finalization_headroom_fraction"], 0.05)
        self.assertEqual(config["execution"]["time_safety_factor"], 1.5)
        self.assertNotIn("memory_safety_factor", config["execution"])
        self.assertEqual(
            config["execution"]["well_calibrated_memory_uncertainty_factor"],
            1.0,
        )
        self.assertEqual(
            config["execution"]["poorly_calibrated_memory_uncertainty_factor"],
            1.25,
        )
        self.assertEqual(
            config["execution"]["censored_timeout_safety_factor"], 1.5
        )
        self.assertTrue(config["execution"]["autorecovery"])
        self.assertEqual(config["execution"]["maximum_task_attempts"], 2)

        self.assertEqual(len(config["clustering"]["methods"]), 11)
        self.assertFalse(config["clustering"]["methods"]["hdbscan"]["enabled"])
        self.assertTrue(all(
            row["enabled"]
            for method, row in config["clustering"]["methods"].items()
            if method != "hdbscan"
        ))
        self.assertEqual(config["clustering"]["feature_space"], "tica")
        self.assertFalse(config["community_analysis"]["pald"]["enabled"])
        self.assertFalse(
            config["community_analysis"]["pald"]["community_msm_enabled"]
        )
        self.assertIn("03_conformational_bases", config["module_groups"])
        self.assertEqual(
            config["modules"]["pca_fes_basins"]["depends_on"],
            ["common_pca"],
        )
        self.assertIn(
            "pca_fes_basins",
            config["modules"]["common_pca"]["turning_off_also_disables"],
        )

    def test_autorecovery_can_be_disabled_and_attempts_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis-config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {
                    "autorecovery": False,
                    "maximum_task_attempts": 1,
                },
            }), encoding="utf-8")
            config = load_analysis_config(path, ["common_pca"], ["global"])
        self.assertFalse(config["execution"]["autorecovery"])
        self.assertEqual(config["execution"]["maximum_task_attempts"], 1)

    def test_autorecovery_rejects_unbounded_attempt_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis-config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {"maximum_task_attempts": 6},
            }), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisConfigError, "1 through 5"):
                load_analysis_config(path, ["common_pca"], ["global"])

    def test_legacy_analysis_memory_factor_maps_to_named_uncertainty(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis-config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {"memory_safety_factor": 1.25},
            }), encoding="utf-8")
            config = load_analysis_config(
                path, ["common_pca"], ["global"]
            )
        self.assertNotIn("memory_safety_factor", config["execution"])
        self.assertEqual(
            config["execution"]["poorly_calibrated_memory_uncertainty_factor"],
            1.25,
        )

    def test_generated_complete_config_can_be_reloaded_unchanged(self):
        module_ids = [
            "provenance_manifest", "preflight_inventory", "common_atom_mapping",
            "common_pca", "pca_fes_basins", "representative_frames",
        ]
        expected = default_analysis_config(module_ids, ["global_common_heavy"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis-config.json"
            path.write_text(json.dumps(expected), encoding="utf-8")
            observed = load_analysis_config(
                path, module_ids, ["global_common_heavy"]
            )
        self.assertEqual(observed, expected)

    def test_protected_core_only_mode_materializes_explicit_module_selection(self):
        module_ids = [
            "provenance_manifest", "preflight_inventory", "common_atom_mapping",
            "structural_integrity_qc", "replica_rmsd_rg", "pooled_rmsf",
            "individual_pca", "common_pca", "dccm", "pca_fes_basins",
            "representative_frames", "solvent_accessible_surface_area",
            "clustering_kmeans", "state_coordinate_exports",
        ]
        config = load_analysis_config(
            None,
            module_ids,
            ["global_common_heavy"],
            module_selection_override="protected_core_only",
        )
        self.assertEqual(
            config["planning"]["module_selection"], "protected_core_only"
        )
        enabled = {
            module_id for module_id, row in config["modules"].items()
            if row["enabled"]
        }
        self.assertEqual(enabled, set(module_ids[:11] + ["state_coordinate_exports"]))
        self.assertFalse(config["clustering"]["methods"]["kmeans"]["enabled"])
        self.assertFalse(
            config["views"]["global_common_heavy"][
                "state_trajectory_exports_enabled"
            ]
        )

    def test_planning_modes_can_be_selected_in_config_or_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "planning": {
                    "module_selection": "all_enabled",
                    "stride_mode": "uniform_cache_stride",
                },
            }), encoding="utf-8")
            configured = load_analysis_config(
                path, ["provenance_manifest"], ["global"]
            )
            overridden = load_analysis_config(
                path,
                ["provenance_manifest"],
                ["global"],
                module_selection_override="protected_core_only",
                stride_mode_override="balanced_per_method",
            )
        self.assertEqual(
            configured["planning"]["stride_mode"], "uniform_cache_stride"
        )
        self.assertEqual(
            overridden["planning"], {
                "module_selection": "protected_core_only",
                "stride_mode": "balanced_per_method",
            }
        )

    def test_reporting_contract_requires_ten_to_twelve_of_fifty(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis-config.json"
            for headline_findings in (10, 11, 12):
                path.write_text(json.dumps({
                    "config_schema": "salsbury-analysis-config-v1",
                    "reporting": {
                        "minimum_headline_findings": 10,
                        "headline_findings": headline_findings,
                        "maximum_findings": 50,
                    },
                }), encoding="utf-8")
                config = load_analysis_config(
                    path, ["provenance_manifest"], ["global"]
                )
                self.assertEqual(
                    config["reporting"]["headline_findings"],
                    headline_findings,
                )

            for headline_findings in (9, 13):
                path.write_text(json.dumps({
                    "config_schema": "salsbury-analysis-config-v1",
                    "reporting": {"headline_findings": headline_findings},
                }), encoding="utf-8")
                with self.assertRaisesRegex(
                    AnalysisConfigError, "integer from 10 through 12"
                ):
                    load_analysis_config(
                        path, ["provenance_manifest"], ["global"]
                    )

            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "reporting": {"maximum_findings": 49},
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisConfigError, "maximum_findings must be 50"
            ):
                load_analysis_config(
                    path, ["provenance_manifest"], ["global"]
                )

            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "reporting": {
                    "minimum_headline_findings": 12,
                    "headline_findings": 10,
                },
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisConfigError, "cannot exceed"
            ):
                load_analysis_config(
                    path, ["provenance_manifest"], ["global"]
                )

    def test_minimums_and_reusable_cache_paths_resolve_from_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minimums = root / "minimums.json"
            minimums.write_text("{}", encoding="utf-8")
            cache = root / "lossless-cache"
            cache.mkdir()
            path = root / "analysis-config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "sampling": {"scientific_minimums_file": "minimums.json"},
                "execution": {"coordinate_cache_input": "lossless-cache"},
            }), encoding="utf-8")
            config = load_analysis_config(
                path, ["provenance_manifest"], ["global"]
            )
        self.assertEqual(
            config["sampling"]["scientific_minimums_file"],
            str(minimums.resolve()),
        )
        self.assertEqual(
            config["execution"]["coordinate_cache_input"],
            str(cache.resolve()),
        )

    def test_ion_site_classification_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "inference": {"ion_site_classification_enabled": False},
            }), encoding="utf-8")
            config = load_analysis_config(
                path, ["provenance_manifest"], ["global_common_heavy"]
            )
        self.assertFalse(config["inference"]["ion_site_classification_enabled"])
        self.assertTrue(config["inference"]["automatic_chemical_context"])

    def test_slurm_profile_and_calibration_paths_resolve_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "cluster.json"
            calibration = root / "calibration.json"
            profile.write_text("{}", encoding="utf-8")
            calibration.write_text("{}", encoding="utf-8")
            config_path = root / "analysis.json"
            config_path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "execution": {
                    "submission_adapter": "slurm",
                    "slurm_profile": "cluster.json",
                    "resource_calibration_catalog": "calibration.json",
                },
            }), encoding="utf-8")
            config = load_analysis_config(
                config_path, ["provenance_manifest"], ["global"]
            )
        self.assertEqual(
            config["execution"]["slurm_profile"], str(profile.resolve())
        )
        self.assertEqual(
            config["execution"]["resource_calibration_catalog"],
            str(calibration.resolve()),
        )

    def test_disabled_upstream_disables_dependents_and_options_apply(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {
                    "clustering_kmeans": {"enabled": False},
                    "solvent_accessible_surface_area": {
                        "options": {"frame_stride": 7}
                    },
                },
            }), encoding="utf-8")
            module_ids = [
                "provenance_manifest", "preflight_inventory", "common_atom_mapping",
                "clustering_kmeans", "grouped_ml",
                "solvent_accessible_surface_area",
            ]
            config = load_analysis_config(path, module_ids, ["global"])
            definitions, commands, requested, reasons = apply_module_configuration(
                {
                    "clustering_kmeans": {}, "grouped_ml": {},
                    "solvent_accessible_surface_area": {"frame_stride": 1},
                },
                ["cluster-kmeans", "grouped-ml", "sasa"],
                [
                    "clustering_kmeans", "grouped_ml",
                    "solvent_accessible_surface_area",
                ],
                config,
            )
            self.assertEqual(commands, ["sasa"])
            self.assertEqual(requested, ["solvent_accessible_surface_area"])
            self.assertEqual(
                definitions["solvent_accessible_surface_area"]["frame_stride"], 7
            )
            self.assertIn("grouped_ml", reasons)

    def test_scientific_core_is_protected(self):
        module_ids = [
            "provenance_manifest", "preflight_inventory", "common_atom_mapping",
            "structural_integrity_qc", "common_pca", "pca_fes_basins",
        ]
        generated = default_analysis_config(module_ids, ["global_common_heavy"])
        self.assertTrue(
            generated["modules"]["structural_integrity_qc"]["protected"]
        )
        self.assertTrue(generated["modules"]["common_pca"]["protected"])
        self.assertTrue(generated["modules"]["pca_fes_basins"]["protected"])
        self.assertIn(
            "structural_integrity_qc",
            generated["module_groups"]["01_infrastructure"]["modules"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {"common_pca": {"enabled": False}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisConfigError,
                "protected module common_pca cannot be disabled",
            ):
                load_analysis_config(
                    path, module_ids, ["global_common_heavy"]
                )
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {"structural_integrity_qc": {"enabled": False}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisConfigError,
                "protected module structural_integrity_qc cannot be disabled",
            ):
                load_analysis_config(path, module_ids, ["global_common_heavy"])

    def test_comparison_context_protects_integrated_finalizer(self):
        module_ids = ["provenance_manifest", "integrated_comparison"]
        generated = default_analysis_config(
            module_ids, [],
            protected_modules=["provenance_manifest", "integrated_comparison"],
        )
        self.assertTrue(generated["modules"]["integrated_comparison"]["protected"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "modules": {"integrated_comparison": {"enabled": False}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisConfigError,
                "protected module integrated_comparison cannot be disabled",
            ):
                load_analysis_config(
                    path, module_ids, [],
                    additional_protected_modules=("integrated_comparison",),
                )

    def test_memory_fit_refuses_to_disable_protected_structural_qc(self):
        config = default_analysis_config(
            ["structural_integrity_qc", "replica_rmsd_rg"], []
        )
        with self.assertRaisesRegex(
            AnalysisConfigError, "no acceptable reduced configuration"
        ):
            make_memory_fit_config(config, ["structural_integrity_qc"])

    def test_memory_fit_config_materializes_dependency_disables(self):
        config = default_analysis_config(
            [
                "clustering_kmeans", "grouped_ml",
                "solvent_accessible_surface_area",
            ],
            ["global_common_heavy"],
        )
        reduced, direct, transitive = make_memory_fit_config(
            config, ["clustering_kmeans", "coordinate_cache"]
        )
        self.assertEqual(direct, ["clustering_kmeans", "coordinate_cache"])
        self.assertEqual(transitive, ["grouped_ml"])
        self.assertEqual(reduced["execution"]["coordinate_cache"], "off")
        self.assertFalse(reduced["modules"]["clustering_kmeans"]["enabled"])
        self.assertFalse(reduced["modules"]["grouped_ml"]["enabled"])
        self.assertTrue(
            reduced["modules"]["solvent_accessible_surface_area"]["enabled"]
        )

    def test_resource_fit_config_materializes_planner_switch_closure(self):
        config = default_analysis_config(
            [
                "structural_integrity_qc", "common_pca", "pca_fes_basins",
                "representative_frames", "clustering_kmeans", "grouped_ml",
                "solvent_accessible_surface_area",
            ],
            ["global_common_heavy"],
        )
        reduced, direct, transitive = make_resource_fit_config(
            config, ["modules.clustering_kmeans.enabled"]
        )
        self.assertEqual(direct, ["modules.clustering_kmeans.enabled"])
        self.assertEqual(transitive, ["grouped_ml"])
        self.assertTrue(
            reduced["modules"]["structural_integrity_qc"]["enabled"]
        )
        self.assertFalse(reduced["modules"]["clustering_kmeans"]["enabled"])
        self.assertFalse(reduced["modules"]["grouped_ml"]["enabled"])

    def test_resource_fit_refuses_protected_switch(self):
        config = default_analysis_config(
            ["structural_integrity_qc", "replica_rmsd_rg"], []
        )
        with self.assertRaisesRegex(
            AnalysisConfigError, "protected module structural_integrity_qc"
        ):
            make_resource_fit_config(
                config, ["modules.structural_integrity_qc.enabled"]
            )

    def test_clustering_method_switches_filter_dedicated_and_alternative_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "clustering": {"methods": {
                    "kmeans": {"enabled": False},
                    "hdbscan": {"enabled": True},
                    "ward": {"enabled": False},
                }},
                "community_analysis": {"pald": {
                    "enabled": True,
                    "community_msm_enabled": True,
                }},
            }), encoding="utf-8")
            module_ids = [
                "common_pca", "time_lagged_independent_component_analysis",
                "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
                "clustering_imwkmeans", "alternative_clustering",
                "pald_community_analysis",
                "representative_frames", "grouped_ml", "markov_state_models",
            ]
            config = load_analysis_config(path, module_ids, ["global"])
            definitions, commands, requested, reasons = apply_module_configuration(
                {
                    "clustering_kmeans": {"feature_source": "common_pca"},
                    "clustering_hdbscan": {"feature_source": "common_pca"},
                    "clustering_imwkmeans": {"feature_source": "common_pca"},
                    "alternative_clustering": {
                        "feature_source": "common_pca", "algorithms": ["ward"],
                    },
                    "pald_community_analysis": {
                        "feature_source": "common_pca",
                        "community_msm_enabled": False,
                    },
                    "representative_frames": {"source": "clustering_kmeans"},
                    "grouped_ml": {},
                    "markov_state_models": {},
                },
                [
                    "cluster-kmeans", "cluster-hdbscan", "cluster-imwkmeans",
                    "alternative-clustering", "pald-community",
                    "representative-frames",
                    "grouped-ml", "markov-models",
                ],
                [
                    "clustering_kmeans", "clustering_hdbscan",
                    "clustering_imwkmeans", "alternative_clustering",
                    "pald_community_analysis",
                    "representative_frames", "grouped_ml", "markov_state_models",
                ],
                config,
            )
        self.assertNotIn("clustering_kmeans", definitions)
        self.assertNotIn("grouped_ml", definitions)
        self.assertEqual(definitions["representative_frames"]["source"], "pca_fes_basins")
        self.assertEqual(definitions["clustering_hdbscan"]["feature_source"], "tica")
        self.assertNotIn("ward", definitions["alternative_clustering"]["algorithms"])
        self.assertTrue(
            definitions["pald_community_analysis"]["community_msm_enabled"]
        )
        self.assertEqual(
            definitions["pald_community_analysis"]["feature_source"], "tica"
        )
        self.assertIn("pald-community", commands)
        self.assertNotIn("cluster-kmeans", commands)
        self.assertNotIn("clustering_kmeans", requested)
        self.assertIn("clustering_kmeans", reasons)

    def test_dependency_metadata_tracks_selected_clustering_feature_space(self):
        module_ids = [
            "common_pca", "time_lagged_independent_component_analysis",
            "clustering_kmeans",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({
                "config_schema": "salsbury-analysis-config-v1",
                "clustering": {"feature_space": "common_pca"},
            }), encoding="utf-8")
            config = load_analysis_config(path, module_ids, ["global"])
        self.assertEqual(
            config["modules"]["clustering_kmeans"]["depends_on"],
            ["common_pca"],
        )
        self.assertIn(
            "clustering_kmeans",
            config["modules"]["common_pca"]["turning_off_also_disables"],
        )


if __name__ == "__main__":
    unittest.main()
