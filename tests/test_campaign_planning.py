import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_planning import (
    _apply_measured_resource_calibrations,
    _automatic_context_tasks,
    _campaign_infeasibility_detail,
    _apply_system_memory_scaling,
    _view_tasks,
)


class CampaignPlanningTests(unittest.TestCase):
    def test_small_system_memory_scaling_retains_floor_and_headroom(self):
        tasks = [{
            "task_id": "small:sasa",
            "estimated_peak_memory_gib": 24.0,
            "power_law_cost_model": {
                "calibration_memory_gib": 10.0,
            },
        }]
        _apply_system_memory_scaling(tasks, 423)
        self.assertEqual(tasks[0]["memory_atom_scale"], 0.1)
        self.assertAlmostEqual(tasks[0]["estimated_peak_memory_gib"], 2.4)
        self.assertEqual(
            tasks[0]["power_law_cost_model"]["calibration_memory_gib"], 1.0
        )
        self.assertEqual(tasks[0]["reference_peak_memory_gib"], 24.0)

    def test_artifact_materialization_memory_floor_is_not_atom_scaled_away(self):
        tasks = [{
            "task_id": "small:derived-join",
            "estimated_peak_memory_gib": 16.0,
            "minimum_materialized_working_set_gib": 16.0,
        }]
        _apply_system_memory_scaling(tasks, 423)
        self.assertEqual(tasks[0]["memory_atom_scale"], 0.1)
        self.assertEqual(tasks[0]["estimated_peak_memory_gib"], 16.0)

    def test_measured_overlay_preserves_declared_workload_scaling(self):
        calibration = {
            "coordinate_cache": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 1024.0,
                "catalog_sha256": "a" * 64,
                "measurement_count": 1,
                "complete_measurement_count": 1,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "complete_only",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100,
                "maximum_measured_observation_count": 100,
            }
        }
        scaled = {
            "task_id": "preprocessing:coordinate_cache",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 1.0,
            "measured_cpu_rate_multiplier": 0.1,
            "measured_memory_multiplier": 0.1,
        }
        reference = {
            "task_id": "reference",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 1.0,
        }
        _apply_measured_resource_calibrations(
            [scaled, reference], calibration,
            time_safety_factor=1.5,
        )
        self.assertAlmostEqual(scaled["cpu_seconds_per_physical_frame"], 0.075)
        self.assertAlmostEqual(reference["cpu_seconds_per_physical_frame"], 0.75)
        self.assertEqual(
            scaled["measured_resource_calibration"]["cpu_rate_workload_multiplier"],
            0.1,
        )

    def test_qualified_memory_measurement_replaces_legacy_default(self):
        calibration = {
            "coordinate_cache": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 1024.0,
                "maximum_completed_resident_memory_mib": 1024.0,
                "catalog_sha256": "b" * 64,
                "measurement_count": 2,
                "complete_measurement_count": 2,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100,
                "maximum_measured_observation_count": 100,
                "memory_replacement_qualified": True,
                "memory_replacement_policy": (
                    "replace_legacy_baseline_with_conservative_completed_measurement"
                ),
            }
        }
        task = {
            "task_id": "preprocessing:coordinate_cache",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 24.0,
        }
        _apply_measured_resource_calibrations(
            [task], calibration,
            time_safety_factor=1.5,
        )
        self.assertAlmostEqual(task["estimated_peak_memory_gib"], 1.0)
        self.assertAlmostEqual(
            task["measured_memory_cost_model"]["calibration_memory_gib"],
            1.0,
        )
        self.assertTrue(
            task["measured_resource_calibration"]["memory_replacement_qualified"]
        )

    def test_qualified_memory_recalibrates_power_model_at_measured_workload(self):
        calibration = {
            "alternative_clustering": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 16384.0,
                "maximum_completed_resident_memory_mib": 16384.0,
                "catalog_sha256": "c" * 64,
                "measurement_count": 2,
                "complete_measurement_count": 2,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100_000,
                "maximum_measured_observation_count": 100_000,
                "memory_replacement_qualified": True,
            }
        }
        task = {
            "task_id": "clustering:alternative",
            "module_id": "alternative_clustering",
            "fixed_cpu_hours": 1.0,
            "estimated_peak_memory_gib": 32.0,
            "power_law_cost_model": {
                "calibration_observations": 3_000,
                "calibration_cpu_hours": 1.0,
                "time_exponent": 2.0,
                "calibration_memory_gib": 32.0,
                "memory_exponent": 1.0,
            },
        }
        _apply_measured_resource_calibrations(
            [task], calibration,
            time_safety_factor=1.5,
        )
        model = task["measured_memory_cost_model"]
        self.assertEqual(model["calibration_observations"], 100_000)
        self.assertEqual(model["calibration_memory_gib"], 16.0)
        self.assertEqual(model["memory_exponent"], 1.0)

    def test_base_automatic_chemistry_tasks_are_budgeted_in_dependency_order(self):
        project = {
            "requested_modules": [
                "trajectory_features", "scalar_feature_distributions",
                "scalar_threshold_states",
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _automatic_context_tasks(
                path, [1_000], time_safety_factor=1.5,
                context_id="base", task_namespace="base",
                task_scope="base_automatic_chemistry",
            )
        self.assertEqual(
            {row["task_id"] for row in tasks},
            {
                "base:trajectory_features",
                "base:scalar_feature_distributions",
                "base:scalar_threshold_states",
            },
        )
        stages = {row["module_id"]: row["dependency_stage"] for row in tasks}
        self.assertLess(
            stages["trajectory_features"], stages["scalar_feature_distributions"]
        )
        scalar = next(
            row for row in tasks
            if row["module_id"] == "scalar_feature_distributions"
        )
        self.assertFalse(scalar["measured_calibration_eligible"])
        self.assertIn(
            "validated upstream report",
            scalar["measured_calibration_exclusion_reason"],
        )

    def test_infeasible_message_reports_actual_shortfall_and_retry_bound(self):
        detail = _campaign_infeasibility_detail({
            "infeasibility_reasons": ["minimum wall budget exceeded"],
            "maximum_wall_hours_input": 24.0,
            "science_budget_wall_hours": 16.8,
            "minimum_wall_hours_lower_bound": 20.0,
            "minimum_known_cpu_hours": 200.0,
            "science_budget_cpu_hours": 537.6,
            "maximum_parallel_cpus_input": 32,
        })
        self.assertIn("minimum calibrated critical path 20.000 h", detail)
        self.assertIn("science wall allowance 16.800 h", detail)
        self.assertIn("--target-wall-hours 29", detail)

    def test_grouped_ml_rejects_insufficient_time_block_groups(self):
        project = {
            "requested_modules": ["common_pca", "grouped_ml"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "grouped_ml": {
                    "group_block_size_frames": 10,
                    "minimum_groups": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-groups.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 3 segment/time-block groups"
            ):
                _view_tasks(path, [30], 10_000, time_safety_factor=1.5)

    def test_grouped_ml_carries_group_minimum_into_plan(self):
        project = {
            "requested_modules": ["common_pca", "grouped_ml"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "grouped_ml": {
                    "group_block_size_frames": 10,
                    "minimum_groups": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-minimum-groups.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [31], 10_000, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks if row["module_id"] == "grouped_ml"
        )
        self.assertEqual(task["minimum_frames_per_replica"], 50)
        self.assertEqual(task["minimum_frames_per_replica_for_groups"], 31)
        self.assertEqual(task["maximum_available_groups"], 4)

    def test_information_dynamics_rejects_insufficient_lag_pairs(self):
        project = {
            "requested_modules": ["common_pca", "information_dynamics"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "information_dynamics": {
                    "analyses": ["transfer_entropy", "coskewness"],
                    "lag_frames": 1,
                    "minimum_pairs": 20,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 19 segment-safe lag pairs"
            ):
                _view_tasks(path, [20], 10_000, time_safety_factor=1.5)

    def test_information_dynamics_carries_lag_pair_minimum_into_plan(self):
        project = {
            "requested_modules": ["common_pca", "information_dynamics"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "information_dynamics": {
                    "analyses": ["transfer_entropy", "coskewness"],
                    "lag_frames": 1,
                    "minimum_pairs": 20,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-minimum-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [21], 10_000, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks
            if row["module_id"] == "information_dynamics"
        )
        self.assertEqual(task["minimum_frames_per_replica"], 50)
        self.assertEqual(
            task["minimum_frames_per_replica_for_lag_pairs"], 21
        )
        self.assertEqual(task["minimum_lag_pairs"], 20)
        self.assertEqual(task["maximum_available_lag_pairs"], 20)

    def test_tica_requires_configured_pairs_in_every_segment(self):
        project = {
            "requested_modules": [
                "common_pca",
                "time_lagged_independent_component_analysis",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "time_lagged_independent_component_analysis": {
                    "lag_frames": 3,
                    "minimum_pairs_per_segment": 10,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-tica.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "at least 13 projected frames"
            ):
                _view_tasks(path, [12], 10_000, time_safety_factor=1.5)

    def test_tica_carries_configured_lag_pair_minimum_into_plan(self):
        project = {
            "requested_modules": [
                "common_pca",
                "time_lagged_independent_component_analysis",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "time_lagged_independent_component_analysis": {
                    "lag_frames": 3,
                    "minimum_pairs_per_segment": 10,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-tica.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(path, [20], 10_000, time_safety_factor=1.5)
        task = next(
            row for row in tasks
            if row["module_id"]
            == "time_lagged_independent_component_analysis"
        )
        self.assertEqual(task["lag_frames"], 3)
        self.assertEqual(task["minimum_lag_pairs_per_segment"], 10)
        self.assertEqual(
            task["minimum_frames_per_replica_for_lag_pairs"], 13
        )
        self.assertEqual(task["maximum_available_lag_pairs"], 17)

    def test_msm_uses_largest_configured_lag_and_transition_minimum(self):
        project = {
            "requested_modules": ["common_pca", "markov_state_models"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "markov_state_models": {
                    "lag_frames": [1, 3, 5],
                    "minimum_transition_count": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-msm.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(path, [10], 10_000, time_safety_factor=1.5)
        task = next(
            row for row in tasks if row["module_id"] == "markov_state_models"
        )
        self.assertEqual(task["largest_configured_lag_frames"], 5)
        self.assertEqual(task["minimum_transition_count"], 4)
        self.assertEqual(
            task["minimum_frames_per_replica_for_transition_pairs"], 9
        )
        self.assertEqual(
            task["maximum_available_transition_pairs_at_largest_lag"], 5
        )

    def test_msm_rejects_insufficient_configured_transition_pairs(self):
        project = {
            "requested_modules": ["common_pca", "markov_state_models"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "markov_state_models": {
                    "lag_frames": [1, 5],
                    "minimum_transition_count": 2,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-msm.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 0 segment-safe transition pairs"
            ):
                _view_tasks(path, [5], 10_000, time_safety_factor=1.5)

    def test_alternative_families_are_independent_logical_tasks_in_one_bundle(self):
        project = {
            "requested_modules": ["common_pca", "alternative_clustering"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 5616,
                    "symmetry_expansion": {"member_count": 2},
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "alternative_clustering": {
                    "algorithms": ["pam", "gaussian_mixture", "ward"],
                    "component_indices": list(range(1, 11)),
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-member-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [10_000, 10_000, 10_000], 100_000,
                time_safety_factor=1.5,
            )
        algorithms = {
            row["algorithm_id"]: row for row in tasks
            if row.get("task_scope") == "conformational_view_algorithm_fit"
        }
        self.assertEqual(set(algorithms), {"pam", "gaussian_mixture"})
        self.assertEqual(
            algorithms["pam"]["execution_bundle_id"],
            algorithms["gaussian_mixture"]["execution_bundle_id"],
        )
        self.assertNotEqual(
            algorithms["pam"]["balance_group"],
            algorithms["gaussian_mixture"]["balance_group"],
        )
        self.assertEqual(
            algorithms["pam"]["power_law_cost_model"]["time_exponent"], 2.0
        )
        self.assertEqual(
            algorithms["gaussian_mixture"]["power_law_cost_model"][
                "time_exponent"
            ],
            1.0,
        )
        self.assertEqual(
            algorithms["pam"]["algorithm_memory_model"],
            "feature_aware_pairwise_matrix",
        )
        self.assertEqual(
            algorithms["pam"]["projection_feature_count"], 10
        )
        self.assertAlmostEqual(
            algorithms["pam"]["measured_memory_cost_model"][
                "calibration_memory_gib"
            ],
            0.125 + 20_000 ** 2 * 186 / 2 ** 30,
        )
        self.assertEqual(
            algorithms["gaussian_mixture"]["measured_memory_cost_model"][
                "calibration_memory_gib"
            ],
            1.0,
        )
        self.assertTrue(algorithms["pam"]["memory_replacement_qualified"])
        self.assertFalse(algorithms["pam"]["measured_calibration_eligible"])
        self.assertEqual(
            algorithms["pam"]["performance_calibration_provenance"][
                "measurement_count"
            ],
            120,
        )
        self.assertFalse(
            algorithms["pam"]["performance_calibration_provenance"][
                "scientific_observations_saved"
            ]
        )

    def test_pald_uses_separate_measured_cubic_observation_model(self):
        project = {
            "requested_modules": ["common_pca", "pald_community_analysis"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 5616,
                    "symmetry_expansion": {"member_count": 2},
                },
                "pald_community_analysis": {"maximum_observations": 500},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-member-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [100_000, 100_000, 100_000], 100_000,
                time_safety_factor=1.5,
            )
        pald = next(
            row for row in tasks
            if row["module_id"] == "pald_community_analysis"
        )
        self.assertEqual(pald["fit_observation_limit"], 500)
        self.assertEqual(
            pald["calibration_status"], "completed_bounded_cubic_calibration"
        )
        self.assertAlmostEqual(
            pald["fixed_cpu_hours"], (0.5 + 28.637) * 1.5 / 3600.0
        )
        self.assertEqual(pald["cpu_seconds_per_physical_frame"], 0.0)
        self.assertEqual(pald["estimated_peak_memory_gib"], 1.0)

    def test_representative_only_state_exports_do_not_scale_as_trajectories(self):
        project = {
            "requested_modules": ["common_pca", "state_coordinate_exports"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "state_coordinate_exports": {"write_trajectories": False},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-representative-exports.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [1_000, 1_000], 10_000, time_safety_factor=1.5
            )
        export = next(
            row for row in tasks
            if row["module_id"] == "state_coordinate_exports"
        )
        self.assertEqual(export["coordinate_export_mode"], "representatives_only")
        self.assertFalse(export["state_trajectory_exports_enabled"])
        self.assertFalse(export["measured_calibration_eligible"])
        self.assertEqual(export["cpu_seconds_per_physical_frame"], 0.0)
        self.assertEqual(export["fixed_cpu_hours"], 0.20)
        self.assertIn(
            "multi-frame trajectory materialization",
            export["measured_calibration_exclusion_reason"],
        )


if __name__ == "__main__":
    unittest.main()
