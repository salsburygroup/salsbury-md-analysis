import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_planning import (
    _apply_measured_resource_calibrations,
    _automatic_context_tasks,
    _campaign_infeasibility_detail,
    _view_tasks,
)


class CampaignPlanningTests(unittest.TestCase):
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
            time_safety_factor=1.5, memory_safety_factor=1.25,
        )
        self.assertAlmostEqual(scaled["cpu_seconds_per_physical_frame"], 0.075)
        self.assertAlmostEqual(reference["cpu_seconds_per_physical_frame"], 0.75)
        self.assertEqual(
            scaled["measured_resource_calibration"]["cpu_rate_workload_multiplier"],
            0.1,
        )

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
                    "algorithms": ["pam", "gaussian_mixture", "ward"]
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


if __name__ == "__main__":
    unittest.main()
