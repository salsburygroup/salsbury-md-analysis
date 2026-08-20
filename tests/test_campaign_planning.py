import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_planning import _view_tasks


class CampaignPlanningTests(unittest.TestCase):
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
