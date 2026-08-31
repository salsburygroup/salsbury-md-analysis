import unittest

from salsbury_md_analysis.memory_policy import (
    apply_memory_calibration_uncertainty,
    resolve_memory_uncertainty_policy,
)
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget


class MemoryPolicyTests(unittest.TestCase):
    def test_named_factors_distinguish_qualified_and_weak_models(self):
        policy = resolve_memory_uncertainty_policy({
            "well_calibrated_memory_uncertainty_factor": 1.0,
            "poorly_calibrated_memory_uncertainty_factor": 1.25,
        })
        qualified = {
            "estimated_peak_memory_gib": 8.0,
            "measured_resource_calibration": {
                "memory_replacement_qualified": True,
            },
            "measured_memory_cost_model": {"calibration_memory_gib": 8.0},
        }
        weak = {
            "estimated_peak_memory_gib": 8.0,
            "power_law_cost_model": {"calibration_memory_gib": 8.0},
        }
        apply_memory_calibration_uncertainty([qualified, weak], policy)
        self.assertEqual(qualified["estimated_peak_memory_gib"], 8.0)
        self.assertEqual(weak["estimated_peak_memory_gib"], 10.0)
        self.assertEqual(
            qualified["memory_calibration_uncertainty"]["quality"],
            "well_calibrated",
        )
        self.assertEqual(
            weak["power_law_cost_model"]["calibration_memory_gib"], 10.0
        )

    def test_application_is_idempotent_for_same_policy(self):
        policy = resolve_memory_uncertainty_policy({})
        task = {"estimated_peak_memory_gib": 4.0}
        apply_memory_calibration_uncertainty([task], policy)
        apply_memory_calibration_uncertainty([task], policy)
        self.assertEqual(task["estimated_peak_memory_gib"], 5.0)

    def test_deac_adjustment_follows_uncertainty_and_is_final(self):
        task = {
            "task_id": "direct:test",
            "module_id": "test",
            "task_scope": "direct_trajectory_estimator",
            "dependency_stage": 0,
            "effective_cpu_cap": 1,
            "source_frames_per_replica": [100],
            "minimum_frames_per_replica": 10,
            "maximum_frames_per_replica": 100,
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 8.0,
        }
        apply_memory_calibration_uncertainty(
            [task], resolve_memory_uncertainty_policy({})
        )
        plan = plan_campaign_resource_budget(
            [task],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=32.0,
            memory_safety_factor=1.5,
            memory_overhead_gib=1.0,
            minimum_scheduler_memory_gib=0.0,
        )
        row = plan["tasks"][0]
        self.assertEqual(
            row["estimated_peak_memory_gib_at_selected_observations"], 10.0
        )
        self.assertEqual(
            row["estimated_scheduler_memory_gib_at_selected_observations"],
            16.0,
        )


if __name__ == "__main__":
    unittest.main()
