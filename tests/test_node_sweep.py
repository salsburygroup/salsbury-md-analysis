import unittest
from unittest.mock import patch

from salsbury_md_analysis.node_sweep import (
    maximum_useful_node_inventory,
    plan_node_sweep,
    scientific_minimum_multiples,
)
from salsbury_md_analysis.resource_planning import ResourcePlanningError
from salsbury_md_analysis.scientific_sampling import (
    profile_contract,
    scientific_sampling_profile,
)


class NodeSweepTests(unittest.TestCase):
    @staticmethod
    def _task(task_id="analysis", rate=1.0):
        return {
            "task_id": task_id,
            "module_id": "hydrogen_bond_discovery",
            "dependency_stage": 1,
            "effective_cpu_cap": 1,
            "intrinsic_cpu_cap": 1,
            "source_frames_per_replica": [1_000, 1_000],
            "minimum_frames_per_replica": 500,
            "maximum_frames_per_replica": 1_000,
            "cpu_seconds_per_physical_frame": rate,
            "fixed_cpu_hours": 0.0,
            "estimated_peak_memory_gib": 1.0,
            "priority_weight": 1.0,
            "replica_sampling_mode": "balanced_pooled",
            "scientific_sampling_requirements": profile_contract(
                scientific_sampling_profile("hydrogen_bond_discovery")
            ),
        }

    def test_scientific_minimum_multiple_uses_physical_frames(self):
        task = self._task()
        task["selected_physical_frames_per_replica"] = [600, 600]
        summary = scientific_minimum_multiples([task])
        self.assertEqual(summary["task_count"], 1)
        self.assertEqual(summary["mean_multiple_of_scientific_minimum"], 1.2)
        self.assertEqual(summary["median_multiple_of_scientific_minimum"], 1.2)

    def test_sweep_uses_all_cores_per_node_and_selects_smallest_plateau(self):
        report = plan_node_sweep(
            [self._task(rate=30.0)],
            cpus_per_node=44,
            memory_gib_per_node=185.0,
            maximum_nodes=3,
            maximum_wall_hours=24.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
            finalization_headroom_fraction=0.0,
            memory_safety_factor=1.0,
            memory_overhead_gib=0.0,
        )
        self.assertEqual(
            [row["requested_parallel_cpus"] for row in report["curve"]],
            [44, 88, 132],
        )
        self.assertEqual(report["sweet_spot"]["requested_nodes"], 1)
        self.assertEqual(
            [row["fraction_of_best_threshold"]
             for row in report["threshold_sensitivity"]],
            [0.75, 0.80, 0.90, 0.95, 0.99, 1.00],
        )
        self.assertEqual(
            report["task_inventory_ceiling"][
                "maximum_useful_nodes_within_campaign_cap"
            ],
            1,
        )
        self.assertEqual(
            [row["above_task_inventory_useful_node_ceiling"]
             for row in report["curve"]],
            [False, True, True],
        )
        self.assertGreater(report["curve"][0]["planned_makespan_hours"], 0.0)
        self.assertEqual(report["curve"][0]["planned_peak_node_count"], 1)
        self.assertGreater(
            report["curve"][0]["planned_reserved_node_hours"], 0.0
        )
        self.assertEqual(
            report["curve"][0]["selected_stride_by_balance_group"],
            {"analysis": 1},
        )
        self.assertEqual(
            report["threshold_sensitivity"][0]["planned_makespan_hours"],
            report["curve"][0]["planned_makespan_hours"],
        )
        self.assertEqual(
            report["operational_balance"]["recommended_node_count"], 1
        )
        self.assertFalse(report["operational_balance"]["queue_wait_included"])
        self.assertEqual(
            report["operational_balance"]["scientific_minimum_multiples"][
                "task_count"
            ],
            1,
        )
        self.assertTrue(report["curve"][0]["pareto_efficient"])
        self.assertFalse(report["execution_started"])
        self.assertFalse(report["jobs_submitted"])

    def test_sweep_replays_better_smaller_plan_on_raw_heuristic_regression(self):
        plans = []
        for coverage in (0.81, 0.04, 1.0):
            task = self._task()
            task.update({
                "coverage_fraction": coverage,
                "selected_member_observation_count": int(2_000 * coverage),
                "selected_physical_frames_per_replica": [500, 500],
            })
            plans.append({
                "feasibility_status": "feasible",
                "tasks": [task],
                "stages": [{
                    "dependency_stage": 1,
                    "planned_node_count": len(plans) + 1,
                    "estimated_wall_hours_with_resource_lanes": 1.0,
                }],
                "estimated_selected_cpu_hours": 1.0,
                "estimated_selected_wall_hours_lower_bound": 1.0,
                "minimum_known_cpu_hours": 1.0,
                "minimum_wall_hours_lower_bound": 1.0,
            })
        plans_by_nodes = {
            index: plan for index, plan in enumerate(plans, start=1)
        }
        def fake_plan(_tasks, **kwargs):
            return plans_by_nodes[int(kwargs["maximum_nodes"])]
        with patch(
            "salsbury_md_analysis.node_sweep.plan_campaign_resource_budget",
            side_effect=fake_plan,
        ):
            report = plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=3,
                maximum_wall_hours=24.0,
            )
        self.assertEqual(
            [round(row["balanced_information_utility"], 6)
             for row in report["curve"]],
            [0.9, 0.9, 1.0],
        )
        self.assertEqual(
            [row["replayed_from_node_count"] for row in report["curve"]],
            [1, 1, 3],
        )
        self.assertEqual(
            [round(row["raw_balanced_information_utility"], 6)
             for row in report["curve"]],
            [0.9, 0.2, 1.0],
        )

    def test_minimum_nodes_policy_selects_smallest_pareto_point(self):
        plans = []
        for coverage, wall in ((0.64, 10.0), (1.0, 5.0)):
            task = self._task()
            task.update({
                "coverage_fraction": coverage,
                "selected_member_observation_count": int(2_000 * coverage),
                "selected_physical_frames_per_replica": [500, 500],
            })
            plans.append({
                "feasibility_status": "feasible",
                "tasks": [task],
                "stages": [{
                    "dependency_stage": 1,
                    "planned_node_count": len(plans) + 1,
                    "estimated_wall_hours_with_resource_lanes": wall,
                }],
                "estimated_selected_cpu_hours": 1.0,
                "estimated_selected_wall_hours_lower_bound": wall,
                "minimum_known_cpu_hours": 1.0,
                "minimum_wall_hours_lower_bound": wall,
            })
        plans_by_nodes = {
            index: plan for index, plan in enumerate(plans, start=1)
        }

        def fake_plan(_tasks, **kwargs):
            return plans_by_nodes[int(kwargs["maximum_nodes"])]

        with patch(
            "salsbury_md_analysis.node_sweep.plan_campaign_resource_budget",
            side_effect=fake_plan,
        ):
            balanced = plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=2,
                maximum_wall_hours=24.0,
                pareto_selection_policy="balanced",
            )
        with patch(
            "salsbury_md_analysis.node_sweep.plan_campaign_resource_budget",
            side_effect=fake_plan,
        ):
            minimum_nodes = plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=2,
                maximum_wall_hours=24.0,
            )

        self.assertEqual(
            balanced["operational_balance"]["recommended_node_count"], 2
        )
        self.assertEqual(
            minimum_nodes["operational_balance"]["recommended_node_count"], 1
        )
        self.assertEqual(
            minimum_nodes["operational_balance"]["selection_policy"],
            "minimum_nodes",
        )
        self.assertTrue(
            minimum_nodes["operational_balance"]["pareto_filtering_enabled"]
        )
        self.assertEqual(
            minimum_nodes["operational_balance"]["pareto_front_node_counts"],
            [1, 2],
        )
        self.assertTrue(minimum_nodes["curve"][0]["pareto_efficient"])

        with patch(
            "salsbury_md_analysis.node_sweep.plan_campaign_resource_budget",
            side_effect=fake_plan,
        ):
            walltime_information = plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=2,
                maximum_wall_hours=24.0,
                pareto_objectives="walltime_information",
            )
        self.assertEqual(
            walltime_information["operational_balance"][
                "pareto_front_node_counts"
            ],
            [2],
        )
        self.assertEqual(
            walltime_information["operational_balance"][
                "pareto_objective_mode"
            ],
            "walltime_information",
        )
        self.assertEqual(
            walltime_information["operational_balance"]["pareto_objectives"],
            ["planned_makespan_hours", "information"],
        )

    def test_unknown_pareto_selection_policy_fails_closed(self):
        with self.assertRaisesRegex(
            ResourcePlanningError, "pareto_selection_policy must be one of"
        ):
            plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=2,
                maximum_wall_hours=24.0,
                pareto_selection_policy="fewest_cores",
            )

    def test_unknown_pareto_objectives_fail_closed(self):
        with self.assertRaisesRegex(
            ResourcePlanningError, "pareto_objectives must be one of"
        ):
            plan_node_sweep(
                [self._task()], cpus_per_node=44,
                memory_gib_per_node=185.0, maximum_nodes=2,
                maximum_wall_hours=24.0,
                pareto_objectives="walltime_only",
            )

    def test_full_inventory_ceiling_uses_tasks_and_scheduler_memory(self):
        tasks = []
        for index in range(10):
            task = self._task(task_id=f"task-{index}")
            task.update({
                "estimated_peak_memory_gib": 49.0,
                "source_frames_per_replica": [1_000],
            })
            tasks.append(task)
        inventory = maximum_useful_node_inventory(
            tasks,
            cpus_per_node=44,
            memory_gib_per_node=185.0,
            maximum_nodes=16,
            memory_safety_factor=1.0,
            memory_overhead_gib=1.0,
        )
        self.assertEqual(
            inventory["uncapped_task_inventory_node_ceiling"], 4
        )
        self.assertEqual(
            inventory["maximum_useful_nodes_within_campaign_cap"], 4
        )
        self.assertEqual(
            inventory["dependency_stages"][0]["task_count"], 10
        )

    def test_parallel_workers_set_full_inventory_node_ceiling(self):
        task = self._task()
        task.update({
            "effective_cpu_cap": 100,
            "intrinsic_cpu_cap": 100,
            "parallel_execution_model": (
                "replica_worker_exact_global_reducer_v1"
            ),
            "parallel_worker_count": 100,
            "estimated_peak_memory_gib_per_parallel_worker": 5.0,
            "reducer_memory_gib": 5.0,
        })
        inventory = maximum_useful_node_inventory(
            [task], cpus_per_node=44, memory_gib_per_node=185.0,
            maximum_nodes=2, memory_safety_factor=1.0,
            memory_overhead_gib=0.0,
        )
        self.assertEqual(
            inventory["uncapped_task_inventory_node_ceiling"], 3
        )
        self.assertEqual(
            inventory["maximum_useful_nodes_within_campaign_cap"], 2
        )
        self.assertTrue(inventory["campaign_cap_limits_full_parallelism"])


if __name__ == "__main__":
    unittest.main()
