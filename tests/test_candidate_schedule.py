"""Shared scheduling must govern candidate search and executable replay."""
import copy
import itertools
import unittest

from salsbury_md_analysis.execution_adapters import _slurm_resource_epochs, _RESOURCE_POLICY_DEFAULTS
from salsbury_md_analysis.planning_dependencies import bind_execution_dependencies
from salsbury_md_analysis.resource_planning import (
    plan_campaign_resource_budget, workflow_useful_parallel_cpu_ceiling,
)
from salsbury_md_analysis.resource_schedule import schedule_resource_tasks, ResourceScheduleError


def task(name, *, deps=(), stage=0, hours=4, cpus=1, memory=10):
    return {"task_id": name, "execution_bundle_id": name, "module_id": "test",
            "source_frames_per_replica": [100], "minimum_frames_per_replica": 100,
            "maximum_frames_per_replica": 100, "required_integer_stride": 1,
            "effective_cpu_cap": cpus, "estimated_peak_memory_gib": memory,
            "dependency_stage": stage, "cpu_seconds_per_physical_frame": 0,
            "fixed_cpu_hours": hours * cpus,
            "planning_dependencies": {"depends_on_bundle_ids": list(deps), "wait_for_bundle_ids": []}}


class CandidateScheduleTests(unittest.TestCase):
    def plan(self, rows, hours=8):
        return plan_campaign_resource_budget(rows, maximum_parallel_cpus=4,
            maximum_wall_hours=hours, maximum_memory_gib=185,
            planning_utilization=1, pilot_budget_fraction=0,
            memory_safety_factor=1.5, memory_overhead_gib=1,
            maximum_cpus_per_node=44, maximum_memory_gib_per_node=185, maximum_nodes=1)

    def test_candidate_uses_overlap_not_stage_sum_and_adapter_replays_it(self):
        rows = [task("a", stage=1), task("b", stage=2), task("c", deps=["a", "b"], stage=3, hours=1)]
        result = self.plan(rows, hours=6)
        self.assertEqual(result["feasibility_status"], "feasible")
        self.assertEqual(result["estimated_selected_wall_hours_lower_bound"], 5)
        self.assertEqual(result["effective_parallel_cpu_cap"], 2)
        executable = []
        for row in result["tasks"]:
            layout = row["parallel_node_layout_at_selected_observations"]
            executable.append({"task_id": "run-" + row["task_id"], "script": row["task_id"] + ".slurm",
                "planner_task_ids": [row["task_id"]], "cpu_slots": layout["execution_cpu_slots"],
                "planned_wall_hours": row["estimated_wall_hours_at_effective_cpu_cap"],
                "requested_wall_minutes": row["estimated_wall_hours_at_effective_cpu_cap"] * 60,
                "requested_memory_gib": row["estimated_scheduler_memory_gib_per_node_at_selected_observations"],
                "node_count": layout["node_count"], "workers_per_node": layout["workers_per_node"]})
        phases = [{"phase_id": "mixed", "tasks": executable}]
        bind_execution_dependencies(phases, result["tasks"])
        plan = {"maximum_parallel_cpus": result["effective_parallel_cpu_cap"],
                "maximum_parallel_memory_gib": 185, "node_policy": {"cpus_per_node":44,
                    "memory_gib_per_node":185, "maximum_nodes_per_campaign":1,"memory_reserve_gib":1},
                "phases":phases}
        replay = _slurm_resource_epochs(plan, {}, {}, {}, _RESOURCE_POLICY_DEFAULTS, plan["node_policy"])[0]
        self.assertEqual(replay["planned_wall_hours"], 5)
        native = result["stages"][0]["native_schedule"]
        self.assertEqual([t["planned_resource_start_hours"] for t in replay["scheduled_items"]],
                         [t["planned_resource_start_hours"] for t in native["scheduled_items"]])

    def test_real_dependencies_and_padded_memory_still_reject(self):
        result = self.plan([task("a"), task("b", deps=["a"])], hours=6)
        self.assertEqual(result["feasibility_status"], "infeasible")
        result = self.plan([task("a", memory=100), task("b", memory=100)], hours=6)
        self.assertEqual(result["feasibility_status"], "infeasible")
        self.assertEqual(result["estimated_selected_wall_hours_lower_bound"], 8)

    def test_added_nodes_keep_unchanged_smaller_placements(self):
        items = [{"item_id":str(i), "submission_index":i, "cpu_slots":1,
                  "memory_gib":120, "requested_memory_gib":120,"node_count":1,
                  "planned_wall_hours":i + 1, "wall_hours":i + 2,
                  "depends_on_task_ids":[], "wait_for_task_ids":[]}
                 for i in range(6)]
        unchanged = copy.deepcopy(items)
        previous = float("inf")
        for count in (1, 2, 3):
            schedule = schedule_resource_tasks(items, maximum_cpus=44*count, maximum_memory=185*count,
                node_policy={"cpus_per_node":44,"memory_gib_per_node":185,
                             "maximum_nodes_per_campaign":count,"memory_reserve_gib":1})[0]
            self.assertLessEqual(schedule["planned_wall_hours"], previous)
            previous = schedule["planned_wall_hours"]
            for time in {t["planned_resource_start_hours"] for t in schedule["scheduled_items"]}:
                active = [t for t in schedule["scheduled_items"]
                          if t["planned_resource_start_hours"] <= time < t["planned_resource_finish_hours"]]
                for node in range(count):
                    memory = sum(t["requested_memory_gib"] for t in active if node in t["planned_node_indices"])
                    self.assertLessEqual(memory + 1, 185)
        self.assertEqual(items, unchanged)

    def test_dependency_cpu_ceiling_matches_brute_force(self):
        for edge_bits in range(64):
            edges = [(j,i) for j in range(4) for i in range(j)]
            deps = {j:set() for j in range(4)}
            for bit,(child,parent) in enumerate(edges):
                if edge_bits & (1 << bit):
                    deps[child].add(parent)
            for child in range(4):
                for parent in list(deps[child]):
                    deps[child].update(deps[parent])
            rows = [task(str(i), cpus=i+1, deps=list(map(str,deps[i]))) for i in range(4)]
            possible = [subset for r in range(1,5) for subset in itertools.combinations(range(4),r)
                        if not any(a in deps[b] or b in deps[a] for a,b in itertools.combinations(subset,2))]
            expected = max(sum(i+1 for i in subset) for subset in possible)
            self.assertEqual(workflow_useful_parallel_cpu_ceiling(rows), expected)

    def test_unknown_and_cyclic_dependencies_fail(self):
        for deps in (["absent"], ["a"]):
            rows = [task("a", deps=deps)]
            if deps == ["a"]:
                with self.assertRaisesRegex(ValueError, "cycle"):
                    self.plan(rows)
            else:
                self.assertEqual(self.plan(rows)["feasibility_status"], "infeasible")


if __name__ == "__main__":
    unittest.main()
