import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.execution_adapters import (
    ExecutionAdapterError, prepare_execution_artifacts,
    validate_native_campaign_schedule,
)
from salsbury_md_analysis.quickstart import _slurm_files


class NativeScheduleGateTests(unittest.TestCase):
    def fixture(self, root, *, sequential=False):
        duration = 24.391027457724757
        resources = {
            "native_schedule_validation": {
                "required": True, "status": "pending",
                "stage_schedule_estimated_wall_hours": 55.91,
            },
            "feasibility_status": "pending_execution_schedule",
            "infeasibility_reasons": [
                "minimum calibrated critical path exceeds the campaign science wall-time budget"
            ],
            "fixed_sampling": {"sampling_preserved": True},
            "science_budget_wall_hours": 27.0,
            "science_budget_cpu_hours": 1188.0,
            "tasks": [{"task_id": name, "estimated_cpu_hours": duration}
                      for name in ("a", "b")],
        }
        tasks = [{
            "task_id": name, "script": name + ".slurm", "planner_task_ids": [name],
            "cpu_slots": 1, "planned_wall_hours": duration,
            "requested_wall_minutes": 1600.0, "requested_memory_gib": 2.0,
            "depends_on_task_ids": ["a"] if sequential and name == "b" else [],
            "wait_for_task_ids": [],
        } for name in ("a", "b")]
        plan = {
            "maximum_parallel_cpus": 44, "maximum_parallel_memory_gib": 185.0,
            "maximum_campaign_wall_hours": 48.0,
            "node_policy": {"cpus_per_node": 44, "memory_gib_per_node": 185,
                            "maximum_nodes_per_campaign": 1, "memory_reserve_gib": 1},
            "resource_policy": {"campaign_walltime_headroom_fraction": 1 / 3,
                                "campaign_walltime_rounding_minutes": 60},
            "phases": [{"phase_id": "stage1", "tasks": tasks[:1]},
                       {"phase_id": "stage2", "tasks": tasks[1:]}],
        }
        self.save(root, resources)
        return resources, plan

    def save(self, root, resources):
        (root / "campaign-resource-plan.json").write_text(json.dumps(resources))
        (root / "sampling-plan.json").write_text(json.dumps({"campaign_resource_plan": resources}))

    def test_overlapping_native_schedule_replaces_coarse_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            resources, plan = self.fixture(root)
            before = copy.deepcopy(plan)
            result = validate_native_campaign_schedule(root, plan, adapter="slurm")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["campaign_walltime_request"]["requested_wall_hours"], 33)
            self.assertEqual(result["science_budget_wall_hours"], 27)
            self.assertEqual(plan, before)
            after = json.loads((root / "campaign-resource-plan.json").read_text())
            self.assertEqual(after["feasibility_status"], "feasible")
            self.assertEqual(after["tasks"], resources["tasks"])
            self.assertEqual(after["native_schedule_validation"]["stage_schedule_estimated_wall_hours"], 55.91)
            # Replaying the same schedule gives the same binding and decision.
            repeated = validate_native_campaign_schedule(root, plan, adapter="slurm")
            self.assertEqual(repeated["schedule_sha256"], result["schedule_sha256"])
            self.assertEqual(repeated["status"], "complete")

    def test_real_dependencies_still_reject_and_write_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plan = self.fixture(root, sequential=True)
            with self.assertRaisesRegex(ExecutionAdapterError, "No feasible fixed-sampling plan.*science wall allowance"):
                validate_native_campaign_schedule(root, plan, adapter="slurm")
            result = json.loads((root / "native-schedule-validation.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["campaign_walltime_request"]["submission_time_feasible"])

    def test_missing_duplicate_or_unmapped_work_is_rejected(self):
        for defect in ("missing", "duplicate", "unmapped"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, plan = self.fixture(root)
                plan["phases"][1]["tasks"][0]["planner_task_ids"] = {
                    "missing": [], "duplicate": ["a"], "unmapped": ["unknown"],
                }[defect]
                with self.assertRaisesRegex(ExecutionAdapterError, "every logical task exactly once"):
                    validate_native_campaign_schedule(root, plan, adapter="slurm")

    def test_cpu_and_scientific_failures_are_not_overridden(self):
        for defect in ("cpu", "science", "calibration"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                resources, plan = self.fixture(root)
                if defect == "cpu":
                    resources["science_budget_cpu_hours"] = 1.0
                elif defect == "science":
                    resources["scientific_sampling_feasibility"] = {"below_standard_tasks": ["a"]}
                else:
                    resources["tasks_requiring_project_pilots"] = ["a"]
                self.save(root, resources)
                with self.assertRaises(ExecutionAdapterError):
                    validate_native_campaign_schedule(root, plan, adapter="slurm")

    def test_memory_reserve_serializes_independent_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plan = self.fixture(root)
            for phase in plan["phases"]:
                phase["tasks"][0]["requested_memory_gib"] = 92.5
            # 185 GiB of tasks plus the once-per-node 1 GiB reserve cannot overlap.
            with self.assertRaisesRegex(ExecutionAdapterError, "science wall allowance"):
                validate_native_campaign_schedule(root, plan, adapter="slurm")

    def test_failed_native_validation_never_creates_launchers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plan = self.fixture(root, sequential=True)
            with patch("salsbury_md_analysis.execution_adapters.build_local_execution_plan", return_value=plan):
                with self.assertRaises(ExecutionAdapterError):
                    prepare_execution_artifacts(root, {"execution": {"submission_adapter": "local"}, "reporting": {}})
            for name in ("run-local.sh", "run-custom.sh", "submit.sh", "run-campaign.slurm", "launcher-contract.json"):
                self.assertFalse((root / name).exists(), name)

    def test_local_validation_does_not_add_slurm_allowance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, plan = self.fixture(root)
            result = validate_native_campaign_schedule(root, plan, adapter="local")
            self.assertEqual(result["status"], "complete")
            self.assertIsNone(result["campaign_walltime_request"])

    def test_interim_submit_script_stops_before_scheduler_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            _slurm_files(root, "pending-plan", ["rmsd-rg"], target_wall_hours=48,
                         python_executable=sys.executable, package_root=str(root))
            result = subprocess.run(["bash", str(root / "submit.sh")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("submission is disabled", result.stderr)
            self.assertNotIn("sbatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
