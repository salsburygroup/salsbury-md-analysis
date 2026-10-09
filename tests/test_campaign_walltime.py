import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_walltime import campaign_walltime_budget, campaign_walltime_request
from salsbury_md_analysis.execution_adapters import (
    ExecutionAdapterError, apply_slurm_profile, load_slurm_profile,
    run_local_workflow, execution_walltime_budget,
)


class CampaignWalltimeTests(unittest.TestCase):
    def test_buffered_eleven_point_seven_five_becomes_sixteen(self):
        r = campaign_walltime_request(11.747817823072733, 48)
        self.assertEqual(r["requested_wall_hours"], 16)
        self.assertEqual(r["estimated_execution_hours"], 11.747817823072733)
        self.assertTrue(r["submission_time_feasible"])

    def test_rounding_boundaries_and_explicit_policy(self):
        for estimate, expected in [(0, 1), (0.1, 1), (12, 16), (12.01, 17)]:
            self.assertEqual(campaign_walltime_request(estimate, 48)["requested_wall_hours"], expected)
        self.assertEqual(campaign_walltime_request(11.75, 48, {
            "campaign_walltime_headroom_fraction": 0.25,
            "campaign_walltime_rounding_minutes": 30,
        })["requested_wall_hours"], 15)

    def test_user_ceiling_preserves_full_headroom(self):
        r = campaign_walltime_request(11.75, 15)
        self.assertFalse(r["submission_time_feasible"])
        self.assertIsNone(r["requested_wall_hours"])
        self.assertFalse(r["headroom_limited_by_campaign_cap"])
        r = campaign_walltime_request(16, 15)
        self.assertFalse(r["submission_time_feasible"])
        self.assertIsNone(r["requested_wall_hours"])

    def test_different_padded_limits_reserve_allowance_before_planning(self):
        for cap, execution in [(8, 6), (16, 12), (24, 18), (48, 36), (168, 126), (48.5, 36)]:
            with self.subTest(cap=cap):
                budget = campaign_walltime_budget(cap)
                self.assertEqual(budget["maximum_estimated_execution_hours"], execution)
                self.assertLessEqual(campaign_walltime_request(execution, cap)["requested_wall_hours"], cap)
                self.assertFalse(campaign_walltime_request(execution + 0.01, cap)["submission_time_feasible"])
        self.assertEqual(campaign_walltime_budget(48, {"campaign_walltime_headroom_fraction": 0.5})["maximum_estimated_execution_hours"], 32)
        self.assertEqual(execution_walltime_budget({"maximum_hours_per_cpu": 48, "submission_adapter": "local"})["maximum_estimated_execution_hours"], 48)
        repo = Path(__file__).resolve().parents[1]
        for profile in ("deac.json", "generic-template.json"):
            self.assertEqual(execution_walltime_budget({
                "maximum_hours_per_cpu": 48, "submission_adapter": "slurm",
                "slurm_profile": str(repo / "profiles/slurm" / profile),
            })["maximum_estimated_execution_hours"], 36)

    def test_invalid_or_nonfinite_values_fail(self):
        for value in (True, -1, math.inf, math.nan, "11.75"):
            with self.assertRaises(ValueError):
                campaign_walltime_request(value, 48)
        with self.assertRaises(ValueError):
            campaign_walltime_request(1, 48, {"campaign_walltime_rounding_minutes": 0})

    def fixture(self, root, profile_name, cpus=44, memory=185):
        repo = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repo / "profiles/slurm" / profile_name)
        profile["environment"]["python_executable"] = None
        profile["paths"]["allowed_output_roots"] = []
        # Two serialized tasks demonstrate that kill-limit sums can exceed the
        # 16-hour allocation even though their buffered runtime is 11.75 hours.
        tasks = []
        for index in range(2):
            script = f"task{index}.slurm"
            (root / script).write_text("#!/usr/bin/env bash\nset -euo pipefail\n")
            tasks.append({
                "task_id": str(index), "script": script, "array_task_id": None,
                "depends_on_task_ids": ["0"] if index else [], "wait_for_task_ids": [],
                "cpu_slots": 1, "planned_wall_hours": 5.875,
                "planned_peak_memory_gib": 2, "requested_memory_gib": 4,
                "requested_wall_minutes": 1000, "planner_task_ids": [str(index)],
                "resource_request_source": "test_fixture",
                "wall_request_limited_by_campaign_cap": False,
                "memory_request_limited_by_campaign_cap": False,
            })
        (root / "submit.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n")
        plan = {"dependency_model": "task_dag_v1", "phases": [{"phase_id": "x", "tasks": tasks}],
                "maximum_parallel_cpus": cpus, "maximum_parallel_memory_gib": memory,
                "maximum_campaign_wall_hours": 48, "resource_policy": profile["resource_policy"]}
        return profile, plan

    def test_both_profiles_generate_sixteen_hour_single_allocation(self):
        for name in ("deac.json", "generic-template.json"):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                profile, plan = self.fixture(root, name)
                before = copy.deepcopy(plan)
                out = apply_slurm_profile(root, profile, plan)
                preview = out["submission_preview"]
                self.assertTrue(preview["submission_permitted"])
                self.assertEqual(preview["campaign_walltime_request"]["requested_wall_hours"], 16)
                self.assertGreater(preview["scheduler_time_limit_reservation_critical_path_hours"], 16)
                self.assertEqual(plan, before)  # no sampling or resource mutation
                script = (root / "run-campaign.slurm").read_text()
                self.assertIn("#SBATCH --time=16:00:00", script)
                self.assertIn("#SBATCH --mem=185G", script)
                self.assertIn("#SBATCH --cpus-per-task=44", script)
                self.assertIn("--maximum-wall-hours 16", script)
                self.assertEqual(out["tasks"][0]["requested_memory_gib"], 4)
                subprocess.run(["bash", "-n", str(root / "run-campaign.slurm")], check=True)
                subprocess.run(["bash", "-n", str(root / "submit.sh")], check=True)

    def test_single_allocation_refuses_multi_node_or_excess_node_resources(self):
        for cpus, memory, distributed in [(88, 185, False), (44, 370, False), (44, 185, True)]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                profile, plan = self.fixture(root, "deac.json", cpus, memory)
                if distributed:
                    plan["phases"][0]["tasks"][0].update({
                        "node_count": 2, "cpu_slots": 2, "workers_per_node": 1,
                        "distributed_worker_count": 2, "distributed_replica_execution": True,
                        "aggregate_requested_memory_gib": 8,
                    })
                out = apply_slurm_profile(root, profile, plan)
                self.assertFalse(out["single_allocation"]["submission_permitted"])
                self.assertIn('run "$ROOT" --single-allocation', (root / "submit.sh").read_text())

    def test_native_single_allocation_preview_and_mock_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile, plan = self.fixture(root, "generic-template.json")
            submit = root / "fake-sbatch"
            submit.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$(dirname "$0")/submitted.txt"\nprintf "12345\\n"\n')
            submit.chmod(0o755)
            profile["submit_command"] = str(submit)
            queue = root / "fake-squeue"
            queue.write_text("#!/bin/sh\nexit 0\n")
            queue.chmod(0o755)
            profile["status_command"] = str(queue)
            for index, t in enumerate(plan["phases"][0]["tasks"]):
                t["completion_reports"] = [f"report{index}.json"]
            (root / "analysis-config.json").write_text(json.dumps({"execution":{"submission_adapter":"slurm"}}))
            (root / "local-execution-plan.json").write_text(json.dumps(plan))
            (root / "slurm-profile.json").write_text(json.dumps({k:v for k,v in profile.items() if k != "source_path"}))
            apply_slurm_profile(root, profile, plan)
            result = subprocess.run(["bash", str(root / "submit.sh"), "--single-allocation", "--preview"],
                                    capture_output=True, text=True, check=True)
            displayed = json.loads(result.stdout)["slurm_preview"]
            self.assertEqual(displayed["single_allocation"]["requested_wall_hours"], 16)
            self.assertEqual(displayed["packaging_contract"]["strategy"], "single_allocation")
            self.assertEqual(displayed["packaging_contract"]["reserved_cpu_hours_upper_bound"], 44 * 16)
            self.assertFalse((root / "submitted.txt").exists())
            submitted = subprocess.run(["bash", str(root / "submit.sh"), "--single-allocation"],
                           capture_output=True, text=True)
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            args = (root / "submitted.txt").read_text().splitlines()
            self.assertEqual(args, ["--parsable", f"--chdir={root.resolve()}", str(root.resolve() / "run-campaign.slurm")])
            ledgers = list((root / "submission-ledgers").glob("*.tsv"))
            self.assertEqual(len(ledgers), 1)
            self.assertIn("campaign\t12345", ledgers[0].read_text())

    def test_infeasible_estimate_blocks_both_submission_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile, plan = self.fixture(root, "generic-template.json")
            plan["maximum_campaign_wall_hours"] = 10
            result = apply_slurm_profile(root, profile, plan)
            self.assertFalse(result["submission_preview"]["submission_permitted"])
            self.assertFalse(result["single_allocation"]["submission_permitted"])
            for args in ([], ["--single-allocation"]):
                submitted = subprocess.run(["bash", str(root / "submit.sh"), *args],
                                           capture_output=True, text=True)
                self.assertNotEqual(submitted.returncode, 0)
            self.assertFalse((root / "submission-ledgers").exists())

    def test_preview_rejects_runtime_that_fits_but_leaves_insufficient_allowance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile, plan = self.fixture(root, "generic-template.json")
            plan["maximum_campaign_wall_hours"] = 15
            result = apply_slurm_profile(root, profile, plan)
            self.assertFalse(result["submission_preview"]["submission_permitted"])
            codes = {r["code"] for r in result["submission_preview"]["warnings"]}
            self.assertIn("CAMPAIGN_PADDED_TIME_EXCEEDS_WALL_LIMIT", codes)
            self.assertFalse(result["single_allocation"]["submission_permitted"])

    def test_native_executor_records_shorter_deadline_without_changing_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {"local_execution_plan_schema": "salsbury-local-execution-plan-v6",
                    "dependency_model": "task_dag_v1", "phases": [],
                    "maximum_parallel_cpus": 1, "maximum_parallel_memory_gib": 4,
                    "maximum_campaign_wall_hours": 48}
            path = root / "local-execution-plan.json"
            path.write_text(json.dumps(plan))
            before = path.read_bytes()
            result = run_local_workflow(root, maximum_wall_hours=16)
            self.assertEqual(result["technical_status"], "complete")
            self.assertEqual(result["prepared_campaign_wall_hours"], 48)
            self.assertEqual(result["effective_campaign_wall_hours"], 16)
            self.assertEqual(path.read_bytes(), before)

    def test_local_allocation_override_cannot_extend_prepared_ceiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "local-execution-plan.json").write_text(json.dumps({
                "local_execution_plan_schema": "salsbury-local-execution-plan-v6",
                "maximum_parallel_cpus": 1, "maximum_parallel_memory_gib": 4,
                "maximum_campaign_wall_hours": 1,
            }))
            for value in (2, 0, math.nan):
                with self.assertRaises(ExecutionAdapterError):
                    run_local_workflow(root, maximum_wall_hours=value)
