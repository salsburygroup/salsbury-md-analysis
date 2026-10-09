"""Submission tests: frozen science, independent jobs, explicit limits, replay."""
import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from salsbury_md_analysis.execution_adapters import load_slurm_profile, apply_slurm_profile
from salsbury_md_analysis.slurm_packaging import packaging_schedule, package_execution_plan, validate_packaging, check_single_allocation
from salsbury_md_analysis.user_workflow import execute_workflow, _check_submission_history


def task(name, cpus=1, memory=64, requires=(), waits=()):
    return {"task_id": name, "script": name + ".slurm", "array_task_id": None,
        "cpu_slots": cpus, "requested_memory_gib": memory,
        "requested_wall_minutes": 30, "planned_wall_hours": .25,
        "planned_peak_memory_gib": memory / 1.5, "resource_request_source": "test",
        "wall_request_limited_by_campaign_cap": False, "memory_request_limited_by_campaign_cap": False,
        "depends_on_task_ids": list(requires), "wait_for_task_ids": list(waits),
        "completion_reports": [name + "/report.json"]}


def plan():
    return {"dependency_model": "task_dag_v1", "maximum_parallel_cpus": 4,
        "maximum_parallel_memory_gib": 185, "maximum_campaign_wall_hours": 8,
        "node_policy": {"cpus_per_node": 44, "memory_gib_per_node": 185},
        "node_memory_reserve_gib": 1, "autorecovery": True, "maximum_task_attempts": 2,
        "phases": [{"phase_id": "work", "tasks": [task(f"export{i}") for i in range(6)]},
            {"phase_id": "report", "tasks": [task("final", memory=36, waits=[f"export{i}" for i in range(6)])]}]}


def profile():
    return load_slurm_profile(Path(__file__).resolve().parents[1] / "profiles/slurm/generic-template.json")


class PackagingTests(unittest.TestCase):
    def test_independent_allocations_exceed_plan_envelope_without_changing_it(self):
        original = plan()
        before = json.dumps(original, sort_keys=True)
        packaged, epochs, preview = packaging_schedule(original, profile())
        self.assertEqual(before, json.dumps(original, sort_keys=True))
        self.assertEqual(packaged["phases"], original["phases"])
        self.assertEqual(preview["resource_token_edge_count"], 0)
        self.assertEqual(preview["completion_wait_edge_count"], 6)
        self.assertEqual(preview["maximum_parallel_cpus_in_generated_waves"], 6)
        self.assertEqual(preview["maximum_parallel_memory_gib_in_generated_waves"], 384)
        self.assertIsNone(preview["planned_node_count"])
        self.assertTrue(preview["submission_permitted"])
        self.assertEqual([row["memory_gib_per_node"] for row in preview["packaging_contract"]["tasks"]], [64]*6 + [36])

    def test_explicit_limits_add_only_resource_edges(self):
        p = profile()
        p["packaging"] = {"maximum_concurrent_cpus": 2, "maximum_concurrent_memory_gib": 128}
        _, epochs, preview = packaging_schedule(plan(), p)
        self.assertLessEqual(preview["maximum_parallel_cpus_in_generated_waves"], 2)
        self.assertLessEqual(preview["maximum_parallel_memory_gib_in_generated_waves"], 128)
        self.assertGreater(preview["resource_token_edge_count"], 0)
        rows = [t for epoch in epochs for t in epoch["scheduled_items"]]
        self.assertTrue(all(not row["depends_on_task_ids"] for row in rows))
        self.assertEqual(len(rows[-1]["wait_for_task_ids"]), 6)

    def test_fail_closed_without_splitting_or_reestimating_indivisible_task(self):
        p = profile()
        for limits in ({"maximum_job_memory_gib": 63}, {"maximum_concurrent_memory_gib": 63},
                       {"maximum_reserved_cpu_hours": 6}):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "ceiling"):
                package_execution_plan(plan(), {**p, "packaging": limits})
        original = plan()
        original["phases"][0]["tasks"][0]["requested_memory_gib"] = 184.01
        with self.assertRaisesRegex(ValueError, "node reserve"):
            packaging_schedule(original, p)

    def test_memory_rounding_and_attempts_count_actual_reservations(self):
        original = plan()
        original["phases"][0]["tasks"] = [task("a", memory=63.1), task("b", memory=63.1)]
        original["phases"] = original["phases"][:1]
        p = profile()
        p["packaging"] = {"maximum_concurrent_memory_gib": 127}
        _, _, preview = packaging_schedule(original, p)
        self.assertEqual(preview["maximum_parallel_memory_gib_in_generated_waves"], 64)
        self.assertEqual(preview["packaging_contract"]["reserved_cpu_hours_upper_bound"], 2)

    def test_distributed_worker_group_not_reexpanded_and_no_implicit_node_count_cap(self):
        original = plan()
        original.update(maximum_parallel_cpus=60, maximum_parallel_memory_gib=370)
        original["node_policy"]["maximum_nodes_per_campaign"] = 2
        group = task("cache", 60, 100)
        group.update(node_count=2, workers_per_node=30, distributed_worker_count=60,
                     distributed_replica_execution=True)
        original["phases"][0]["tasks"] = [group, {**group, "task_id": "cache2", "script": "cache2.slurm"}]
        original["phases"] = original["phases"][:1]
        _, _, preview = packaging_schedule(original, profile())
        self.assertEqual(preview["maximum_parallel_cpus_in_generated_waves"], 120)
        self.assertEqual(preview["maximum_parallel_memory_gib_in_generated_waves"], 400)
        self.assertEqual(preview["resource_token_edge_count"], 0)

    def test_invalid_settings_are_not_silently_ignored(self):
        for value in ({"unknown": 1}, {"maximum_job_cpus": True}, {"maximum_job_cpus": 1.5},
                      {"maximum_concurrent_memory_gib": float("inf")}, {"maximum_reserved_cpu_hours": 0}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_packaging(value)

    def test_single_allocation_limits_use_actual_allocation(self):
        single = {"submission_permitted":True, "cpus":4, "memory_gib":185, "requested_wall_hours":8}
        for limits in ({"maximum_job_cpus":1}, {"maximum_job_memory_gib":64},
                       {"maximum_reserved_cpu_hours":7}):
            checked = check_single_allocation(single, {"packaging":limits})
            self.assertFalse(checked["submission_permitted"])
            self.assertEqual(checked["reserved_cpu_hours_upper_bound"], 32)
        self.assertTrue(check_single_allocation(single, {"packaging":{"maximum_reserved_cpu_hours":32}})["submission_permitted"])

    def test_profile_and_guarded_launcher_preserve_tasks_and_bound_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = plan()
            before = copy.deepcopy(original)
            for phase in original["phases"]:
                for t in phase["tasks"]:
                    (root / t["script"]).write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"\nexport CUDA_VISIBLE_DEVICES=0\n')
            (root / "submit.sh").write_text("#!/bin/bash\n")
            result = apply_slurm_profile(root, profile(), original)
            self.assertEqual(original, before)
            self.assertIn('run "$ROOT"', (root / "submit.sh").read_text())
            self.assertIn('package-slurm "$ROOT"', (root / "submit.sh").read_text())
            worker = (root / "export0.slurm").read_text()
            self.assertIn("export OMP_NUM_THREADS=1", worker)
            self.assertNotIn('OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"', worker)
            self.assertIn("export CUDA_VISIBLE_DEVICES=0", worker)
            self.assertEqual(result["submission_preview"]["resource_token_edge_count"], 0)
            self.assertEqual(subprocess.run(["bash", "-n", str(root / "submit.sh")]).returncode, 0)


class SubmissionTests(unittest.TestCase):
    def fixture(self, root):
        original = plan()
        for phase in original["phases"]:
            for t in phase["tasks"]:
                (root / t["script"]).write_text("#!/bin/bash\nset -euo pipefail\n")
        for name, payload in (("analysis-config.json", {"execution":{"submission_adapter":"slurm"}}),
            ("local-execution-plan.json", original), ("slurm-profile.json", {k:v for k,v in profile().items() if k != "source_path"}),
            ("slurm-submission-preview.json", {"generated_schedule_feasibility_status":"feasible"})):
            (root / name).write_text(json.dumps(payload))
        return {"technical_status":"incomplete", "counts":{"complete":5,"not_complete":2},
            "tasks":[{"task_id":t["task_id"], "state":"complete" if t["task_id"] not in {"export5", "final"} else "not_complete"}
                     for p in original["phases"] for t in p["tasks"]]}

    def test_preview_reuses_accepted_tasks_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            status = self.fixture(root)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            with patch("salsbury_md_analysis.user_workflow.workflow_status", return_value=status), patch(
                "salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout="")) as run:
                result = execute_workflow(root, execute=False)
            self.assertEqual(run.call_count, 1)  # queue query only
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
            self.assertFalse(result["jobs_submitted"])
            self.assertEqual(result["slurm_preview"]["task_count"], 2)
            self.assertEqual(len(result["slurm_preview"]["reused_task_ids"]), 5)
            self.assertEqual(result["slurm_preview"]["completion_wait_edge_count"], 1)

    def test_real_status_preview_does_not_create_lock_or_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch("salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout="")):
                result = execute_workflow(root, execute=False)
            self.assertEqual(result["slurm_preview"]["task_count"], 7)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_partial_submission_is_durable_and_never_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            status = self.fixture(root)
            with patch("salsbury_md_analysis.user_workflow.workflow_status", return_value=status), patch(
                "salsbury_md_analysis.user_workflow.subprocess.run", side_effect=[SimpleNamespace(stdout=""), SimpleNamespace(returncode=1)]):
                result = execute_workflow(root)
            self.assertEqual(result["jobs_submitted"], "possibly_partial_check_submission_ledger")
            intent = next((root / "submission-intents").glob("*.json"))
            self.assertEqual(json.loads(intent.with_suffix(".result").read_text())["status"], "uncertain")
            with self.assertRaisesRegex(ValueError, "Unresolved submission intent"):
                _check_submission_history(root, {"export5"}, profile())

    def test_accounting_required_before_retry_even_with_empty_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "submission-ledgers").mkdir()
            (root / "submission-ledgers/first.tsv").write_text("task_id\tjob_id\na\t123\n")
            for state in ("", "123|RUNNING\n", "123|COMPLETED\n", "123_0|REQUEUED\n"):
                with self.subTest(state=state), patch("salsbury_md_analysis.user_workflow.subprocess.run",
                    return_value=SimpleNamespace(stdout=state)), self.assertRaises(ValueError):
                    _check_submission_history(root, {"a"}, profile())
            with patch("salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout="123_0|TIMEOUT\n")):
                _check_submission_history(root, {"a"}, profile())

    def test_live_work_and_single_allocation_recovery_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            status = self.fixture(root)
            with patch("salsbury_md_analysis.user_workflow.workflow_status", return_value=status), patch(
                "salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout=f"123|PENDING|{root}\n")), self.assertRaisesRegex(ValueError, "still has Slurm job"):
                execute_workflow(root)
            with patch("salsbury_md_analysis.user_workflow.workflow_status", return_value=status), patch(
                "salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout="")), self.assertRaisesRegex(ValueError, "original envelope"):
                execute_workflow(root, single_allocation=True)


if __name__ == "__main__":
    unittest.main()
