import copy
import json
import tempfile
import threading
import time
import unittest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.execution_adapters import (
    ExecutionAdapterError, _ResourcePool, _run_ready_dag, validate_worker_projects,
)
from salsbury_md_analysis.observation_guards import validate_projection_guards, validate_project_observation_guards
from salsbury_md_analysis.task_status import persist_task_status, latest_task_statuses, record_slurm_event
from salsbury_md_analysis.user_workflow import workflow_status
from salsbury_md_analysis.structural_qc import structural_qc_project, structural_qc_project_safe
from test_structural_qc import _write_project


class TaskRepairTests(unittest.TestCase):
    def setup_status(self, root):
        task = {"task_id": "a", "module_id": "grouped_ml", "script": "a.sh",
                "cpu_slots": 1, "completion_reports": ["results/a/report.json"]}
        (root / "local-execution-plan.json").write_text(json.dumps({"phases": [{"tasks": [task]}]}))
        return task

    def failure(self, root, module="grouped_ml", content=None):
        output = root / "results/a/report.json.tmp.old-attempt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content if content is not None else json.dumps({
            "module_id": module, "technical_status": "failed",
            "issues": [{"severity": "error", "message": "maximum_observations gate exceeded"}],
        }))
        return output

    def status(self, root, complete=False):
        with patch("salsbury_md_analysis.user_workflow.campaign_activity", return_value={
            "slurm_jobs": [{"task_ids": ["a"], "state": "RUNNING"}]}), patch(
                "salsbury_md_analysis.accepted_artifacts.reports_complete", return_value=complete):
            return workflow_status(root)["tasks"][0]

    def test_failure_visible_inside_running_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_status(root)
            self.failure(root)
            row = self.status(root)
            self.assertEqual(row["state"], "failed")
            self.assertEqual(row["allocation_state"], "RUNNING")
            self.assertFalse(row["failure_evidence"][0]["acceptance"])
            persist_task_status(root, task, "run1", {"status": "timed_out"})
            self.assertEqual(self.status(root)["state"], "timed_out")
            self.assertEqual(self.status(root, complete=True)["state"], "complete")

    def test_partial_unrelated_and_stale_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_status(root)
            self.failure(root, content='{"technical_status":')
            self.assertEqual(self.status(root)["state"], "running")
            self.failure(root, module="other")
            self.assertEqual(self.status(root)["state"], "running")
            self.failure(root)
            persist_task_status(root, task, "run2", {"status": "running"})
            self.assertEqual(self.status(root)["state"], "running")
            persist_task_status(root, task, "run2", {"status": "complete"})
            self.assertEqual(self.status(root, complete=True)["state"], "complete")
            changed = {**task, "cpu_slots": 2}
            self.assertEqual(latest_task_statuses(root, [changed]), {})
            status_dir = root / "local-execution-status/task-attempts"
            (status_dir / "interrupted.tmp").write_text("{")
            self.assertEqual(latest_task_statuses(root, [task])["a"]["status"], "complete")

    def test_failed_worker_persisted_before_independent_worker_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [{"task_id": key, "script": key + ".sh", "cpu_slots": 1,
                      "requested_memory_gib": 1} for key in ("bad", "healthy")]
            observed = []
            def worker(root, task, *args):
                if task["task_id"] == "healthy":
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        records = latest_task_statuses(root, tasks)
                        if records.get("bad", {}).get("status") == "failed":
                            observed.append(True)
                            break
                        threading.Event().wait(0.01)
                    return {**task, "status": "complete"}
                return {**task, "status": "failed", "exit_code": 2}
            with patch("salsbury_md_analysis.execution_adapters._run_local_task", side_effect=worker):
                reports = _run_ready_dag(root, [{"phase_id": "p", "tasks": tasks}], "run1",
                                         time.monotonic() + 10, _ResourcePool(2, 4), False, 1)
            self.assertEqual(observed, [True])
            self.assertEqual([r["status"] for r in reports[0]["tasks"]], ["failed", "complete"])

    def test_population_guard_reports_actual_count_without_mutation(self):
        for module, field in (("grouped_ml", "maximum_observations"), ("representative_frames", "maximum_candidates")):
            project = {"project_id": "pooled", "definitions": {module: {field: 100000}}}
            before = copy.deepcopy(project)
            with self.assertRaisesRegex(ValueError, "200000 selected pooled observations.*100000"):
                validate_projection_guards(project, 200000)
            self.assertEqual(project, before)
            validate_projection_guards(project, 100000)

    def test_exact_worker_path_rejects_double_prefix_and_accepts_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "with spaces.json"
            project.write_text("{}")
            script = root / "worker.sh"
            task = {"script": script.name, "array_task_id": 0, "project_filename": str(project)}
            plan = {"phases": [{"tasks": [task]}]}
            for value in (str(project), project.name):
                script.write_text('PROJECTS=(\n' + json.dumps(value) + '\n)\nPROJECT="${PROJECTS[$SLURM_ARRAY_TASK_ID]}"\n')
                validate_worker_projects(root, plan)
            script.write_text('PROJECTS=(\n' + json.dumps(str(project)) + '\n)\nPROJECT="$ROOT/${PROJECTS[$SLURM_ARRAY_TASK_ID]}"\n')
            with self.assertRaisesRegex(ExecutionAdapterError, "project argument differs from its plan"):
                validate_worker_projects(root, plan)

    def test_projection_not_basis_count_and_members_and_disabled_consumers(self):
        from test_pca import _write_project as pca_project
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = pca_project(root, common=True)
            project = json.loads(path.read_text())
            project["requested_modules"] = ["common_pca", "grouped_ml"]
            pca = project["definitions"]["common_pca"]
            pca["frame_stride"] = 2
            pca["projection_frame_stride"] = 1
            project["definitions"]["grouped_ml"] = {"maximum_observations": 5}
            with self.assertRaisesRegex(ValueError, "6 selected pooled observations"):
                validate_project_observation_guards(project, path)
            project["definitions"]["grouped_ml"]["maximum_observations"] = 6
            self.assertEqual(validate_project_observation_guards(project, path), 6)
            pca["symmetry_expansion"] = {"member_count": 2}
            with self.assertRaisesRegex(ValueError, "12 selected pooled observations"):
                validate_project_observation_guards(project, path)
            project["requested_modules"] = ["common_pca"]
            self.assertIsNone(validate_project_observation_guards(project, path))

    def test_tica_guard_respects_method_specific_segment_omission(self):
        from test_pca import _write_project as pca_project
        with tempfile.TemporaryDirectory() as tmp:
            path = pca_project(Path(tmp), common=True)
            project = json.loads(path.read_text())
            project["requested_modules"].append("grouped_ml")
            definitions = project["definitions"]
            definitions["grouped_ml"] = {"maximum_observations": 4}
            definitions["clustering_kmeans"] = {"feature_source": "tica"}
            definitions["time_lagged_independent_component_analysis"] = {
                "short_segment_policy": "omit", "lag_frames": 1, "minimum_pairs_per_segment": 2}
            self.assertEqual(validate_project_observation_guards(project, path), 6)
            definitions["grouped_ml"]["maximum_observations"] = 3
            with self.assertRaisesRegex(ValueError, "4 selected pooled observations"):
                validate_project_observation_guards(project, path)

    def test_preparation_checks_unmaterialized_cache_against_final_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "project-view.json"
            path.write_text(json.dumps({"system_manifest": "future-cache/system.json", "definitions": {
                "common_pca": {}, "grouped_ml": {"maximum_observations": 100000}}}))
            (root / "worker.sh").write_text('PROJECT="project-view.json"\n')
            plan = {"phases": [{"tasks": [{"script": "worker.sh"}]}]}
            (root / "campaign-resource-plan.json").write_text(json.dumps({"tasks": [{
                "task_id": "view:view:common_pca", "selected_physical_frames_per_replica": [10000] * 20}]}))
            with self.assertRaisesRegex(ExecutionAdapterError, "200000 selected pooled observations"):
                validate_worker_projects(root, plan)
            before = path.read_bytes()
            with self.assertRaises(ExecutionAdapterError):
                validate_worker_projects(root, plan)
            self.assertEqual(path.read_bytes(), before)

    def test_allocation_preserves_explicit_consumer_limits(self):
        from test_pca import _write_project as pca_project
        from salsbury_md_analysis.campaign_planning import _apply_view_allocation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = pca_project(root, common=True)
            project = json.loads(source.read_text())
            project["definitions"]["grouped_ml"] = {"maximum_observations": 6}
            project["requested_modules"].append("grouped_ml")
            path = root / "project-view.json"
            path.write_text(json.dumps(project))
            _apply_view_allocation(path, {"view:view:common_pca": {
                "selected_physical_frames_per_replica": [2, 4],
                "frame_selection": {"mode": "fixed_stride_v1"}, "integer_stride": 1,
            }}, [2, 4], target_wall_hours=24,
                explicit_guard_options={"grouped_ml": {"maximum_observations": 3}})
            final = json.loads(path.read_text())
            self.assertEqual(final["definitions"]["grouped_ml"]["maximum_observations"], 3)
            with self.assertRaisesRegex(ValueError, "6 selected pooled observations"):
                validate_project_observation_guards(final, path)

    def test_changed_selection_requires_replanning_not_only_larger_guard(self):
        from test_pca import _write_project as pca_project
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = pca_project(root, common=True)
            project = json.loads(source.read_text())
            project["definitions"]["grouped_ml"] = {"maximum_observations": 100}
            project["requested_modules"].append("grouped_ml")
            (root / "project-view.json").write_text(json.dumps(project))
            (root / "worker.sh").write_text('PROJECT="project-view.json"\n')
            (root / "campaign-resource-plan.json").write_text(json.dumps({"tasks": [{
                "task_id": "view:view:common_pca", "selected_physical_frames_per_replica": [1, 2]}]}))
            with self.assertRaisesRegex(ExecutionAdapterError, "current selection gives 6.*resource plan costs 3"):
                validate_worker_projects(root, {"phases": [{"tasks": [{"script": "worker.sh"}]}]})

    def test_slurm_receipts_bind_array_element_and_preserve_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_status(root)
            (root / "a.sh").write_text("exit 0\n")
            with patch.dict("os.environ", {"SLURM_ARRAY_TASK_ID": "7"}):
                with self.assertRaisesRegex(ValueError, "cannot bind"):
                    record_slurm_event(root, "a.sh", "failed", "1", "2")
            with patch.dict("os.environ", {}, clear=True):
                record_slurm_event(root, "a.sh", "failed", "1", "2")
                record_slurm_event(root, "a.sh", "started", "2", "0")
                self.assertEqual(latest_task_statuses(root, [task])["a"]["status"], "running")
                record_slurm_event(root, "a.sh", "complete", "2", "0")
            self.assertEqual(len(list((root / "local-execution-status/task-attempts").glob("*.json"))), 3)

    def test_generated_slurm_wrapper_records_failed_and_recovered_attempts(self):
        from salsbury_md_analysis.execution_adapters import _render_task_recovery_runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_status(root)
            script = root / "a.sh"
            script.write_text('cd "$(dirname "$0")"\nif [[ ! -e retried ]]; then touch retried; exit 2; fi\nexit 0\n')
            wrapper = root / "wrapper.sh"
            wrapper.write_text(_render_task_recovery_runner(autorecovery=True, maximum_task_attempts=2))
            env = {**os.environ, "SLURM_JOB_ID": "synthetic-no-submission"}
            env.pop("SLURM_ARRAY_TASK_ID", None)
            result = subprocess.run(["bash", str(wrapper), str(script)], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipts = [json.loads(p.read_text()) for p in (root / "local-execution-status/task-attempts").glob("*.json")]
            self.assertEqual(sorted(row["status"] for row in receipts), ["complete", "failed", "running", "running"])
            self.assertEqual(latest_task_statuses(root, [task])["a"]["status"], "complete")

    def test_failed_receipt_does_not_allow_overwriting_invalid_final(self):
        from salsbury_md_analysis.user_workflow import _execute_workflow
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_status(root)
            temporary = self.failure(root)
            temporary.with_name("report.json").write_text("{}")
            persist_task_status(root, task, "run1", {"status": "failed"})
            row = self.status(root)
            self.assertEqual(row["state"], "failed")
            self.assertEqual(row["artifact_state"], "invalid_output")
            with patch("salsbury_md_analysis.user_workflow.campaign_activity", return_value={"slurm_jobs": []}):
                with self.assertRaisesRegex(ValueError, "automatic overwrite is forbidden"):
                    _execute_workflow(root)

    def test_unsampled_segment_remains_unevaluated_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root, "1\nf0\nC 0 0 0\n1\nf1\nC 1 0 0\n1\nf2\nC 2 0 0\n", 1)
            (root / "second.xyz").write_text("1\nf3\nC 3 0 0\n")
            system_path = root / "system.json"
            system = json.loads(system_path.read_text())
            system["systems"][0]["replicas"][0]["segments"].append({
                "segment_id": "segment-2", "trajectory": "second.xyz", "continuous_with_previous": False,
                "timing": {"first_frame_time": 3, "frame_interval": 1, "unit": "ps"}})
            system_path.write_text(json.dumps(system))
            data = json.loads(project.read_text())
            data["definitions"]["structural_qc"]["frame_selection"] = {"mode": "integer_stride_per_replica_v1", "stride": 2}
            project.write_text(json.dumps(data))
            report = structural_qc_project(project)
            self.assertEqual(report["technical_status"], "complete", report["issues"])
            segments = report["systems"][0]["replicas"][0]["segments"]
            self.assertEqual([s["evaluated_frame_count"] for s in segments], [2, 0])
            self.assertEqual([s["observed_frame_count"] for s in segments], [3, 1])
            self.assertEqual([i["severity"] for i in report["issues"] if i["code"] == "NO_QC_FRAMES_SELECTED"], ["warning"])

    def test_empty_source_and_missing_requested_frames_remain_errors(self):
        for empty in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                project = _write_project(Path(tmp), "" if empty else "1\nf0\nC 0 0 0\n", 1)
                with patch("salsbury_md_analysis.structural_qc.iter_coordinate_frames", return_value=iter(())):
                    report = structural_qc_project_safe(project)
                self.assertNotEqual(report["technical_status"], "complete")
                self.assertTrue(any(i["severity"] == "error" for i in report["issues"]))

    def test_unsampled_truncated_dcd_remains_error(self):
        from test_coordinates import _write_coordinate_dcd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root, "2\nf0\nC 0 0 0\nC 1 0 0\n", 2)
            dcd = root / "trajectory.dcd"
            _write_coordinate_dcd(dcd)
            dcd.write_bytes(dcd.read_bytes()[:-4])
            system_path = root / "system.json"
            system = json.loads(system_path.read_text())
            system["systems"][0]["replicas"][0]["segments"][0]["trajectory"] = "trajectory.dcd"
            system_path.write_text(json.dumps(system))
            data = json.loads(project.read_text())
            data["definitions"]["structural_qc"]["frame_selection"] = {"mode": "integer_stride_per_replica_v1", "stride": 10}
            project.write_text(json.dumps(data))
            report = structural_qc_project_safe(project)
            self.assertNotEqual(report["technical_status"], "complete")
