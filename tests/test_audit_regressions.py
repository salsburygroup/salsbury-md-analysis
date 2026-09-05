"""Regressions from the September 2026 cross-branch audit."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from salsbury_md_analysis import convergence
from salsbury_md_analysis.execution_adapters import _reports_complete, run_local_workflow
from salsbury_md_analysis.finding_picker import _balanced_scientific_order
from salsbury_md_analysis.memory_policy import resolve_memory_uncertainty_policy
from salsbury_md_analysis.execution_resources import _measured_subprocess
from salsbury_md_analysis.manifests import sha256_file
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget
from salsbury_md_analysis.scientific_sampling import ScientificSamplingProfile, assess_raw_sampling


def replica_task(counts=(100, 100, 100), slots=2):
    return {
        "task_id": "qc", "module_id": "structural_integrity_qc", "dependency_stage": 0,
        "effective_cpu_cap": slots, "parallel_execution_model": "replica_worker_exact_global_reducer_v1",
        "parallel_worker_count": len(counts), "estimated_peak_memory_gib_per_parallel_worker": 1,
        "reducer_memory_gib": 1, "source_frames_per_replica": list(counts),
        "minimum_frames_per_replica": min(counts), "maximum_frames_per_replica": min(counts),
        "required_integer_stride": 1, "cpu_seconds_per_physical_frame": 36,
        "estimated_peak_memory_gib": len(counts),
    }


class AuditRegressions(unittest.TestCase):
    def test_new_project_paths_cannot_copy_an_upstream_project_checksum(self):
        import ast
        import salsbury_md_analysis
        root = Path(salsbury_md_analysis.__file__).parent
        checked = 0
        upstream_names = {"upstream", "pca", "clustering", "first", "dccm", "discovery", "tica_report"}
        for file in root.glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if not isinstance(node, ast.Dict):
                    continue
                fields = {k.value: v for k, v in zip(node.keys, node.values)
                          if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                path, digest = fields.get("project_manifest_path"), fields.get("project_manifest_sha256")
                if not isinstance(path, ast.Call) or digest is None:
                    continue
                checked += 1
                self.assertFalse(isinstance(digest, ast.Subscript) and
                    isinstance(digest.value, ast.Name) and digest.value.id in upstream_names,
                    f"{file.name}:{node.lineno} copies a checksum from a different project")
        self.assertGreater(checked, 10)

    def test_final_resource_summary_checks_its_output_files(self):
        from salsbury_md_analysis.accepted_artifacts import validate_complete_report, ArtifactValidationError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "final-resource-summary.json"
            payload = {"technical_status": "complete", "source_report_records": []}
            for kind in ("json", "csv", "markdown"):
                companion = root / ("table." + kind)
                companion.write_text("test fixture")
                payload[kind + "_path"] = str(companion)
                payload[kind + "_sha256"] = sha256_file(companion)
            report.write_text(json.dumps(payload))
            validate_complete_report(report)
            (root / "table.csv").write_text("changed")
            with self.assertRaises(ArtifactValidationError):
                validate_complete_report(report)

    def test_permission_denied_tree_inspection_does_not_abort_child(self):
        from types import SimpleNamespace
        from salsbury_md_analysis.execution_measurement import main as measure
        def denied(*args, **kwargs):
            raise PermissionError("process inspection is restricted")
        restricted = SimpleNamespace(Process=denied, Error=RuntimeError)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measured.json"
            with patch.dict(sys.modules, {"psutil": restricted}):
                result = measure([str(output), "--", sys.executable, "-c",
                                  "import time; time.sleep(.04)"])
            record = json.loads(output.read_text())
            self.assertEqual(result, 0)
            self.assertEqual(record["memory_measurement_scope"], "largest_child_only")
            self.assertFalse(record["memory_replacement_qualified"])

    def test_measurements_are_per_invocation_not_cumulative(self):
        cpu = "import time; end=time.process_time()+0.16\nwhile time.process_time()<end: pass"
        for concurrent in (False, True):
            def measure(code):
                result, metrics = _measured_subprocess([sys.executable, "-c", code])
                self.assertEqual(result.returncode, 0)
                self.assertFalse(metrics["memory_replacement_qualified"])
                self.assertIn(metrics["memory_measurement_scope"],
                              {"largest_child_only", "sampled_process_tree_and_largest_child"})
                return metrics["total_cpu_seconds"]
            if concurrent:
                with ThreadPoolExecutor(2) as workers:
                    busy, idle = list(workers.map(measure, [cpu, "pass"]))
            else:
                busy, idle = measure(cpu), measure("pass")
            self.assertGreater(busy, 0.15)
            self.assertLess(idle, busy / 2)

    def test_report_companion_hash_must_match_for_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            companion = root / "values.npy"
            companion.write_bytes(b"fixture")
            report = root / "report.json"
            report.write_text(json.dumps({"technical_status": "complete", "module_id": "fixture",
                "artifact": {"path": str(companion), "sha256": sha256_file(companion)}}))
            (root / "report.json.summary.json").write_text(json.dumps({
                "technical_status": "complete", "module_id": "fixture", "report_sha256": sha256_file(report)}))
            self.assertTrue(_reports_complete(root, ["report.json"]))
            companion.write_bytes(b"changed")
            self.assertFalse(_reports_complete(root, ["report.json"]))

    def test_node_reserve_is_once_for_two_colocated_tasks(self):
        tasks = [dict(replica_task(counts=(10,), slots=1), task_id=name) for name in ("a", "b")]
        plan = plan_campaign_resource_budget(tasks, maximum_parallel_cpus=2,
            maximum_wall_hours=2, maximum_memory_gib=4, memory_safety_factor=1.5,
            memory_overhead_gib=1, maximum_cpus_per_node=2, maximum_memory_gib_per_node=4,
            maximum_nodes=1, minimum_scheduler_memory_gib=0)
        # Whole-GiB task requests round 1.5 up to 2. Both do not fit with the
        # shared 1-GiB reserve and must be serialized, not under-reserved.
        self.assertEqual(plan["feasibility_status"], "feasible")
        for stage in plan["stages"]:
            for lane in stage["resource_lanes"]:
                self.assertLessEqual(lane["memory_gib"] + lane["campaign_node_memory_reserve_gib"], 4)

    def test_export_companions_use_declared_export_root_and_reject_missing_pdb(self):
        from salsbury_md_analysis.accepted_artifacts import validate_complete_report, ArtifactValidationError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = root / "separate-exports"
            exports.mkdir()
            structure = exports / "rep.pdb"
            structure.write_text("END\n")
            report = root / "report.json"
            report.write_text(json.dumps({"module_id": "state_coordinate_exports",
                "technical_status": "complete", "export_directory": str(exports),
                "outputs": [{"path": "rep.pdb", "sha256": sha256_file(structure)}]}))
            validate_complete_report(report)
            structure.unlink()
            with self.assertRaises(ArtifactValidationError):
                validate_complete_report(report)

    def test_indivisible_replica_waves_determine_feasibility(self):
        plan = plan_campaign_resource_budget(
            [replica_task()], maximum_parallel_cpus=2, maximum_wall_hours=1.6,
            maximum_memory_gib=8, planning_utilization=1, pilot_budget_fraction=0,
            finalization_headroom_fraction=0,
        )
        self.assertNotEqual(plan["feasibility_status"], "feasible")
        self.assertGreaterEqual(plan["estimated_selected_wall_hours_lower_bound"], 2)

    def test_wave_costs_include_uneven_replicas_and_serial_reduction(self):
        for counts, slots in (((100,) * 63, 44), ((100, 300, 200), 2),
                              ((50, 100, 150, 300), 3)):
            task = replica_task(counts, slots)
            task["fixed_cpu_hours"] = .4
            task["minimum_frames_per_replica"] = 1
            task["maximum_frames_per_replica"] = max(counts)
            plan = plan_campaign_resource_budget([task], maximum_parallel_cpus=slots,
                maximum_wall_hours=100, maximum_memory_gib=200,
                planning_utilization=1, pilot_budget_fraction=0,
                finalization_headroom_fraction=0)
            self.assertEqual(plan["feasibility_status"], "feasible")
            lanes = [0.] * slots
            for count in counts:
                index = min(range(slots), key=lambda i: (lanes[i], i))
                lanes[index] += count / 100
            self.assertAlmostEqual(plan["estimated_selected_wall_hours_lower_bound"],
                                   .4 + max(lanes), places=6)

    def test_unrelated_slow_task_does_not_hold_ready_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = {"a": "sleep 0.5\necho a >> order", "b": "echo b >> order", "c": "echo c >> order"}
            for name, body in scripts.items():
                (root / f"{name}.slurm").write_text("#!/bin/bash\nset -eu\n" + body + "\n")
            def task(name, deps=()):
                return {"task_id": name, "script": name + ".slurm", "cpu_slots": 1,
                        "requested_memory_gib": 1, "requested_wall_minutes": 1,
                        "depends_on_task_ids": list(deps), "wait_for_task_ids": []}
            plan = {"local_execution_plan_schema": "salsbury-local-execution-plan-v6",
                    "dependency_model": "task_dag_v1", "maximum_parallel_cpus": 2,
                    "maximum_parallel_memory_gib": 4, "maximum_campaign_wall_hours": .01,
                    "phases": [{"phase_id": "p0", "tasks": [task("a"), task("b")]},
                               {"phase_id": "p1", "tasks": [task("c", ["b"])]}]}
            (root / "local-execution-plan.json").write_text(json.dumps(plan))
            self.assertEqual(run_local_workflow(root)["technical_status"], "complete")
            self.assertEqual((root / "order").read_text().splitlines(), ["b", "c", "a"])

    def test_stale_error_report_cannot_authorize_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(json.dumps({"technical_status": "complete",
                "module_id": "wrong-module", "project_manifest_sha256": "0" * 64, "error_count": 3}))
            (root / "summary.json").write_text(json.dumps({"technical_status": "complete", "report_sha256": "1" * 64}))
            self.assertFalse(_reports_complete(root, ["report.json"]))

    def test_temporal_floor_uses_complete_intervals(self):
        profile = ScientificSamplingProfile("ordered", "ordered", 2, 2, 2.0, True, "uniform")
        assessment = assess_raw_sampling(profile, selected_frames_per_replica=[2],
            source_frames_per_replica=[5], integer_stride=2,
            frame_intervals_ns_per_replica=[1.0], source_time_spans_ns_per_replica=[4.0])
        self.assertTrue(assessment["keep_enabled"])

    def test_default_has_no_additional_uncertainty_multiplier(self):
        self.assertEqual(resolve_memory_uncertainty_policy({})["poorly_calibrated_factor"], 1)

    def test_current_convergence_contract_is_supported(self):
        settings = convergence._settings({"definitions": {"convergence_uncertainty": {
            "source_module": "replica_rmsd_rg", "metrics": ["rmsd_angstrom"],
            "block_size_frames": 2, "include_partial_final_block": False, "minimum_blocks": 2,
            "effective_sample_size_reference": 20, "split_mean_difference_reference_in_sd": 1}}})
        result = convergence._series_diagnostic([0, 1, 2, 3, 4, 5], settings)
        self.assertNotIn("passes_all_declared_series_gates", result)
        self.assertIn("effective_sample_size_reference_relation", result)

    def test_category_quota_does_not_promote_negligible_effect(self):
        rows = [
            {"statement": "tiny FES", "category": "free_energy_surface", "comparison_family": "fes",
             "system_ids": ["A", "B"], "absolute_effect_value": .000001,
             "statistically_significant": False, "adjusted_p_value": .9},
            {"statement": "large DCCM", "category": "coupled_interaction", "comparison_family": "dccm",
             "system_ids": ["A", "B"], "absolute_effect_value": 1.8,
             "statistically_significant": True, "adjusted_p_value": .000001},
        ]
        self.assertEqual(_balanced_scientific_order(rows)[0]["statement"], "large DCCM")
