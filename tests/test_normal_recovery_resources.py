import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from salsbury_md_analysis.imwkmeans_resources import (
    REFERENCE_SETTINGS, runtime_model, runtime_workload, validate_held_out,
    work_per_fit_observation,
)
from salsbury_md_analysis.campaign_planning import _apply_measured_resource_calibrations
from salsbury_md_analysis.execution_adapters import _run_ready_dag, _ResourcePool, _slurm_resource_epochs, _run_local_task
from salsbury_md_analysis.resource_calibrations import (
    _aggregate_calibrations, qualify_calibration, _entry_from_timeout,
    TIMEOUT_SCHEMA, ResourceCalibrationError,
)
from salsbury_md_analysis.memory_workload import memory_workload, report_memory_workload


class NormalRecoveryResourceTests(unittest.TestCase):
    def task(self, name, memory, hours, deps=()):
        return dict(task_id=name, module_id="test", script=name + ".sh",
                    cpu_slots=1, requested_memory_gib=memory, planned_wall_hours=hours,
                    requested_wall_minutes=hours * 60, depends_on_task_ids=list(deps))

    def test_local_replays_planned_reservations_and_resource_failure_does_not_block_science(self):
        tasks = [self.task("initial", 3, 3), self.task("large", 183, 4),
                 self.task("later", 3, 2, ("initial",))]
        phases = [dict(phase_id="p", tasks=tasks)]
        epochs = _slurm_resource_epochs(dict(phases=phases, maximum_parallel_cpus=44,
            maximum_parallel_memory_gib=184), {}, {}, {}, {"large_memory_threshold_gib": float("inf")}, {})
        scheduled = epochs[0]["scheduled_items"]
        self.assertEqual([r["task_id"] for r in scheduled], ["initial", "large", "later"])
        self.assertIn("large", scheduled[2]["resource_predecessor_task_ids"])
        order = []
        before = copy.deepcopy(tasks)
        def run(root, task, *args):
            order.append(task["task_id"])
            return dict(task, status="failed" if task["task_id"] == "large" else "complete")
        with tempfile.TemporaryDirectory() as directory, patch(
                "salsbury_md_analysis.execution_adapters._run_local_task", side_effect=run):
            result = _run_ready_dag(Path(directory), phases, "test", time.monotonic() + 10,
                                    _ResourcePool(44, 184), False, 1)
        self.assertEqual(order, ["initial", "large", "later"])
        self.assertEqual(result[0]["tasks"][-1]["status"], "complete")
        self.assertEqual(tasks, before)

    def test_independent_task_continues_while_real_dependency_fails(self):
        tasks = [self.task("bad", 1, 1), self.task("child", 1, 1, ("bad",)), self.task("independent", 1, 1)]
        with tempfile.TemporaryDirectory() as directory, patch(
                "salsbury_md_analysis.execution_adapters._run_local_task",
                side_effect=lambda root, task, *args: dict(task, status="failed" if task["task_id"] == "bad" else "complete")):
            result = _run_ready_dag(Path(directory), [dict(phase_id="p", tasks=tasks)], "test",
                time.monotonic() + 10, _ResourcePool(2, 2), False, 1)
        states = {r["task_id"]: r["status"] for r in result[0]["tasks"]}
        self.assertEqual(states, dict(bad="failed", child="skipped_dependency", independent="complete"))

    def test_never_dispatched_is_not_timeout_or_calibration(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "salsbury_md_analysis.execution_adapters._run_local_task") as worker:
            result = _run_ready_dag(Path(directory), [dict(phase_id="p", tasks=[self.task("a", 1, 1)])],
                "test", time.monotonic() - 1, _ResourcePool(1, 2), False, 1)
            worker.assert_not_called()
        row = result[0]["tasks"][0]
        self.assertEqual(row["status"], "not_started_deadline")
        self.assertEqual(row["wall_seconds"], 0)
        self.assertFalse(row["runtime_calibration_eligible"])

    def test_deadline_between_dispatch_and_launch_and_real_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "a.sh").write_text("sleep 2\n")
            task = self.task("a", 1, 1)
            with patch("salsbury_md_analysis.execution_adapters.validate_worker_projects"):
                row = _run_local_task(root, task, "p", 0, "notstarted", time.monotonic() - 1,
                                      _ResourcePool(1, 2), False, 1)
                self.assertEqual(row["status"], "not_started_deadline")
                self.assertFalse(row["execution_started"])
                task["requested_wall_minutes"] = .002
                row = _run_local_task(root, task, "p", 0, "timeout", time.monotonic() + 5,
                                      _ResourcePool(1, 2), False, 1)
                self.assertEqual(row["status"], "timed_out")
                self.assertTrue(row["execution_started"])
                self.assertGreater(row["wall_seconds"], 0)

    def test_never_started_record_cannot_enter_timeout_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(dict(evidence_schema=TIMEOUT_SCHEMA, technical_status="timeout",
                execution_started=False, elapsed_seconds=99999)))
            with self.assertRaisesRegex(ResourceCalibrationError, "executed timeout"):
                _entry_from_timeout(path)

    def test_reused_result_is_not_a_new_runtime_measurement(self):
        task = dict(self.task("a", 1, 1), completion_reports=["report.json"])
        with patch("salsbury_md_analysis.execution_adapters._reports_complete", return_value=True):
            row = _run_local_task(Path("unused"), task, "p", 0, "reuse", time.monotonic() + 1,
                                  _ResourcePool(1, 2), False, 1)
        self.assertEqual(row["status"], "reused_complete")
        self.assertFalse(row["execution_started"])
        self.assertFalse(row["runtime_calibration_eligible"])

    def test_work_model_responds_to_each_grid_dimension_and_member_count(self):
        settings = copy.deepcopy(REFERENCE_SETTINGS)
        original = work_per_fit_observation(settings)
        for key, value in (("component_indices", [1, 2, 3, 4]), ("k_values", list(range(2, 15))),
                           ("minkowski_p_values", [1.5, 2, 3, 4]), ("initialization_ranks", [0, 1, 2, 3, 4]),
                           ("maximum_iterations", 1000), ("maximum_silhouette_observations", 2000)):
            self.assertGreater(work_per_fit_observation(dict(settings, **{key: value})), original, key)
        self.assertGreater(work_per_fit_observation(dict(settings, minkowski_p_values=[1.5])),
                           work_per_fit_observation(dict(settings, minkowski_p_values=[2.0])))
        tasks = [dict(module_id="clustering_imwkmeans", imwkmeans_settings=settings,
                      member_observation_multiplier=n) for n in (1, 2)]
        _apply_measured_resource_calibrations(tasks, {}, time_safety_factor=1.5)
        self.assertAlmostEqual(tasks[0]["cpu_seconds_per_physical_frame"], .75)
        self.assertAlmostEqual(tasks[1]["cpu_seconds_per_physical_frame"], 1.5)
        self.assertEqual(tasks[0]["fixed_cpu_hours"], tasks[1]["fixed_cpu_hours"])
        feature_settings = dict(settings)
        feature_settings.pop("component_indices")
        feature_settings["trajectory_feature_columns"] = [dict(feature_id="x", value_indices=[1, 2, 3])]
        self.assertEqual(work_per_fit_observation(feature_settings), original)
        runtime_model(feature_settings, [dict(wall_seconds=10,
            imwkmeans_workload=runtime_workload(feature_settings, 100, 200))])

    def test_censored_workload_envelope_and_independent_heldout(self):
        settings = copy.deepcopy(REFERENCE_SETTINGS)
        def point(n, seconds, timeout=False):
            return dict(imwkmeans_workload=runtime_workload(settings, n, n * 2),
                **({"wall_seconds_lower_bound": seconds, "evidence_status": "right_censored_timeout"}
                   if timeout else {"wall_seconds": seconds, "evidence_status": "complete_execution"}))
        training = [point(1000, 500), point(10000, 7000), point(20000, 15000, True)]
        before = copy.deepcopy(training)
        model = runtime_model(settings, training)
        self.assertEqual(model["censored_measurement_count"], 1)
        self.assertEqual(validate_held_out(model, [point(5000, 5000)])["status"], "complete")
        self.assertEqual(validate_held_out(model, [point(5000, 50000)])["status"], "failed")
        self.assertEqual(training, before)
        self.assertEqual(runtime_model(settings, [dict(wall_seconds=9999999)])["qualified_measurement_count"], 0)
        for record in (dict(point(10, 10), execution_started=False), point(10, 0), point(10, float("nan"))):
            with self.assertRaises(ValueError):
                validate_held_out(model, [record])
        identified = dict(point(1000, 500), source_sidecar_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "used to fit"):
            validate_held_out(runtime_model(settings, [identified]), [identified])

    def test_memory_workload_emission_preserves_unknown_axes(self):
        module = "solvent_accessible_surface_area"
        report = dict(module_id=module, settings=dict(sphere_point_count=960, output_detail="bounded_summary_v1"),
                      replicas=[dict(source_atom_count=100), dict(source_atom_count=200)],
                      replica_execution=dict(workers_used=2))
        target = memory_workload(module, report["settings"], maximum_atom_count=200,
                                 selected_frames=1000, workers=2)
        self.assertEqual(report_memory_workload(report, 1000), target)
        report.pop("replica_execution")
        self.assertEqual(report_memory_workload(report, 1000), {})
        self.assertIsNone(memory_workload(module, {}, maximum_atom_count=1,
                                         selected_frames=1, workers=1)["sphere_point_count"])

    def test_task_memory_evidence_needs_scope_and_workload_coverage(self):
        for module in ("solvent_accessible_surface_area", "water_mediated_hydrogen_bond_networks"):
            target = dict(implementation_id="impl", maximum_atom_count=1000, selected_frames=4000, concurrent_workers=4)
            context = dict(coordinate_source="raw_source", task_scope="direct_trajectory_estimator")
            rows = [dict(module_id=module, resource_context=context, memory_workload=target,
                total_cpu_seconds=20, wall_seconds=10, selected_source_physical_frames=4000,
                symmetry_expanded_observations=4000, cpu_seconds_per_selected_physical_frame=.005,
                maximum_resident_memory_mib=1024, memory_replacement_qualified=True,
                measurement_scope="one fresh child process for one analysis command",
                memory_measurement_scope="validated_simultaneous_peak") for _ in range(2)]
            def qualify(records, work=target):
                catalog = _aggregate_calibrations({module: records}, "path", "x" * 64, {}, 1.5)[module]
                return qualify_calibration(catalog, context, memory_workload=work)[0]
            self.assertTrue(qualify(rows)["memory_replacement_qualified"])
            for key in ("maximum_atom_count", "selected_frames", "concurrent_workers"):
                self.assertFalse(qualify(rows, dict(target, **{key: target[key]*2}))["memory_replacement_qualified"])
            unscoped = [dict(row, measurement_scope="Slurm batch MaxRSS", maximum_resident_memory_mib=120000) for row in rows]
            result = qualify(unscoped)
            self.assertFalse(result["memory_replacement_qualified"])
            self.assertEqual(result["maximum_resident_memory_mib"], 0)
            self.assertEqual(result["maximum_observed_resident_memory_mib_all_records"], 120000)
            self.assertFalse(qualify([dict(row, memory_workload={}) for row in rows])["memory_replacement_qualified"])
            # Never promote RSS sampling (a lower bound) to a qualified peak.
            self.assertFalse(qualify([dict(row, memory_replacement_qualified=False) for row in rows])["memory_replacement_qualified"])
            for scope in ("largest_child_only", "sampled_process_tree_and_largest_child"):
                self.assertFalse(qualify([dict(row, memory_measurement_scope=scope) for row in rows])["memory_replacement_qualified"])


if __name__ == "__main__":
    unittest.main()
