import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.orchestration_resources import manifest_workload, orchestration_tasks
from salsbury_md_analysis.execution_adapters import _task_planner_rows, ExecutionAdapterError
from salsbury_md_analysis.resource_planning import (
    plan_campaign_resource_budget,
    plan_global_stride_projection_coupled_campaign_resource_budget,
    recommend_scientifically_valid_task_subset,
)
from salsbury_md_analysis.quickstart import prepare_standard_analysis
from salsbury_md_analysis.planning_report import build_planning_report, render_planning_report_markdown
from tests.test_quickstart import _write_inputs


class OrchestrationResourceTests(unittest.TestCase):
    def fixture(self, root):
        (root / "top.pdb").write_bytes(b"x" * 10)
        (root / "traj.dcd").write_bytes(b"x" * 100)
        replica = {"topology": "top.pdb", "segments": [{"trajectory": "traj.dcd"}]}
        (root / "system.json").write_text(json.dumps({
            "systems": [{"replicas": [replica, replica]}]}))
        (root / "view.json").write_text((root / "system.json").read_text())
        (root / "project-v.json").write_text(json.dumps({"system_manifest": "view.json"}))
        return orchestration_tasks(root, [root / "project-v.json"], [], {},
                                   maximum_atom_count=423, time_safety_factor=1.5)

    def test_file_reads_and_one_time_margin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            workload = manifest_workload(root / "system.json")
            self.assertEqual(workload["input_bytes_read"], 220)
            self.assertEqual(workload["input_file_reads"], 4)
            row = rows[0]
            self.assertAlmostEqual(row["fixed_cpu_hours"] * 3600,
                                   row["resource_model"]["estimated_unpadded_wall_seconds"] * 1.5)
            self.assertEqual(len(rows), 3)
            self.assertFalse(row["resource_model"]["independently_validated"])
            self.assertEqual(row["work_unit"], "job_invocation")

    def test_estimates_scale_and_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            small = self.fixture(root)
            with (root / "traj.dcd").open("r+b") as f:
                f.truncate(1024**3)
            large = orchestration_tasks(root, [], [{"dependency_stage": 4}] * 100,
                                        {}, maximum_atom_count=100000, time_safety_factor=1.5)
            self.assertGreater(large[0]["fixed_cpu_hours"], small[0]["fixed_cpu_hours"])
            self.assertGreater(large[-1]["estimated_peak_memory_gib"], small[-1]["estimated_peak_memory_gib"])
            (root / "traj.dcd").unlink()
            with self.assertRaises(FileNotFoundError):
                manifest_workload(root / "system.json")

    def test_adapter_mapping_and_missing_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = self.fixture(root)
            for row in rows:
                self.assertEqual(_task_planner_rows(root, {"script": row["execution_script"]}, rows), [row])
            with self.assertRaisesRegex(ExecutionAdapterError, "missing orchestration"):
                _task_planner_rows(root, {"script": "run_preflight.slurm"}, rows[1:])

    def test_overhead_not_strided_scored_or_auto_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            overhead = self.fixture(root)
            cache = dict(overhead[0], task_id="cache", module_id="coordinate_cache",
                         task_scope="cache", source_frames_per_replica=[100],
                         minimum_frames_per_replica=10, maximum_frames_per_replica=100,
                         dependency_stage=1, required_integer_stride=None)
            science = dict(cache, task_id="science", module_id="test_science",
                           task_scope="direct_trajectory_estimator", dependency_stage=3)
            kwargs = dict(maximum_parallel_cpus=2, maximum_wall_hours=24,
                          maximum_memory_gib=32, planning_utilization=0.85,
                          pilot_budget_fraction=0.05)
            plan = plan_global_stride_projection_coupled_campaign_resource_budget(
                [cache, science, *overhead], overall_stride_candidate_strides=[2],
                coordinate_cache_minimum_frames_per_replica=10, **kwargs)
            for row in plan["tasks"]:
                if row["task_scope"] == "orchestration_overhead":
                    self.assertEqual(row["selected_physical_frames_per_replica"], [1])
                    self.assertNotIn("effective_raw_integer_stride", row)
            too_small = plan_campaign_resource_budget(overhead, **dict(kwargs, maximum_memory_gib=0.1))
            self.assertFalse(too_small["execution_authorized"])
            self.assertEqual(too_small["memory_feasibility"]["configuration_switches_to_disable_to_fit_configured_memory"], [])
            reduced = recommend_scientifically_valid_task_subset(overhead, **dict(kwargs, maximum_wall_hours=0.0001))
            self.assertEqual(reduced["recommendation_status"], "no_feasible_subset_found")
            self.assertEqual(reduced["disabled_configuration_switches"], [])
            self.assertEqual(reduced["attempted_configuration_switches"], [])
            self.assertEqual(len(reduced["protected_task_ids"]), 3)

    @patch("salsbury_md_analysis.quickstart._discover_dssp_executable", return_value=None)
    def test_fresh_generated_plan_maps_overhead_and_reports_no_fake_frames(self, _):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs = root / "inputs"
            inputs.mkdir()
            pdb, psf, trajectories = _write_inputs(inputs)
            output = root / "analysis"
            prepare_standard_analysis(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                                      output_directory=output, project_id="overhead-test", frame_interval_ps=10)
            resources = json.loads((output / "campaign-resource-plan.json").read_text())
            execution = json.loads((output / "local-execution-plan.json").read_text())
            generated = [t for p in execution["phases"] for t in p["tasks"]]
            for task in generated:
                if "preflight" in task["script"] or "finalize_reporting" in task["script"]:
                    self.assertTrue(task["planner_task_ids"], task["script"])
                    self.assertLess(task["planned_wall_hours"], 0.5)
            report = build_planning_report(output)
            self.assertEqual(report["orchestration"]["task_count"], 3)
            self.assertEqual(report["sampling"]["task_count"], len(resources["tasks"]) - 3)
            self.assertIn("Required execution overhead", render_planning_report_markdown(report))


if __name__ == "__main__":
    unittest.main()
