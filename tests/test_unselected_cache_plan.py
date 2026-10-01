"""Rejected cache searches must remain diagnostics, not sampling schedules."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis import cli
from salsbury_md_analysis.campaign_planning import (
    plan_global_stride_projection_coupled_campaign_resource_budget,
)
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from salsbury_md_analysis.quickstart import QuickstartPlanningError, prepare_standard_analysis
from tests.test_quickstart import _write_inputs, _write_dcd


WALL_REASON = "minimum calibrated critical path exceeds the campaign science wall-time budget"


class UnselectedCachePlanTests(unittest.TestCase):
    def fixture(self, root, fail_on_minimum=True):
        pdb, psf, trajectories = _write_inputs(root)
        for path in trajectories:
            _write_dcd(path, 4, 1_000)
        config = root / "config.json"
        config.write_text(json.dumps({
            "config_schema": "salsbury-analysis-config-v1",
            "planning": {"module_selection": "protected_core_only"},
            "clustering": {"feature_space": "common_pca"},
            "execution": {
                "maximum_parallel_cpus": 4, "maximum_memory_gib": 185,
                "overall_stride_candidates": [1],
                "coordinate_cache_materialization": "planned_strided",
                "fail_if_minimum_coverage_unaffordable": fail_on_minimum,
            },
        }))
        request = root / "request.json"
        request.write_text(json.dumps({
            "request_schema": "salsbury-comparative-analysis-input-v1",
            "systems": [{"system_id": name, "pdb": str(pdb), "psf": str(psf),
                         "trajectories": [str(p) for p in trajectories],
                         "frame_interval_ps": 10} for name in ("control", "variant")],
        }))
        return pdb, psf, trajectories, config, request

    def rejected_search(self, *args, **kwargs):
        # Keep real task identities and allocations. Simulate the documented
        # no-candidate return from the global search, not a malformed input.
        result = plan_global_stride_projection_coupled_campaign_resource_budget(*args, **kwargs)
        self.assertEqual(result["feasibility_status"], "feasible")
        result["feasibility_status"] = "infeasible"
        result["infeasibility_reasons"] = [WALL_REASON]
        coupling = {
            "converged": False,
            "selected_coordinate_cache_integer_stride": None,
            "selected_overall_trajectory_integer_stride": None,
            "candidate_evaluations": [{"coordinate_cache_integer_stride": 1,
                                       "feasibility_status": "infeasible"}],
            "reason": "no candidate satisfies the complete campaign envelope",
        }
        result["global_stride_coupling"] = coupling
        result["coordinate_cache_coupling"] = copy.deepcopy(coupling)
        return result

    def test_unselected_search_stops_before_snapshot_even_when_minimum_failure_disabled(self):
        for comparative in (False, True):
            for fail_on_minimum in (False, True):
                with self.subTest(comparative=comparative, fail_on_minimum=fail_on_minimum), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    pdb, psf, trajectories, config, request = self.fixture(root, fail_on_minimum)
                    output = root / "prepared"
                    with patch("salsbury_md_analysis.campaign_planning.plan_global_stride_projection_coupled_campaign_resource_budget", side_effect=self.rejected_search), patch(
                        "salsbury_md_analysis.campaign_planning.recommend_scientifically_valid_task_subset",
                        return_value={"recommendation_status": "no_feasible_subset_found"},
                    ) as recommendation, patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None):
                        with self.assertRaisesRegex(QuickstartPlanningError, "No acceptable reduced plan") as caught:
                            if comparative:
                                prepare_comparative_analysis(request_path=request, config_path=config,
                                    output_directory=output, project_id="rejected", target_wall_hours=168)
                            else:
                                prepare_standard_analysis(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                                    config_path=config, output_directory=output, project_id="rejected",
                                    frame_interval_ps=10, target_wall_hours=168)
                    recommendation.assert_called_once()
                    self.assertTrue(recommendation.call_args.kwargs["use_global_stride_coupling"])
                    plan = caught.exception.plan
                    self.assertEqual(plan["feasibility_status"], "infeasible")
                    self.assertIn(WALL_REASON, plan["infeasibility_reasons"])
                    self.assertFalse(plan["execution_authorized"])
                    self.assertIsNone(plan["global_stride_coupling"]["selected_coordinate_cache_integer_stride"])
                    self.assertEqual(len(plan["global_stride_coupling"]["candidate_evaluations"]), 1)
                    self.assertTrue((output / "campaign-resource-plan.json").exists())
                    for filename in ("fixed-sampling-schedule.json", "submit.sh", "run-local.sh"):
                        self.assertFalse((output / filename).exists(), filename)

    def test_cli_retains_reduction_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, _, config, request = self.fixture(root)
            stdout = io.StringIO()
            with patch("salsbury_md_analysis.campaign_planning.plan_global_stride_projection_coupled_campaign_resource_budget", side_effect=self.rejected_search), patch(
                "salsbury_md_analysis.campaign_planning.recommend_scientifically_valid_task_subset",
                return_value={"recommendation_status": "no_feasible_subset_found"},
            ), patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None), redirect_stdout(stdout):
                code = cli.main(["prepare-comparison", str(request), "--config", str(config),
                                 "--output", str(root / "prepared"), "--plan-only",
                                 "--project-id", "rejected",
                                 "--target-wall-hours", "168"])
            report = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(report["issues"][0]["code"], "NO_ACCEPTABLE_REDUCED_PLAN")
            self.assertEqual(report["planning_outcome"], "no_acceptable_reduced_plan")
            self.assertFalse(report["execution_started"])
            self.assertFalse(report["jobs_submitted"])
            self.assertEqual(len(report["campaign_resource_plan"]["global_stride_coupling"]["candidate_evaluations"]), 1)

    def test_concrete_selected_cache_schedule_can_reach_native_validation(self):
        def conservative_stage(*args, **kwargs):
            plan = plan_global_stride_projection_coupled_campaign_resource_budget(*args, **kwargs)
            self.assertEqual(plan["feasibility_status"], "feasible")
            plan["feasibility_status"] = "infeasible"
            plan["infeasibility_reasons"] = [WALL_REASON]
            return plan

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdb, psf, trajectories, config, _ = self.fixture(root)
            output = root / "prepared"
            with patch("salsbury_md_analysis.campaign_planning.plan_global_stride_projection_coupled_campaign_resource_budget", side_effect=conservative_stage), patch(
                "salsbury_md_analysis.campaign_planning.recommend_scientifically_valid_task_subset"
            ) as reduction:
                prepare_standard_analysis(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                    config_path=config, output_directory=output, project_id="selected",
                    frame_interval_ps=10, target_wall_hours=168)
            reduction.assert_not_called()
            plan = json.loads((output / "campaign-resource-plan.json").read_text())
            self.assertEqual(plan["feasibility_status"], "feasible")
            self.assertEqual(plan["native_schedule_validation"]["status"], "complete")
            frozen = json.loads((output / "fixed-sampling-schedule.json").read_text())
            self.assertEqual(frozen["global_stride_coupling"]["selected_coordinate_cache_integer_stride"], 1)

    def test_incomplete_or_invalid_selection_cannot_be_exported_as_feasible(self):
        for defect in ("missing_coupling", "not_converged", None, True, 0, 1.5):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                pdb, psf, trajectories, config, _ = self.fixture(root, fail_on_minimum=False)
                def invalid_selection(*args, **kwargs):
                    plan = plan_global_stride_projection_coupled_campaign_resource_budget(*args, **kwargs)
                    self.assertEqual(plan["feasibility_status"], "feasible")
                    if defect == "missing_coupling":
                        del plan["global_stride_coupling"]
                    elif defect == "not_converged":
                        plan["global_stride_coupling"]["converged"] = False
                    else:
                        plan["global_stride_coupling"]["selected_coordinate_cache_integer_stride"] = defect
                    return plan
                output = root / "prepared"
                with patch("salsbury_md_analysis.campaign_planning.plan_global_stride_projection_coupled_campaign_resource_budget", side_effect=invalid_selection), patch(
                    "salsbury_md_analysis.campaign_planning.recommend_scientifically_valid_task_subset",
                    return_value={"recommendation_status": "no_feasible_subset_found"},
                ), self.assertRaisesRegex(QuickstartPlanningError, "no valid coordinate-cache sampling schedule was selected"):
                    prepare_standard_analysis(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                        config_path=config, output_directory=output, project_id="invalid",
                        frame_interval_ps=10, target_wall_hours=168)
                self.assertFalse((output / "fixed-sampling-schedule.json").exists())


if __name__ == "__main__":
    unittest.main()
