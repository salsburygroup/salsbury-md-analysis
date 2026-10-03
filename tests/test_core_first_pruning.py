"""Synthetic planning regressions. These fixtures are not scientific results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.analysis_config import (
    AnalysisConfigError, default_analysis_config, make_resource_fit_config,
)
from salsbury_md_analysis.resource_planning import (
    PlanningSearchError, plan_campaign_resource_budget,
    plan_global_stride_projection_coupled_campaign_resource_budget,
    plan_projection_coupled_campaign_resource_budget,
    recommend_scientifically_valid_task_subset,
)
from salsbury_md_analysis.resource_fit_preparation import prepare_core_first_resource_fit
from salsbury_md_analysis.quickstart import (
    QuickstartPlanningError, prepare_standard_analysis,
    prepare_standard_analysis_resource_fit,
)
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis_resource_fit
from tests.test_quickstart import _write_inputs
import tests.test_resource_planning as resource_planning_fixtures


def task(name, module, hours=1.0):
    return {"task_id": name, "module_id": module, "dependency_stage": 0,
            "effective_cpu_cap": 1, "source_frames_per_replica": [1000],
            "minimum_frames_per_replica": 1000, "maximum_frames_per_replica": 1000,
            "cpu_seconds_per_physical_frame": 0.0, "fixed_cpu_hours": hours,
            "estimated_peak_memory_gib": 1.0, "priority_weight": 1.0,
            "balance_group": name}


LIMITS = dict(maximum_parallel_cpus=8, maximum_wall_hours=10.0,
              maximum_memory_gib=64.0, planning_utilization=1.0,
              pilot_budget_fraction=0.0)


class CoreFirstPruningTests(unittest.TestCase):
    def test_native_parallel_plateau_is_crossed_without_changing_sampling(self):
        tasks = [task(f"protected_{i}", "replica_rmsd_rg") for i in range(3)] + [
            task("ion", "ion_atmosphere", 20), task("sasa", "solvent_accessible_surface_area", 20)]
        before = deepcopy(tasks)
        report = recommend_scientifically_valid_task_subset(tasks, **LIMITS)
        self.assertEqual(tasks, before)
        self.assertEqual(report["recommendation_status"], "feasible_subset_found")
        self.assertEqual(len(report["decisions"]), 2)
        self.assertTrue(report["decisions"][0]["tied_bottleneck_step"])
        self.assertEqual(report["recommended_plan"]["feasibility_status"], "feasible")
        for row in report["recommended_plan"]["tasks"]:
            self.assertEqual(row["selected_physical_frames_per_replica"], [1000])
            self.assertEqual(row["integer_stride"], 1)

    def test_core_failure_prevents_optional_search(self):
        tasks = [task("qc", "structural_integrity_qc", 20), task("sasa", "solvent_accessible_surface_area")]
        with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", wraps=plan_campaign_resource_budget) as planner:
            report = recommend_scientifically_valid_task_subset(tasks, **LIMITS)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(report["recommendation_status"], "no_feasible_subset_found")
        self.assertEqual(report["configuration_patch"], {})
        self.assertIsNone(report["full_requested_plan"])

    def test_tied_slow_methods_removed_before_unrelated_cheap_method(self):
        tasks = [task(f"core{i}", "replica_rmsd_rg") for i in range(3)] + [
            task("slow1", "ion_atmosphere", 20),
            task("slow2", "hydrogen_bond_discovery", 20),
            task("cheap", "solvent_accessible_surface_area", 0.1)]
        report = recommend_scientifically_valid_task_subset(tasks, **LIMITS)
        self.assertIn("cheap", report["retained_task_ids"])
        self.assertNotIn("slow1", report["retained_task_ids"])
        self.assertNotIn("slow2", report["retained_task_ids"])

    def test_core_fit_does_not_drop_anything_when_full_scope_fits(self):
        tasks = [task("qc", "structural_integrity_qc"), task("sasa", "solvent_accessible_surface_area")]
        calls = []
        def observe(rows, **kwargs):
            calls.append([r["task_id"] for r in rows])
            return plan_campaign_resource_budget(rows, **kwargs)
        with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", side_effect=observe):
            report = recommend_scientifically_valid_task_subset(tasks, **LIMITS)
        self.assertEqual(calls, [["qc"], ["qc", "sasa"]])
        self.assertEqual(report["configuration_patch"], {})
        self.assertEqual(report["retained_task_ids"], ["qc", "sasa"])

    def test_coupling_error_retains_validated_core_and_diagnostic_history(self):
        tasks = [task("qc", "structural_integrity_qc"), task("sasa", "solvent_accessible_surface_area")]
        def fail_optional(rows, **kwargs):
            if len(rows) > 1:
                raise PlanningSearchError("stalled", diagnostics={"iteration_history": [1, 2]})
            return plan_campaign_resource_budget(rows, **kwargs)
        with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", side_effect=fail_optional):
            report = recommend_scientifically_valid_task_subset(tasks, **LIMITS)
        self.assertTrue(report["protected_core_fallback_used"])
        self.assertEqual(report["search_errors"][0]["diagnostics"]["iteration_history"], [1, 2])
        self.assertEqual(report["recommendation_status"], "feasible_subset_found")

    def test_core_search_failure_is_not_resource_infeasibility(self):
        with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", side_effect=PlanningSearchError("stalled", diagnostics={})):
            report = recommend_scientifically_valid_task_subset([task("qc", "structural_integrity_qc")], **LIMITS)
        self.assertEqual(report["recommendation_status"], "planning_search_failed")
        self.assertIsNone(report["recommended_plan"])
        self.assertEqual(report["configuration_patch"], {})

    def test_required_bundle_dependency_is_preserved(self):
        core = task("qc", "structural_integrity_qc")
        core["planning_dependencies"] = {"depends_on_bundle_ids": ["input"], "wait_for_bundle_ids": []}
        upstream = task("input", "optional_input", 20)
        upstream["planning_dependencies"] = {"depends_on_bundle_ids": [], "wait_for_bundle_ids": []}
        report = recommend_scientifically_valid_task_subset([core, upstream], **LIMITS)
        self.assertEqual(report["recommendation_status"], "no_feasible_subset_found")
        self.assertEqual(report["protected_task_ids"], ["input", "qc"])

    def test_optional_wait_does_not_become_a_required_producer(self):
        core = task("qc", "structural_integrity_qc")
        core["planning_dependencies"] = {"depends_on_bundle_ids": [], "wait_for_bundle_ids": ["optional"]}
        optional = task("optional", "ion_atmosphere", 20)
        optional["planning_dependencies"] = {"depends_on_bundle_ids": [], "wait_for_bundle_ids": []}
        report = recommend_scientifically_valid_task_subset([core, optional], **LIMITS)
        self.assertEqual(report["retained_task_ids"], ["qc"])
        self.assertEqual(report["protected_task_ids"], ["qc"])

    def test_trajectory_writing_removed_without_removing_representatives(self):
        export = task("export", "state_coordinate_exports", 20)
        export.update(workflow_id="system_A__global_common_heavy",
                      state_trajectory_exports_enabled=True,
                      representatives_only_fixed_cpu_hours=0.2,
                      maximum_representative_structures=250)
        report = recommend_scientifically_valid_task_subset([export], **LIMITS)
        self.assertEqual(report["configuration_patch"], {"views.global_common_heavy.state_trajectory_exports_enabled": False})
        self.assertEqual(report["retained_task_ids"], ["export"])
        self.assertEqual(report["recommended_plan"]["tasks"][0]["maximum_coordinate_writes"], 250)

    def test_configuration_preserves_comparison_and_representatives(self):
        config = default_analysis_config(["state_coordinate_exports", "common_pca", "pca_fes_basins", "integrated_comparison"], ["global_common_heavy"],
                                         protected_modules=["state_coordinate_exports", "common_pca", "pca_fes_basins", "integrated_comparison"])
        reduced, direct, _ = make_resource_fit_config(config, ["views.global_common_heavy.state_trajectory_exports_enabled"])
        self.assertTrue(reduced["modules"]["state_coordinate_exports"]["enabled"])
        self.assertFalse(reduced["views"]["global_common_heavy"]["state_trajectory_exports_enabled"])
        with self.assertRaises(AnalysisConfigError):
            make_resource_fit_config(config, ["modules.integrated_comparison.enabled"])

    def test_scope_costs_rebuilt_on_every_removal(self):
        tasks = [task("qc", "structural_integrity_qc"), task("sasa", "solvent_accessible_surface_area", 20), task("report", "workflow_final_reporting", 100)]
        def rebuild(rows, switches):
            rows = deepcopy(rows)
            for row in rows:
                if row["task_id"] == "report":
                    row["fixed_cpu_hours"] = len(rows) - 1
            return rows
        report = recommend_scientifically_valid_task_subset(tasks, rebuild_subset=rebuild, **LIMITS)
        self.assertEqual(report["recommendation_status"], "feasible_subset_found")
        self.assertEqual(next(row for row in report["recommended_plan"]["tasks"] if row["task_id"] == "report")["fixed_cpu_hours"], 1)

    def test_one_failed_cache_candidate_does_not_abort_other_candidates(self):
        tasks = resource_planning_fixtures.ResourcePlanningTests._cache_coupling_tasks()
        calls = []
        def candidate(rows, **kwargs):
            calls.append(rows)
            if len(calls) == 1:
                raise PlanningSearchError("stalled", diagnostics={"iteration_history": [1]})
            return plan_projection_coupled_campaign_resource_budget(rows, **kwargs)
        with patch("salsbury_md_analysis.resource_planning.plan_projection_coupled_campaign_resource_budget", side_effect=candidate):
            plan = plan_global_stride_projection_coupled_campaign_resource_budget(
                tasks, **LIMITS, overall_stride_candidate_strides=[1, 2], coordinate_cache_full_scan_fraction=0.0)
        self.assertEqual(plan["feasibility_status"], "feasible")
        self.assertEqual(plan["global_stride_coupling"]["selected_coordinate_cache_integer_stride"], 2)
        self.assertFalse(plan["global_stride_coupling"]["search_complete"])

    def test_all_failed_cache_searches_remain_unknown(self):
        with patch("salsbury_md_analysis.resource_planning.plan_projection_coupled_campaign_resource_budget", side_effect=PlanningSearchError("stalled", diagnostics={"iteration_history": [1]})):
            with self.assertRaises(PlanningSearchError) as caught:
                plan_global_stride_projection_coupled_campaign_resource_budget(
                    resource_planning_fixtures.ResourcePlanningTests._cache_coupling_tasks(), **LIMITS,
                    overall_stride_candidate_strides=[1, 2])
        self.assertEqual(len(caught.exception.diagnostics["candidate_search_failures"]), 2)

    def test_real_iteration_limit_retains_failed_history(self):
        parent = task("pca", "common_pca")
        parent.update(workflow_id="shared", task_scope="conformational_view")
        child = task("pam", "alternative_clustering")
        child.update(workflow_id="shared", task_scope="conformational_view_algorithm_fit",
                     algorithm_id="pam", source_frames_per_replica=[100],
                     minimum_frames_per_replica=10, maximum_frames_per_replica=100)
        with self.assertRaises(PlanningSearchError) as caught:
            plan_projection_coupled_campaign_resource_budget([parent, child], **LIMITS,
                maximum_coupling_iterations=1)
        self.assertEqual(len(caught.exception.diagnostics["iteration_history"]), 1)
        self.assertEqual(caught.exception.diagnostics["maximum_coupling_iterations"], 1)

    def test_terminal_does_not_label_search_failure_resource_infeasibility(self):
        from salsbury_md_analysis.cli import _planning_failure_outcome, _campaign_plan_terminal_summary
        self.assertEqual(_planning_failure_outcome({"planning_search_failed": True}),
                         ("planning_search_failed", "PLANNING_SEARCH_FAILED"))
        self.assertEqual(_planning_failure_outcome({"protected_core_rejected": True}),
                         ("no_acceptable_reduced_plan", "NO_ACCEPTABLE_REDUCED_PLAN"))
        summary = _campaign_plan_terminal_summary({"method_reduction_recommendation": {
            "protected_core_plan": {"minimum_wall_hours_lower_bound": 3},
            "recommended_plan": {"minimum_wall_hours_lower_bound": 8}}})
        self.assertEqual(summary["protected_subset_minimum_critical_path_hours"], 3)

    def test_native_preparation_falls_back_after_full_search_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdb, psf, *trajectories]}
            calls = []
            def prepare(**kwargs):
                calls.append(kwargs["output_directory"].name)
                if kwargs["output_directory"].name == "requested":
                    core = json.loads((root / "out.resource-fit-evidence/protected-core.json").read_text())
                    # Simulate only the search exception; core and final
                    # preparations run the real parser, planner and adapter.
                    config = deepcopy(core["config"])
                    config["planning"]["module_selection"] = "configured"
                    config["modules"]["solvent_accessible_surface_area"]["enabled"] = True
                    raise QuickstartPlanningError("coupling timeout", plan={"planning_search_failed": True}, analysis_config=config, output_directory=kwargs["output_directory"])
                return prepare_standard_analysis(**kwargs)
            report = prepare_core_first_resource_fit(
                prepare=prepare, common=dict(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                project_id="core-first-test", frame_interval_ps=10.0),
                destination=root / "out", target_wall_hours=32.0, config_path=None)
            self.assertEqual(calls, ["protected-core", "requested", "out"])
            fit = report["resource_fit"]
            self.assertTrue(fit["protected_core_fallback_used"])
            self.assertEqual(fit["final_feasibility_status"], "feasible")
            self.assertIn("modules.solvent_accessible_surface_area.enabled", fit["directly_disabled_configuration_switches"])
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
            self.assertTrue((root / "out/resource-fit-evidence/requested.json").is_file())

    def test_native_core_rejection_emits_no_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            config = root / "config.json"
            config.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1", "execution": {"maximum_memory_gib": 0.01}}))
            with self.assertRaisesRegex(QuickstartPlanningError, "No acceptable reduced plan"):
                prepare_standard_analysis_resource_fit(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                    project_id="no-core", frame_interval_ps=10.0, output_directory=root / "out", config_path=config)
            self.assertFalse((root / "out/submit.sh").exists())
            self.assertFalse((root / "out.resource-fit-evidence/requested.json").exists())
            self.assertTrue((root / "out.resource-fit-evidence/protected-core.json").is_file())

    def test_fixed_schedule_is_preserved_in_opt_in_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            common = dict(pdb_path=pdb, psf_path=psf, trajectories=trajectories,
                          project_id="fixed-test", frame_interval_ps=10.0)
            prepare_standard_analysis(**common, output_directory=root / "original")
            config = json.loads((root / "original/analysis-config.json").read_text())
            frozen = root / "original/fixed-sampling-schedule.json"
            config["sampling"]["fixed_schedule_file"] = str(frozen)
            config_path = root / "fixed.json"
            config_path.write_text(json.dumps(config))
            result = prepare_standard_analysis_resource_fit(**common, config_path=config_path,
                output_directory=root / "fixed")
            self.assertFalse(result["resource_fit"]["automatic_changes_applied"])
            self.assertFalse(result["resource_fit"]["protected_core_checked_first"])
            before = json.loads((root / "original/campaign-resource-plan.json").read_text())
            after = json.loads((root / "fixed/campaign-resource-plan.json").read_text())
            def sampling(plan):
                return {r["task_id"]: (r["integer_stride"], r["selected_physical_frames_per_replica"])
                        for r in plan["tasks"]}
            self.assertEqual(sampling(before), sampling(after))

    def test_five_system_preparation_reprices_reports_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            request = root / "systems.json"
            request.write_text(json.dumps({"request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [{"system_id": f"system{i}", "pdb": str(pdb), "psf": str(psf),
                             "trajectories": [str(p) for p in trajectories], "frame_interval_ps": 10.0}
                            for i in range(5)]}))
            config = root / "config.json"
            config.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1",
                "reporting": {"finding_picker_enabled": False},
                "execution": {"maximum_memory_gib": 4.0,
                              "well_calibrated_memory_uncertainty_factor": 2.0,
                              "poorly_calibrated_memory_uncertainty_factor": 2.0}}))
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdb, psf, *trajectories, request, config]}
            output = root / "five-system"
            result = prepare_comparative_analysis_resource_fit(request_path=request, config_path=config,
                output_directory=output, project_id="synthetic-five", target_wall_hours=32.0)
            self.assertTrue(result["resource_fit"]["protected_core_checked_first"])
            self.assertTrue(result["resource_fit"]["automatic_changes_applied"])
            plan = json.loads((output / "campaign-resource-plan.json").read_text())
            self.assertEqual(plan["feasibility_status"], "feasible")
            self.assertEqual(plan["native_schedule_validation"]["status"], "complete")
            original = json.loads((output / "campaign-resource-plan.requested.json").read_text())
            core = json.loads((output / "resource-fit-evidence/protected-core.json").read_text())["plan"]
            recommended_core = original["method_reduction_recommendation"]["protected_core_plan"]
            for module in ("workflow_final_reporting", "workflow_integrated_reporting"):
                def row(document):
                    return next(t for t in document["tasks"] if t["module_id"] == module)
                self.assertLess(row(recommended_core)["fixed_cpu_hours"], row(original)["fixed_cpu_hours"])
                self.assertEqual(row(recommended_core)["fixed_cpu_hours"], row(core)["fixed_cpu_hours"])
                self.assertEqual(row(recommended_core)["estimated_peak_memory_gib"], row(core)["estimated_peak_memory_gib"])
            resolved = json.loads((output / "analysis-config.resource-fit.json").read_text())
            self.assertTrue(resolved["modules"]["integrated_comparison"]["enabled"])
            self.assertTrue(resolved["modules"]["state_coordinate_exports"]["enabled"])
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})


if __name__ == "__main__":
    unittest.main()
