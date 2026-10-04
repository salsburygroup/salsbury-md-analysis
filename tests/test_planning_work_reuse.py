"""Synthetic reuse, minimum-feasibility, and pruning-work regressions."""
from copy import deepcopy
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis import planning_reuse
from salsbury_md_analysis.planning_reuse import memoized_planning, reuse_planning_work
from salsbury_md_analysis.resource_planning import (
    PlanningSearchError, plan_campaign_resource_budget,
    plan_projection_coupled_campaign_resource_budget,
    plan_global_stride_projection_coupled_campaign_resource_budget,
    recommend_scientifically_valid_task_subset,
)
from tests.test_core_first_pruning import task, LIMITS
import tests.test_resource_planning as resource_fixtures


class PlanningReuseTests(unittest.TestCase):
    def test_native_preparation_reuses_work_and_preserves_inputs(self):
        from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
        from salsbury_md_analysis.quickstart import prepare_standard_analysis_resource_fit
        from tests.test_quickstart import _write_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb, psf, trajectories = _write_inputs(root)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            paths = [pdb, psf, *trajectories]
            before = [digest(path) for path in paths]
            with redirect_stderr(io.StringIO()), planning_diagnostics(root / "out", command="test") as record:
                result = prepare_standard_analysis_resource_fit(pdb_path=pdb, psf_path=psf,
                    trajectories=trajectories, output_directory=root / "out", project_id="test",
                    frame_interval_ps=10.0)
            events = [json.loads(line) for line in (record.directory / "events.jsonl").read_text().splitlines()]
            reuse = [row for row in events if row["event"] == "planning_work_reuse_finished"]
            self.assertEqual(len(reuse), 1)
            self.assertGreater(reuse[0]["hits"], 0)
            self.assertEqual(result["resource_fit"]["final_feasibility_status"], "feasible")
            self.assertEqual(before, [digest(path) for path in paths])

    def test_exact_inputs_defaults_and_mutation_isolation(self):
        calls = []
        @memoized_planning
        def calculation(rows, *, limit=3):
            calls.append((deepcopy(rows), limit))
            return {"rows": deepcopy(rows), "limit": limit}
        @reuse_planning_work
        def prepare():
            first = calculation([{"value": 2}])
            first["rows"][0]["value"] = 99
            self.assertEqual(calculation([{"value": 2}], limit=3)["rows"], [{"value": 2}])
            calculation([{"value": 2}], limit=4)
            calculation([{"value": 3}])
        prepare()
        self.assertEqual(len(calls), 3)
        prepare()
        self.assertEqual(len(calls), 6)  # no cross-preparation cache
        self.assertIsNone(planning_reuse._active.get())

    def test_exceptions_never_cached_and_context_released(self):
        calls = []
        @memoized_planning
        def broken(value):
            calls.append(value)
            raise ValueError("broken")
        @reuse_planning_work
        def prepare():
            for _ in range(2):
                with self.assertRaises(ValueError):
                    broken(1)
            raise RuntimeError("exit")
        with self.assertRaises(RuntimeError):
            prepare()
        self.assertEqual(calls, [1, 1])
        self.assertIsNone(planning_reuse._active.get())

    def test_cache_bounds_and_uncacheable_objects(self):
        @memoized_planning
        def calculate(value):
            return {"value": value}
        @reuse_planning_work
        def prepare():
            for value in range(10):
                calculate(value)
            self.assertLessEqual(len(planning_reuse._active.get()["entries"]), 2)
            self.assertLessEqual(planning_reuse._active.get()["bytes"], 30)
            self.assertEqual(calculate({1, 2})["value"], {1, 2})
        with patch.object(planning_reuse, "_MAX_ENTRIES", 2), patch.object(planning_reuse, "_MAX_SERIALIZED_BYTES", 30):
            prepare()

    def test_real_plan_identical_on_cache_hit_and_invalidates_all_settings(self):
        rows = [task("qc", "structural_integrity_qc")]
        @reuse_planning_work
        def prepare():
            expected = plan_campaign_resource_budget.__wrapped__(rows, **LIMITS)
            first = plan_campaign_resource_budget(rows, **LIMITS)
            second = plan_campaign_resource_budget(rows, **LIMITS)
            self.assertEqual(first, expected)
            self.assertEqual(second, expected)
            self.assertEqual(planning_reuse._active.get()["hits"], 1)
            for field, value in (("maximum_wall_hours", 11), ("maximum_memory_gib", 32),
                                 ("memory_safety_factor", 1.5), ("memory_overhead_gib", 1)):
                plan_campaign_resource_budget(rows, **{**LIMITS, field: value})
            rows[0]["source_frames_per_replica"] = [2000]
            plan_campaign_resource_budget(rows, **LIMITS)
            self.assertEqual(planning_reuse._active.get()["misses"], 6)
        prepare()

    def test_already_computed_full_scope_is_reused_by_recommendation(self):
        rows = [task("qc", "structural_integrity_qc"), task("sasa", "solvent_accessible_surface_area", 20)]
        @reuse_planning_work
        def prepare():
            plan_campaign_resource_budget(rows, **LIMITS)
            before = planning_reuse._active.get()["hits"]
            result = recommend_scientifically_valid_task_subset(rows, **LIMITS)
            self.assertGreater(planning_reuse._active.get()["hits"], before)
            self.assertEqual(result["retained_task_ids"], ["qc"])
        prepare()

    def test_coupled_core_not_rejected_on_minimum_probe_alone(self):
        calls = []
        def probe(rows, **kwargs):
            calls.append(kwargs["_minimum_only"])
            result = plan_campaign_resource_budget(rows, **LIMITS)
            if kwargs["_minimum_only"]:
                result["feasibility_status"] = "infeasible"
            return result
        with patch("salsbury_md_analysis.resource_planning.plan_global_stride_projection_coupled_campaign_resource_budget", side_effect=probe):
            result = recommend_scientifically_valid_task_subset([task("qc", "structural_integrity_qc")],
                use_global_stride_coupling=True, **LIMITS)
        self.assertEqual(calls[:2], [True, False])
        self.assertEqual(result["recommendation_status"], "feasible_subset_found")
        self.assertEqual(result["configuration_patch"], {})

    def test_minimum_mode_does_not_upgrade_or_change_fixed_stride(self):
        row = task("qc", "structural_integrity_qc", 0)
        row.update(source_frames_per_replica=[1000], minimum_frames_per_replica=100,
                   cpu_seconds_per_physical_frame=0.001)
        minimal = plan_campaign_resource_budget([row], _minimum_only=True, **LIMITS)
        refined = plan_campaign_resource_budget([row], **LIMITS)
        self.assertEqual(minimal["tasks"][0]["integer_stride"], 10)
        self.assertEqual(refined["tasks"][0]["integer_stride"], 1)
        self.assertEqual(minimal["allocation_saturation"]["stop_reason"], "minimum_feasibility_only")
        row["required_integer_stride"] = 5
        self.assertEqual(plan_campaign_resource_budget([row], _minimum_only=True, **LIMITS)["tasks"][0]["integer_stride"], 5)

    def test_parent_projection_supplies_higher_child_floor(self):
        parent = task("pca", "common_pca", 0)
        parent.update(source_frames_per_replica=[1000], minimum_frames_per_replica=100,
                      workflow_id="shared", task_scope="conformational_view")
        child = task("fit", "alternative_clustering", 0)
        child.update(source_frames_per_replica=[1000], minimum_frames_per_replica=250,
                     workflow_id="shared", task_scope="conformational_view_algorithm_fit", algorithm_id="pam")
        original = deepcopy([parent, child])
        result = plan_projection_coupled_campaign_resource_budget([parent, child], _minimum_only=True, **LIMITS)
        rows = {row["task_id"]: row for row in result["tasks"]}
        self.assertEqual(result["feasibility_status"], "feasible")
        self.assertGreaterEqual(min(rows["pca"]["selected_physical_frames_per_replica"]), 250)
        self.assertEqual(rows["fit"]["source_frames_per_replica"], rows["pca"]["selected_physical_frames_per_replica"])
        self.assertGreaterEqual(min(rows["fit"]["selected_physical_frames_per_replica"]), 250)
        self.assertEqual([parent, child], original)

    def test_coupled_feasibility_early_exit_keeps_valid_effective_strides(self):
        rows = resource_fixtures.ResourcePlanningTests._cache_coupling_tasks()
        result = plan_global_stride_projection_coupled_campaign_resource_budget(rows,
            _minimum_only=True, coordinate_cache_minimum_frames_per_replica=1,
            overall_stride_candidate_strides=[1, 2, 4], **LIMITS)
        self.assertEqual(result["feasibility_status"], "feasible")
        coupling = result["global_stride_coupling"]
        self.assertEqual(coupling["search_goal"], "minimum_feasibility")
        self.assertEqual(coupling["evaluated_candidate_strides"], [1])
        self.assertEqual(coupling["early_terminated_candidate_strides"], [2, 4])
        for row in result["tasks"]:
            self.assertGreater(row["integer_stride"], 0)
            self.assertFalse(row.get("below_standard_scientific_minimum", False))

    def test_pruning_validates_linear_number_of_subsets_then_refines_once(self):
        for count in (4, 8, 12):
            rows = [task("qc", "structural_integrity_qc")] + [
                task(f"optional{i}", f"optional{i}", 20) for i in range(count)]
            with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", wraps=plan_campaign_resource_budget) as planner:
                result = recommend_scientifically_valid_task_subset(rows, **LIMITS)
            self.assertEqual(result["retained_task_ids"], ["qc"])
            self.assertEqual(len(result["decisions"]), count)
            self.assertEqual(planner.call_count, count + 3)
            self.assertEqual(result["planning_evaluations"], {"minimum_feasibility": count + 1, "sampling_refinement": 2})
            self.assertEqual(result["recommended_plan"]["feasibility_status"], "feasible")

    def test_refinement_failure_retains_validated_minimum_plan(self):
        def planner(rows, **kwargs):
            if not kwargs.get("_minimum_only"):
                raise PlanningSearchError("test refinement stalled", diagnostics={})
            return plan_campaign_resource_budget(rows, **kwargs)
        with patch("salsbury_md_analysis.resource_planning.plan_campaign_resource_budget", side_effect=planner):
            result = recommend_scientifically_valid_task_subset([task("qc", "structural_integrity_qc")], **LIMITS)
        self.assertTrue(result["minimum_plan_retained_after_refinement_failure"])
        self.assertEqual(result["recommended_plan"]["feasibility_status"], "feasible")

    def test_memory_blocker_prioritized_and_dependent_removed(self):
        expensive = task("costly", "costly", 2)
        big = task("big", "big", 0.1)
        big["estimated_peak_memory_gib"] = 100
        child = task("child", "child", 0.1)
        rows = [task("qc", "structural_integrity_qc"), expensive, big, child]
        for row in rows:
            row["planning_dependencies"] = {"depends_on_bundle_ids": [], "wait_for_bundle_ids": []}
        child["planning_dependencies"]["depends_on_bundle_ids"] = ["big"]
        result = recommend_scientifically_valid_task_subset(rows, **LIMITS)
        self.assertEqual(result["retained_task_ids"], ["costly", "qc"])
        self.assertEqual(result["decisions"][0]["removed_task_ids"], ["big", "child"])
