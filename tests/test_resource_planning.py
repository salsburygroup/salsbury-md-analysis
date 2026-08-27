import unittest

from salsbury_md_analysis.resource_planning import (
    ResourcePlanningError,
    calibrate_from_benchmark,
    calibrate_from_benchmarks,
    calibrate_quadratic_from_benchmarks,
    plan_alternative_clustering_fit_strides,
    plan_global_stride_projection_coupled_campaign_resource_budget,
    plan_campaign_resource_budget,
    plan_projection_coupled_campaign_resource_budget,
    recommend_frame_budget,
    recommend_scientifically_valid_task_subset,
    recommend_quadratic_observation_budget,
    pack_resource_waves,
    workflow_useful_parallel_cpu_ceiling,
)
from salsbury_md_analysis.scientific_sampling import (
    profile_contract,
    scientific_sampling_profile,
)


class ResourcePlanningTests(unittest.TestCase):
    def test_useful_cpu_ceiling_sums_independent_bundles_in_busiest_stage(self):
        tasks = [
            {
                "task_id": "a", "dependency_stage": 0,
                "effective_cpu_cap": 16, "intrinsic_cpu_cap": 63,
            },
            {"task_id": "b", "dependency_stage": 1, "effective_cpu_cap": 21},
            {"task_id": "c", "dependency_stage": 1, "effective_cpu_cap": 42},
            {
                "task_id": "c-alias", "execution_bundle_id": "c",
                "dependency_stage": 1, "effective_cpu_cap": 42,
            },
        ]
        self.assertEqual(workflow_useful_parallel_cpu_ceiling(tasks), 63)

    @staticmethod
    def _cache_coupling_tasks():
        common = {
            "effective_cpu_cap": 1,
            "estimated_peak_memory_gib": 1.0,
            "source_frames_per_replica": [1_000, 1_000],
            "minimum_frames_per_replica": 50,
            "maximum_frames_per_replica": 1_000,
            "fixed_cpu_hours": 0.0,
        }
        return [
            {
                **common,
                "task_id": "cache",
                "workflow_id": "cache",
                "module_id": "coordinate_cache",
                "task_scope": "lossless_coordinate_preprocessing",
                "dependency_stage": 0,
                "cpu_seconds_per_physical_frame": 1.0,
                "priority_weight": 100.0,
                "replica_sampling_mode": "independent_all_available",
            },
            {
                **common,
                "task_id": "projection",
                "workflow_id": "shared",
                "module_id": "common_pca",
                "task_scope": "conformational_view",
                "dependency_stage": 1,
                "cpu_seconds_per_physical_frame": 1.0,
                "priority_weight": 10.0,
            },
            {
                **common,
                "task_id": "fit",
                "workflow_id": "shared",
                "module_id": "alternative_clustering",
                "task_scope": "conformational_view_algorithm_fit",
                "dependency_stage": 2,
                "cpu_seconds_per_physical_frame": 2.0,
                "priority_weight": 1.0,
            },
        ]

    def test_cache_coupling_keeps_lossless_cache_when_full_scan_dominates(self):
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            self._cache_coupling_tasks(),
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            coordinate_cache_full_scan_fraction=1.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        coupling = plan["coordinate_cache_coupling"]
        self.assertEqual(
            coupling["selected_coordinate_cache_integer_stride"], 1
        )
        self.assertTrue(coupling["execution_ready"])
        self.assertEqual(
            coupling["evaluated_candidate_strides"],
            [1, 2, 3],
        )
        self.assertEqual(
            coupling["early_terminated_candidate_strides"],
            [4, 5, 10, 20],
        )

    def test_default_overall_stride_grid_includes_one_through_one_hundred(self):
        tasks = self._cache_coupling_tasks()
        for row in tasks:
            row["source_frames_per_replica"] = [10_000, 10_000]
            row["maximum_frames_per_replica"] = 10_000
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
            coordinate_cache_full_scan_fraction=1.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        self.assertEqual(
            plan["global_stride_coupling"]["requested_candidate_strides"],
            [1, 2, 3, 4, 5, 10, 20, 100],
        )

    def test_cache_coupling_rebuilds_projection_and_fit_with_raw_strides(self):
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            self._cache_coupling_tasks(),
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            coordinate_cache_full_scan_fraction=0.0,
            overall_stride_candidate_strides=[1, 2],
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        rows = {row["task_id"]: row for row in plan["tasks"]}
        coupling = plan["coordinate_cache_coupling"]
        self.assertEqual(
            coupling["selected_coordinate_cache_integer_stride"], 2
        )
        self.assertTrue(coupling["execution_ready"])
        self.assertEqual(
            rows["fit"]["source_frames_per_replica"],
            rows["projection"]["selected_physical_frames_per_replica"],
        )
        self.assertEqual(rows["projection"]["effective_raw_integer_stride"], 2)
        self.assertEqual(rows["fit"]["effective_raw_integer_stride"], 4)
        self.assertEqual(rows["fit"]["projection_integer_stride"], 1)
        self.assertEqual(coupling["evaluated_candidate_strides"], [1, 2])

    def test_overall_stride_also_applies_to_solvated_trajectory_analysis(self):
        tasks = self._cache_coupling_tasks()
        tasks.append({
            **tasks[1],
            "task_id": "solvated-water-analysis",
            "workflow_id": "solvated-water-analysis",
            "module_id": "water_analysis",
            "task_scope": "direct_trajectory_estimator",
            "dependency_stage": 1,
            "cpu_seconds_per_physical_frame": 1.0,
        })
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            coordinate_cache_full_scan_fraction=0.0,
            overall_stride_candidate_strides=[1, 2],
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        rows = {row["task_id"]: row for row in plan["tasks"]}
        coupling = plan["global_stride_coupling"]
        self.assertEqual(coupling["selected_overall_trajectory_integer_stride"], 2)
        self.assertEqual(
            rows["solvated-water-analysis"]["overall_trajectory_integer_stride"],
            2,
        )
        self.assertEqual(
            rows["solvated-water-analysis"]["effective_raw_integer_stride"],
            2,
        )
        for candidate in coupling["candidate_evaluations"]:
            self.assertIn("balanced_information_utility", candidate)
            self.assertIn("minimum_normalized_analysis_coverage", candidate)
            self.assertGreaterEqual(
                candidate["planner_evaluation_wall_seconds"], 0.0
            )
        self.assertGreaterEqual(coupling["planner_total_wall_seconds"], 0.0)

    def test_effective_raw_stride_preserves_hbond_system_and_replica_floors(self):
        tasks = self._cache_coupling_tasks()[:1]
        tasks[0].update({
            "source_frames_per_replica": [100_000] * 6,
            "minimum_frames_per_replica": 1,
            "maximum_frames_per_replica": 100_000,
            "cpu_seconds_per_physical_frame": 0.000001,
        })
        tasks.append({
            "task_id": "hbond-discovery",
            "workflow_id": "hbond-discovery",
            "module_id": "hydrogen_bond_discovery",
            "task_scope": "direct_trajectory_estimator",
            "dependency_stage": 1,
            "effective_cpu_cap": 1,
            "source_frames_per_replica": [100_000] * 6,
            "system_ids_per_replica": ["a"] * 3 + ["b"] * 3,
            "minimum_frames_per_replica": 334,
            "maximum_frames_per_replica": 100_000,
            "scientific_sampling_requirements": profile_contract(
                scientific_sampling_profile("hydrogen_bond_discovery")
            ),
            "cpu_seconds_per_physical_frame": 0.000001,
            "estimated_peak_memory_gib": 1.0,
        })
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
            coordinate_cache_minimum_frames_per_replica=1,
            coordinate_cache_full_scan_fraction=0.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        hbond = next(
            row for row in plan["tasks"]
            if row["task_id"] == "hbond-discovery"
        )
        self.assertLessEqual(hbond["effective_raw_integer_stride"], 300)
        self.assertGreaterEqual(
            min(hbond["selected_physical_frames_per_replica"]), 334
        )
        self.assertEqual(
            hbond["scientific_sampling_assessment"]["system_floor_failures"],
            [],
        )

    def test_cache_coupling_rejects_invalid_candidate_stride(self):
        with self.assertRaisesRegex(
            ResourcePlanningError, "exceeds the maximum scientifically allowed"
        ):
            plan_global_stride_projection_coupled_campaign_resource_budget(
                self._cache_coupling_tasks(),
                maximum_parallel_cpus=1,
                maximum_wall_hours=1.0,
                maximum_memory_gib=8.0,
                overall_stride_candidate_strides=[10_000],
            )

    def test_explicit_overall_stride_grid_is_not_silently_expanded(self):
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            self._cache_coupling_tasks(),
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            coordinate_cache_full_scan_fraction=0.0,
            overall_stride_candidate_strides=[2],
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        self.assertEqual(
            plan["global_stride_coupling"]["evaluated_candidate_strides"],
            [2],
        )

    def test_cache_coupling_rejects_missing_or_ambiguous_cache(self):
        tasks = self._cache_coupling_tasks()[1:]
        with self.assertRaisesRegex(
            ResourcePlanningError, "exactly one coordinate-cache task"
        ):
            plan_global_stride_projection_coupled_campaign_resource_budget(
                tasks,
                maximum_parallel_cpus=1,
                maximum_wall_hours=1.0,
                maximum_memory_gib=8.0,
            )

    def test_projection_coupled_replanning_rebuilds_clustering_sources(self):
        tasks = [
            {
                "task_id": "view:shared:common_pca",
                "workflow_id": "shared",
                "module_id": "common_pca",
                "task_scope": "conformational_view",
                "dependency_stage": 1,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [1_000, 1_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 1_000,
                "cpu_seconds_per_physical_frame": 0.1,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 5.0,
            },
            {
                "task_id": "view:shared:alternative_clustering:pam",
                "workflow_id": "shared",
                "module_id": "alternative_clustering",
                "algorithm_id": "pam",
                "task_scope": "conformational_view_algorithm_fit",
                "dependency_stage": 2,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100, 100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.1,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
            },
        ]
        plan = plan_projection_coupled_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=2,
            maximum_wall_hours=8.0,
            maximum_memory_gib=8.0,
        )
        rows = {row["task_id"]: row for row in plan["tasks"]}
        parent = rows["view:shared:common_pca"]
        child = rows["view:shared:alternative_clustering:pam"]
        self.assertEqual(
            child["source_frames_per_replica"],
            parent["selected_physical_frames_per_replica"],
        )
        self.assertEqual(child["source_frames_per_replica"], [1_000, 1_000])
        self.assertEqual(child["maximum_frames_per_replica"], 1_000)
        self.assertTrue(plan["projection_clustering_coupling"]["converged"])
        self.assertGreaterEqual(
            plan["projection_clustering_coupling"]["iterations"], 2
        )

    def test_projection_coupled_replanning_rejects_orphan_clustering_source(self):
        task = {
            "task_id": "view:missing:alternative_clustering:pam",
            "workflow_id": "missing",
            "module_id": "alternative_clustering",
            "algorithm_id": "pam",
            "task_scope": "conformational_view_algorithm_fit",
            "dependency_stage": 2,
            "effective_cpu_cap": 1,
            "source_frames_per_replica": [100],
            "minimum_frames_per_replica": 10,
            "maximum_frames_per_replica": 100,
            "cpu_seconds_per_physical_frame": 0.1,
            "estimated_peak_memory_gib": 1.0,
            "priority_weight": 1.0,
        }
        with self.assertRaisesRegex(
            ResourcePlanningError, "has no common-PCA projection task"
        ):
            plan_projection_coupled_campaign_resource_budget(
                [task],
                maximum_parallel_cpus=1,
                maximum_wall_hours=1.0,
                maximum_memory_gib=4.0,
            )

    def test_projection_coupled_replanning_stabilizes_discrete_stride_cycle(self):
        common = {
            "workflow_id": "shared",
            "source_frames_per_replica": [1_000],
            "minimum_frames_per_replica": 100,
            "maximum_frames_per_replica": 1_000,
            "cpu_seconds_per_physical_frame": 1.0,
            "estimated_peak_memory_gib": 1.0,
            "effective_cpu_cap": 1,
        }
        tasks = [
            {
                **common,
                "task_id": "projection",
                "module_id": "common_pca",
                "task_scope": "conformational_view",
                "dependency_stage": 0,
                "priority_weight": 10.0,
            },
            {
                **common,
                "task_id": "fit",
                "module_id": "alternative_clustering",
                "task_scope": "conformational_view_algorithm_fit",
                "dependency_stage": 1,
                "priority_weight": 1.0,
            },
        ]
        plan = plan_projection_coupled_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=0.2,
            maximum_memory_gib=4.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        rows = {row["task_id"]: row for row in plan["tasks"]}
        self.assertEqual(
            rows["projection"]["selected_physical_frames_per_replica"], [334]
        )
        self.assertEqual(rows["fit"]["source_frames_per_replica"], [334])
        coupling = plan["projection_clustering_coupling"]
        self.assertEqual(coupling["dynamic_cycle_resolution_count"], 1)
        self.assertEqual(coupling["iterations"], 4)
        extended = plan_projection_coupled_campaign_resource_budget(
            plan["tasks"],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=4.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        extended_rows = {row["task_id"]: row for row in extended["tasks"]}
        self.assertEqual(
            extended_rows["projection"][
                "selected_physical_frames_per_replica"
            ],
            [1_000],
        )
        self.assertEqual(
            extended_rows["fit"]["source_frames_per_replica"], [1_000]
        )

    def test_alternative_algorithms_receive_distinct_integer_strides(self):
        plan = plan_alternative_clustering_fit_strides(
            [24_700] * 6,
            member_observation_multiplier=1,
            algorithms=[
                "pam", "gaussian_mixture", "affinity_propagation", "ward"
            ],
            target_wall_hours=24.0,
        )
        methods = plan["algorithm_plans"]
        self.assertEqual(methods["pam"]["primary_stride"], 25)
        self.assertEqual(methods["gaussian_mixture"]["primary_stride"], 5)
        self.assertEqual(methods["affinity_propagation"]["primary_stride"], 33)
        self.assertEqual(methods["ward"]["execution"], "skip")
        for method in ("pam", "gaussian_mixture", "affinity_propagation"):
            self.assertLessEqual(
                methods[method]["selected_fit_observation_count"],
                methods[method]["fit_observation_ceiling"],
            )

    def benchmark(self):
        return {
            "technical_status": "complete",
            "module_id": "example",
            "project_sha256": "a" * 64,
            "report_sha256": "b" * 64,
            "finished_utc": "2026-08-12T00:00:00Z",
            "resources": {"wall_seconds": 100.0, "maximum_rss_kib": 1024 * 100},
            "report_size_bytes": 2000,
            "frame_coverage": {"estimator_selected_frame_count": 100},
        }

    def test_recommends_all_frames_when_pilot_extrapolation_fits(self):
        calibration = calibrate_from_benchmark(self.benchmark())
        plan = recommend_frame_budget(
            calibration,
            total_source_frames=1000,
            replica_count=2,
            target_wall_seconds=2000,
        )
        self.assertTrue(plan["all_frames_fit"])
        self.assertEqual(plan["frame_selection"], {"mode": "fixed_stride_v1"})

    def test_recommends_reported_balanced_subsampling_when_time_exceeds(self):
        calibration = calibrate_from_benchmark(self.benchmark())
        plan = recommend_frame_budget(
            calibration,
            total_source_frames=30000,
            replica_count=3,
            target_wall_seconds=4500,
            minimum_frames_per_replica=100,
            sensitivity_check_policy="off",
        )
        self.assertFalse(plan["all_frames_fit"])
        self.assertEqual(plan["resolved_maximum_frames_per_replica"], 1000)
        self.assertEqual(plan["resolved_selected_frame_count"], 3000)
        self.assertEqual(
            plan["frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 10},
        )
        self.assertEqual(plan["sensitivity_check_policy"], "off")

    def test_rejects_failed_calibration(self):
        benchmark = self.benchmark()
        benchmark["technical_status"] = "failed"
        with self.assertRaises(ResourcePlanningError):
            calibrate_from_benchmark(benchmark)

    def test_multi_point_calibration_separates_scan_overhead(self):
        first = self.benchmark()
        second = self.benchmark()
        first["resources"]["wall_seconds"] = 40.0
        first["frame_coverage"]["estimator_selected_frame_count"] = 100
        second["resources"]["wall_seconds"] = 160.0
        second["frame_coverage"]["estimator_selected_frame_count"] = 500
        calibration = calibrate_from_benchmarks([first, second])
        self.assertAlmostEqual(calibration["fixed_overhead_seconds"], 10.0)
        self.assertAlmostEqual(calibration["seconds_per_frame"], 0.3)

    def test_quadratic_calibration_and_budget_keep_full_assignment_scope(self):
        benchmarks = []
        for fit_count, wall_seconds, memory_mib in (
            (750, 66.25, 156.25),
            (1500, 235.0, 325.0),
            (3000, 910.0, 1000.0),
        ):
            benchmark = self.benchmark()
            benchmark["module_id"] = "alternative_clustering"
            benchmark["resources"] = {
                "wall_seconds": wall_seconds,
                "maximum_rss_kib": memory_mib * 1024,
            }
            benchmark["workload_signature_sha256"] = "d" * 64
            benchmark["full_assignment_observation_count"] = 30000
            benchmarks.append(benchmark)
        calibration = calibrate_quadratic_from_benchmarks(
            benchmarks,
            evaluated_fit_observation_counts=[750, 1500, 3000],
            calibration_id="quadratic-test",
        )
        self.assertAlmostEqual(
            calibration["seconds_per_squared_fit_observation"], 0.0001
        )
        self.assertAlmostEqual(calibration["fixed_overhead_seconds"], 10.0)
        plan = recommend_quadratic_observation_budget(
            calibration,
            total_source_observations=30000,
            replica_count=3,
            target_wall_seconds=1500.0,
            target_memory_mib=2000.0,
            sensitivity_check_policy="off",
        )
        self.assertFalse(plan["all_observations_fit"])
        self.assertEqual(len(plan["fit_sampling"]["strides"]), 1)
        self.assertIn(
            plan["fit_sampling"]["primary_stride"],
            plan["fit_sampling"]["strides"],
        )

    def test_quadratic_sensitivity_is_explicitly_optional(self):
        calibration = {
            "module_id": "alternative_clustering",
            "calibration_id": "quadratic-test",
            "fixed_overhead_seconds": 10.0,
            "seconds_per_squared_fit_observation": 0.0001,
            "fixed_memory_mib": 100.0,
            "memory_mib_per_squared_fit_observation": 0.0001,
            "maximum_calibrated_fit_observations": 3000,
        }
        off = recommend_quadratic_observation_budget(
            calibration,
            total_source_observations=30000,
            replica_count=3,
            sensitivity_check_policy="off",
        )
        recommended = recommend_quadratic_observation_budget(
            calibration,
            total_source_observations=30000,
            replica_count=3,
            sensitivity_check_policy="recommend",
        )
        self.assertEqual(len(off["fit_sampling"]["strides"]), 1)
        self.assertEqual(len(recommended["fit_sampling"]["strides"]), 2)

    def test_campaign_envelope_changes_with_cpu_and_wall_configuration(self):
        tasks = [
            {
                "task_id": task_id,
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10_000, 10_000, 10_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 10_000,
                "cpu_seconds_per_physical_frame": 1.0,
                "estimated_peak_memory_gib": 2.0,
                "priority_weight": priority,
            }
            for task_id, priority in (("fes", 10.0), ("sasa", 1.0))
        ]
        constrained = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
        )
        extended = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=32,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
        )
        self.assertEqual(constrained["raw_capacity_cpu_hours"], 1.0)
        self.assertEqual(extended["raw_capacity_cpu_hours"], 768.0)
        self.assertEqual(constrained["feasibility_status"], "feasible")
        constrained_tasks = {
            row["task_id"]: row for row in constrained["tasks"]
        }
        self.assertGreaterEqual(
            constrained_tasks["fes"]["selected_physical_frame_count"],
            constrained_tasks["sasa"]["selected_physical_frame_count"],
        )
        self.assertTrue(all(
            not row["subsampling_triggered"] for row in extended["tasks"]
        ))

    def test_larger_wall_envelope_never_reduces_task_coverage(self):
        tasks = [
            {
                "task_id": "high-priority-quadratic",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.0,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 10.0,
                "power_law_cost_model": {
                    "calibration_observations": 10,
                    "calibration_cpu_hours": 0.5,
                    "time_exponent": 2.0,
                    "calibration_memory_gib": 1.0,
                    "memory_exponent": 1.0,
                },
            },
            {
                "task_id": "lower-priority-linear",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 180.0,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
            },
        ]
        shorter = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=2.0,
            maximum_memory_gib=8.0,
        )
        longer = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=3.25,
            maximum_memory_gib=8.0,
        )
        shorter_rows = {
            row["task_id"]: row for row in shorter["tasks"]
        }
        longer_rows = {
            row["task_id"]: row for row in longer["tasks"]
        }
        self.assertEqual(shorter["feasibility_status"], "feasible")
        self.assertEqual(longer["feasibility_status"], "feasible")
        for task_id in shorter_rows:
            self.assertGreaterEqual(
                longer_rows[task_id]["selected_physical_frame_count"],
                shorter_rows[task_id]["selected_physical_frame_count"],
                task_id,
            )

    def test_unused_cpu_budget_reports_wall_or_parallelism_limit(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "serial-analysis",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 2_500,
                "cpu_seconds_per_physical_frame": 1.1,
                "estimated_peak_memory_gib": 1.0,
            }],
            maximum_parallel_cpus=42,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
        )
        self.assertGreater(plan["unused_science_cpu_hours"], 0.0)
        utilization = plan["resource_budget_utilization"]
        self.assertLess(utilization["science_cpu_hour_fraction"], 0.1)
        self.assertGreater(utilization["science_wall_time_fraction"], 0.9)
        self.assertEqual(
            plan["allocation_saturation"]["stop_reason"],
            "all_eligible_frame_ceilings_reached",
        )
        self.assertIn(
            "parallelism",
            plan["allocation_saturation"]["unused_cpu_hour_interpretation"],
        )

    def test_power_law_methods_are_allocated_separately_but_scheduled_as_bundle(self):
        tasks = []
        for algorithm, exponent in (("linear", 1.0), ("quadratic", 2.0)):
            tasks.append({
                "task_id": f"clustering:{algorithm}",
                "dependency_stage": 1,
                "effective_cpu_cap": 1,
                "execution_bundle_id": "clustering:sweep",
                "source_frames_per_replica": [1_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 1_000,
                "cpu_seconds_per_physical_frame": 0.0,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
                "balance_group": f"clustering:{algorithm}",
                "power_law_cost_model": {
                    "calibration_observations": 100,
                    "calibration_cpu_hours": 0.1,
                    "time_exponent": exponent,
                    "calibration_memory_gib": 1.0,
                    "memory_exponent": exponent,
                },
            })
        plan = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=32,
            maximum_wall_hours=1.0,
            maximum_memory_gib=16.0,
            planning_utilization=0.85,
            pilot_budget_fraction=0.05,
        )
        rows = {row["task_id"]: row for row in plan["tasks"]}
        self.assertNotEqual(
            rows["clustering:linear"]["integer_stride"],
            rows["clustering:quadratic"]["integer_stride"],
        )
        stage = plan["stages"][0]
        self.assertEqual(stage["task_count"], 2)
        self.assertEqual(stage["execution_bundle_count"], 1)
        self.assertAlmostEqual(
            stage["estimated_wall_hours_lower_bound"],
            stage["estimated_cpu_hours"],
        )
        self.assertTrue(all(
            row["estimated_peak_memory_gib_at_selected_observations"] <= 16.0
            for row in rows.values()
        ))

    def test_comparison_balance_group_uses_equal_physical_frame_budgets(self):
        tasks = []
        for system_id, rate in (("control", 1.0), ("lesion", 2.0)):
            tasks.append({
                "task_id": f"{system_id}:member-fes",
                "system_id": system_id,
                "dependency_stage": 1,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100_000, 100_000, 100_000],
                "minimum_frames_per_replica": 500,
                "maximum_frames_per_replica": 100_000,
                "cpu_seconds_per_physical_frame": rate,
                "estimated_peak_memory_gib": 4.0,
                "priority_weight": 10.0,
                "balance_group": "shared-member-fes",
                "member_observation_multiplier": 2,
            })
        plan = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=2,
            maximum_wall_hours=8.0,
            maximum_memory_gib=16.0,
        )
        rows = {row["system_id"]: row for row in plan["tasks"]}
        self.assertEqual(
            rows["control"]["selected_physical_frames_per_replica"],
            rows["lesion"]["selected_physical_frames_per_replica"],
        )
        self.assertEqual(
            rows["control"]["selected_member_observation_count"],
            2 * rows["control"]["selected_physical_frame_count"],
        )
        self.assertFalse(
            rows["control"]["member_observations_are_independent_replicas"]
        )
        self.assertEqual(
            rows["control"]["frame_selection"]["mode"],
            "integer_stride_per_replica_v1",
        )
        stride = rows["control"]["frame_selection"]["stride"]
        self.assertEqual(
            rows["control"]["selected_physical_frames_per_replica"],
            [((100_000 - 1) // stride) + 1] * 3,
        )

    def test_campaign_minimums_fail_closed_when_unaffordable(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "expensive-water",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100_000] * 3,
                "minimum_frames_per_replica": 1_000,
                "maximum_frames_per_replica": 100_000,
                "cpu_seconds_per_physical_frame": 10.0,
                "estimated_peak_memory_gib": 4.0,
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
        )
        self.assertEqual(plan["feasibility_status"], "infeasible")
        self.assertFalse(plan["execution_authorized"])
        self.assertTrue(plan["infeasibility_reasons"])

    def test_memory_shortfall_reports_required_cap_and_modules(self):
        tasks = []
        for task_id, module_id, memory in (
            ("base:small", "small_module", 2.0),
            ("view:global:large", "large_module", 7.25),
        ):
            tasks.append({
                "task_id": task_id,
                "module_id": module_id,
                "workflow_id": "base",
                "task_scope": "test",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.01,
                "estimated_peak_memory_gib": memory,
            })
        plan = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=4.0,
        )
        memory = plan["memory_feasibility"]
        self.assertFalse(memory["fits_configured_memory"])
        self.assertEqual(memory["minimum_required_memory_gib"], 8.0)
        self.assertEqual(memory["recommended_memory_gib"], 8.0)
        self.assertEqual(memory["memory_shortfall_gib"], 4.0)
        self.assertEqual(
            memory["modules_to_disable_to_fit_configured_memory"],
            ["large_module"],
        )
        self.assertEqual(
            memory[
                "configuration_switches_to_disable_to_fit_configured_memory"
            ],
            ["modules.large_module.enabled"],
        )
        self.assertEqual(
            memory["oversized_tasks"][0]["task_id"], "view:global:large"
        )

    def test_buffered_memory_controls_feasibility_and_resource_waves(self):
        tasks = []
        for task_id in ("first", "second"):
            tasks.append({
                "task_id": task_id,
                "module_id": task_id,
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 1.0,
                "estimated_peak_memory_gib": 60.0,
            })
        plan = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=2,
            maximum_wall_hours=2.0,
            maximum_memory_gib=100.0,
            memory_safety_factor=1.5,
            memory_overhead_gib=1.0,
            minimum_scheduler_memory_gib=2.0,
        )
        self.assertEqual(plan["feasibility_status"], "feasible")
        waves = plan["stages"][0]["resource_waves"]
        self.assertEqual(len(waves), 2)
        self.assertTrue(all(wave["memory_gib"] == 91.0 for wave in waves))
        self.assertTrue(all(wave["cpu_slots"] == 1 for wave in waves))

    def test_deac_memory_margin_can_make_raw_working_set_infeasible(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "structural-qc",
                "module_id": "structural_integrity_qc",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.01,
                "estimated_peak_memory_gib": 126.491,
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=185.0,
            memory_safety_factor=1.5,
            memory_overhead_gib=1.0,
            minimum_scheduler_memory_gib=2.0,
        )
        memory = plan["memory_feasibility"]
        self.assertEqual(plan["feasibility_status"], "infeasible")
        self.assertEqual(memory["minimum_required_working_set_gib"], 126.491)
        self.assertEqual(memory["minimum_required_memory_gib"], 191.0)
        self.assertEqual(
            memory["modules_to_disable_to_fit_configured_memory"],
            ["structural_integrity_qc"],
        )

    def test_resource_wave_packer_rejects_one_oversized_task(self):
        with self.assertRaisesRegex(ResourcePlanningError, "campaign limit"):
            pack_resource_waves(
                [{
                    "item_id": "oversized",
                    "cpu_slots": 1,
                    "memory_gib": 101,
                    "wall_hours": 1,
                }],
                maximum_parallel_cpus=2,
                maximum_parallel_memory_gib=100,
            )

    def test_measured_memory_scales_from_observation_coverage(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "base:measured",
                "module_id": "measured_module",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [120_000],
                "minimum_frames_per_replica": 1_000,
                "maximum_frames_per_replica": 120_000,
                "cpu_seconds_per_physical_frame": 0.0001,
                "estimated_peak_memory_gib": 20.0,
                "measured_memory_cost_model": {
                    "calibration_observations": 120_000,
                    "calibration_memory_gib": 20.0,
                    "memory_exponent": 0.5,
                    "minimum_observation_scale": 0.1,
                },
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=2.1,
        )
        self.assertEqual(plan["feasibility_status"], "feasible")
        row = plan["tasks"][0]
        self.assertLessEqual(
            row["estimated_peak_memory_gib_at_selected_observations"], 2.1
        )
        self.assertGreaterEqual(row["selected_physical_frame_count"], 1_000)
        self.assertLess(row["selected_physical_frame_count"], 120_000)

    def test_campaign_requires_pilot_for_unknown_cost(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "new-method",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10_000] * 3,
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 10_000,
                "cpu_seconds_per_physical_frame": None,
                "estimated_peak_memory_gib": 1.0,
            }],
            maximum_parallel_cpus=32,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
        )
        self.assertEqual(plan["feasibility_status"], "pilot_required")
        self.assertEqual(plan["tasks_requiring_project_pilots"], ["new-method"])
        self.assertFalse(plan["execution_authorized"])

    def test_replica_resolved_task_can_retain_unequal_full_lengths(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "rmsd",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [1_000, 2_000, 3_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 3_000,
                "cpu_seconds_per_physical_frame": 0.1,
                "estimated_peak_memory_gib": 1.0,
                "replica_sampling_mode": "independent_all_available",
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
        )
        self.assertEqual(
            plan["tasks"][0]["selected_physical_frames_per_replica"],
            [1_000, 2_000, 3_000],
        )

    def test_pooled_minimum_does_not_force_stride_one_for_sparse_replica(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "pooled-conditioned-analysis",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10_000, 140, 10_000],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 5_000,
                "cpu_seconds_per_physical_frame": 1.0,
                "estimated_peak_memory_gib": 1.0,
                "replica_sampling_mode": "balanced_pooled",
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
        )
        row = plan["tasks"][0]
        self.assertEqual(plan["feasibility_status"], "feasible")
        self.assertGreater(row["integer_stride"], 1)
        self.assertGreaterEqual(row["selected_physical_frame_count"], 300)
        self.assertLessEqual(max(row["selected_physical_frames_per_replica"]), 5_000)
        self.assertEqual(row["minimum_frame_scope"], "pooled_physical_frames")
        self.assertEqual(row["minimum_selected_physical_frame_count"], 300)

    def test_quadratic_pooled_minimum_uses_total_not_each_sparse_replica(self):
        plan = plan_campaign_resource_budget(
            [{
                "task_id": "clustering:affinity",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10_000, 47, 10_000],
                "minimum_frames_per_replica": 250,
                "maximum_frames_per_replica": 10_000,
                "cpu_seconds_per_physical_frame": 0.0,
                "estimated_peak_memory_gib": 3.45,
                "replica_sampling_mode": "balanced_pooled",
                "power_law_cost_model": {
                    "calibration_observations": 3_000,
                    "calibration_cpu_hours": 0.02,
                    "time_exponent": 2.0,
                    "calibration_memory_gib": 3.45,
                    "memory_exponent": 2.0,
                },
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
        )
        row = plan["tasks"][0]
        self.assertEqual(plan["feasibility_status"], "feasible")
        self.assertGreater(row["integer_stride"], 1)
        self.assertGreaterEqual(row["selected_physical_frame_count"], 750)
        self.assertLessEqual(
            row["estimated_peak_memory_gib_at_selected_observations"], 8.0
        )

    def test_finalization_headroom_is_reserved_from_campaign_budget(self):
        task = {
            "task_id": "bounded-method",
            "dependency_stage": 0,
            "effective_cpu_cap": 1,
            "source_frames_per_replica": [10_000],
            "minimum_frames_per_replica": 100,
            "maximum_frames_per_replica": 10_000,
            "cpu_seconds_per_physical_frame": 1.0,
            "estimated_peak_memory_gib": 1.0,
        }
        plan = plan_campaign_resource_budget(
            [task],
            maximum_parallel_cpus=4,
            maximum_wall_hours=10.0,
            maximum_memory_gib=8.0,
            planning_utilization=0.85,
            pilot_budget_fraction=0.05,
            finalization_headroom_fraction=0.10,
        )
        self.assertAlmostEqual(plan["raw_capacity_cpu_hours"], 40.0)
        self.assertAlmostEqual(plan["reserved_pilot_cpu_hours"], 2.0)
        self.assertAlmostEqual(plan["reserved_finalization_cpu_hours"], 4.0)
        self.assertAlmostEqual(plan["science_budget_cpu_hours"], 28.0)
        self.assertAlmostEqual(plan["science_budget_wall_hours"], 7.0)

    def test_rejects_reserved_fraction_that_consumes_utilization(self):
        with self.assertRaisesRegex(
            ResourcePlanningError, "pilot plus finalization"
        ):
            plan_campaign_resource_budget(
                [{
                    "task_id": "method",
                    "dependency_stage": 0,
                    "effective_cpu_cap": 1,
                    "source_frames_per_replica": [100],
                    "minimum_frames_per_replica": 10,
                    "maximum_frames_per_replica": 100,
                    "cpu_seconds_per_physical_frame": 1.0,
                    "estimated_peak_memory_gib": 1.0,
                }],
                maximum_parallel_cpus=1,
                maximum_wall_hours=1.0,
                maximum_memory_gib=8.0,
                planning_utilization=0.85,
                pilot_budget_fraction=0.05,
                finalization_headroom_fraction=0.80,
            )

    def test_method_subset_recommendation_removes_low_priority_bottleneck(self):
        tasks = [
            {
                "task_id": "rmsd",
                "module_id": "replica_rmsd_rg",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.01,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 100.0,
            },
            {
                "task_id": "dccm",
                "module_id": "dccm",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 10.0,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
            },
        ]
        report = recommend_scientifically_valid_task_subset(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=0.1,
            maximum_memory_gib=8.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        self.assertEqual(report["recommendation_status"], "feasible_subset_found")
        self.assertEqual(
            report["disabled_configuration_switches"],
            ["modules.dccm.enabled"],
        )
        self.assertFalse(report["automatic_changes_applied"])

    def test_method_subset_never_removes_protected_structural_qc(self):
        tasks = [
            {
                "task_id": "qc",
                "module_id": "structural_integrity_qc",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 100,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 360.0,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
            },
            {
                "task_id": "dccm",
                "module_id": "dccm",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 100,
                "cpu_seconds_per_physical_frame": 0.01,
                "estimated_peak_memory_gib": 1.0,
                "priority_weight": 1.0,
            },
        ]
        report = recommend_scientifically_valid_task_subset(
            tasks,
            maximum_parallel_cpus=1,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        self.assertEqual(
            report["recommendation_status"], "no_feasible_subset_found"
        )
        self.assertIn("No acceptable reduced plan", report["recommendation_message"])
        self.assertIn("qc", report["retained_task_ids"])
        self.assertTrue(report["protected_set_preserved"])
        self.assertEqual(report["disabled_configuration_switches"], [])
        self.assertEqual(report["configuration_patch"], {})
        self.assertIn(
            "modules.dccm.enabled", report["attempted_configuration_switches"]
        )
        minimum_request = report[
            "best_protected_subset_minimum_resource_request"
        ]
        self.assertEqual(
            minimum_request["request_scope"],
            "best_dependency_closed_subset_that_preserves_all_protected_modules",
        )
        self.assertEqual(
            minimum_request["recommended_request"]["wall_hours"], 10
        )
        self.assertEqual(
            minimum_request["warning"]["code"],
            "PERMISSIVE_MINIMUM_NOT_SCIENTIFIC_SUFFICIENCY",
        )

    def test_campaign_plan_reports_padded_permissive_minimum_request(self):
        tasks = [
            {
                "task_id": task_id,
                "module_id": f"project_local_{task_id}",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 10,
                "cpu_seconds_per_physical_frame": 360.0,
                "estimated_peak_memory_gib": 1.0,
            }
            for task_id in ("a", "b")
        ]
        report = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=2,
            maximum_wall_hours=4.0,
            maximum_memory_gib=10.0,
            planning_utilization=0.8,
            pilot_budget_fraction=0.1,
            finalization_headroom_fraction=0.1,
            memory_safety_factor=2.0,
            memory_overhead_gib=1.0,
        )
        minimum_request = report["permissive_minimum_resource_request"]
        self.assertEqual(
            minimum_request["status"], "available_within_all_input_caps"
        )
        self.assertEqual(minimum_request["recommended_request"], {
            "parallel_cpus": 2,
            "aggregate_memory_gib": 6.0,
            "wall_hours": 2,
        })
        self.assertAlmostEqual(
            minimum_request["unrounded_request"]["wall_hours"], 5.0 / 3.0
        )
        self.assertAlmostEqual(
            minimum_request["padding_factors"]["science_wall_fraction"], 0.6
        )
        self.assertTrue(minimum_request["fits_input_wall_cap"])

    def test_permissive_minimum_reports_larger_wall_request(self):
        report = plan_campaign_resource_budget(
            [{
                "task_id": "protected",
                "module_id": "structural_integrity_qc",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [10],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 10,
                "cpu_seconds_per_physical_frame": 360.0,
                "estimated_peak_memory_gib": 1.0,
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=0.5,
            maximum_memory_gib=8.0,
            planning_utilization=0.8,
            pilot_budget_fraction=0.1,
            finalization_headroom_fraction=0.1,
        )
        request = report["permissive_minimum_resource_request"]
        self.assertEqual(request["status"], "requires_larger_wall_time")
        self.assertFalse(request["fits_input_wall_cap"])
        self.assertEqual(request["recommended_request"]["wall_hours"], 2)
        self.assertEqual(request["additional_wall_hours_required"], 2)

    def test_hydrogen_bond_stride_meets_replica_and_system_floors(self):
        system_ids = [
            system_id
            for system_id in ("system-a", "system-b")
            for _ in range(3)
        ]
        report = plan_campaign_resource_budget(
            [{
                "task_id": "hbond-discovery",
                "module_id": "hydrogen_bond_discovery",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "source_frames_per_replica": [100_000] * 6,
                "system_ids_per_replica": system_ids,
                "minimum_frames_per_replica": 334,
                "maximum_frames_per_replica": 100_000,
                "scientific_sampling_requirements": profile_contract(
                    scientific_sampling_profile("hydrogen_bond_discovery")
                ),
                "cpu_seconds_per_physical_frame": 0.0001,
                "estimated_peak_memory_gib": 1.0,
            }],
            maximum_parallel_cpus=1,
            maximum_wall_hours=24.0,
            maximum_memory_gib=8.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        task = report["tasks"][0]
        self.assertLessEqual(task["integer_stride"], 300)
        self.assertGreaterEqual(
            min(task["selected_physical_frames_per_replica"]), 334
        )
        self.assertEqual(
            task["scientific_sampling_assessment"]["system_floor_failures"],
            [],
        )
        self.assertEqual(report["feasibility_status"], "feasible")

    def test_campaign_plan_warns_when_requested_cpus_exceed_useful_ceiling(self):
        report = plan_campaign_resource_budget(
            [{
                "task_id": "qc",
                "module_id": "structural_integrity_qc",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "intrinsic_cpu_cap": 1,
                "source_frames_per_replica": [10],
                "minimum_frames_per_replica": 1,
                "maximum_frames_per_replica": 10,
                "cpu_seconds_per_physical_frame": 0.01,
                "estimated_peak_memory_gib": 1.0,
            }],
            maximum_parallel_cpus=8,
            maximum_wall_hours=1.0,
            maximum_memory_gib=8.0,
            planning_utilization=1.0,
            pilot_budget_fraction=0.0,
        )
        self.assertEqual(
            report["workflow_parallel_capacity"]["useful_parallel_cpu_ceiling"],
            1,
        )
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(
            report["resource_warnings"][0]["code"],
            "REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM",
        )
        self.assertEqual(report["resource_warnings"][0]["excess_parallel_cpus"], 7)
        self.assertEqual(report["effective_parallel_cpu_cap"], 1)
        self.assertEqual(
            report["workflow_parallel_capacity"]["effective_parallel_cpu_cap"],
            1,
        )
        self.assertIn(
            "Slurm submission will be changed to 1 CPU",
            report["resource_warnings"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()
