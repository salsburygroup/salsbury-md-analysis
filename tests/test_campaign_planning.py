import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_planning import (
    _apply_direct_project_sampling,
    _apply_measured_resource_calibrations,
    _automatic_context_tasks,
    _base_derived_tasks,
    _campaign_infeasibility_detail,
    _apply_system_memory_scaling,
    _view_tasks,
)


class CampaignPlanningTests(unittest.TestCase):
    def test_allosteric_pathways_retains_exact_allocated_dccm_frames(self):
        project = {
            "definitions": {
                "dccm": {
                    "frame_stride": 1,
                    "frame_selection": {"mode": "fixed_stride_v1"},
                },
                "allosteric_pathways": {
                    "frame_stride": 1,
                    "frame_selection": {"mode": "fixed_stride_v1"},
                },
            },
        }
        sampling = {
            "method_plans": [{
                "module_id": "dccm",
                "frame_selection": {
                    "mode": "integer_stride_per_replica_v1",
                    "stride": 7,
                },
            }],
        }
        _apply_direct_project_sampling(project, sampling)
        self.assertEqual(
            project["definitions"]["allosteric_pathways"]["frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 7},
        )

    def test_nemo_calibrated_experimental_view_tasks_are_planned(self):
        project = {
            "requested_modules": [
                "common_pca", "perturbation_response_dynamics",
                "trajectory_reweighting",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 84,
                    "component_count": 10,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "perturbation_response_dynamics": {
                    "maximum_nodes": 28,
                    "random_force_directions": 250,
                },
                "trajectory_reweighting": {},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-macromolecular_trace.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [1_000], 423, time_safety_factor=1.5
            )
        by_module = {row["module_id"]: row for row in tasks}
        for module_id in (
            "perturbation_response_dynamics", "trajectory_reweighting"
        ):
            self.assertIn(module_id, by_module)
            self.assertEqual(
                by_module[module_id]["calibration_status"],
                "completed_single_fixture_provisional_scaling",
            )
            self.assertIn("nemo-zinc-finger-1000f", by_module[module_id]["calibration_id"])
            self.assertEqual(by_module[module_id]["provisional_workload_scale"], 1.0)

    def test_nani_kmeans_grid_is_explicit_in_planner_task(self):
        project = {
            "requested_modules": ["common_pca", "clustering_kmeans"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 84,
                    "component_count": 10,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "clustering_kmeans": {
                    "k_values": list(range(2, 13)),
                    "initialization_methods": [
                        "nani_strat_all", "nani_strat_reduced",
                    ],
                    "nani_percentage": 10,
                    "silhouette_random_seeds": [0, 7, 19, 41],
                    "maximum_silhouette_observations": 1_000,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-global.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [1_000], 423, time_safety_factor=1.5
            )
            sampled_tasks = _view_tasks(
                path, [1_001], 423, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks if row["module_id"] == "clustering_kmeans"
        )
        self.assertEqual(
            task["initialization_methods"],
            ["nani_strat_all", "nani_strat_reduced"],
        )
        self.assertEqual(task["initialization_runs_per_k"], 2)
        self.assertEqual(task["nani_percentage"], 10)
        self.assertEqual(task["silhouette_random_seeds"], [0, 7, 19, 41])
        self.assertFalse(task["silhouette_sampling_required"])
        self.assertEqual(task["silhouette_evaluations_per_k"], 1)
        self.assertEqual(task["provisional_workload_scale"], 1.0)
        sampled_task = next(
            row for row in sampled_tasks
            if row["module_id"] == "clustering_kmeans"
        )
        self.assertTrue(sampled_task["silhouette_sampling_required"])
        self.assertEqual(sampled_task["silhouette_evaluations_per_k"], 4)
        self.assertEqual(sampled_task["provisional_workload_scale"], 4.0)

    def test_random_feature_koopman_grid_and_seed_gates_are_planned(self):
        project = {
            "requested_modules": [
                "common_pca", "random_feature_koopman",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 84,
                    "component_count": 10,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "random_feature_koopman": {
                    "random_feature_counts": [32, 64],
                    "bandwidth_scales": [0.5, 1.0, 2.0],
                    "random_seeds": [0, 7, 19, 41],
                    "lag_frames": 5,
                    "minimum_pairs_per_segment": 10,
                    "cross_validation_folds": 5,
                    "maximum_seed_vamp_e_relative_range": 0.25,
                    "minimum_seed_subspace_similarity": 0.70,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-global.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [1_000], 423, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks
            if row["module_id"] == "random_feature_koopman"
        )
        self.assertEqual(task["hyperparameter_candidate_count"], 6)
        self.assertEqual(task["random_feature_fit_evaluation_count"], 24)
        self.assertEqual(task["random_feature_seeds"], [0, 7, 19, 41])
        self.assertEqual(task["minimum_frames_per_replica_for_lag_pairs"], 15)
        self.assertEqual(task["maximum_seed_vamp_e_relative_range"], 0.25)
        self.assertEqual(task["minimum_seed_subspace_similarity"], 0.70)
        self.assertEqual(task["provisional_workload_scale"], 1.0)

    def test_interaction_persistence_gates_are_planned(self):
        project = {
            "requested_modules": [
                "interaction_fingerprints", "interaction_persistence",
            ],
            "definitions": {
                "interaction_persistence": {
                    "gap_tolerance_observations": [0, 1],
                    "minimum_complete_events": 2,
                    "maximum_event_records": 50_000,
                    "maximum_interval_relative_deviation": 0.01,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _automatic_context_tasks(
                path, [1_000], time_safety_factor=1.5,
                context_id="base", task_namespace="base",
                task_scope="base_automatic_chemistry",
            )
        task = next(
            row for row in tasks
            if row["module_id"] == "interaction_persistence"
        )
        fingerprint = next(
            row for row in tasks
            if row["module_id"] == "interaction_fingerprints"
        )
        self.assertEqual(fingerprint["estimated_peak_memory_gib"], 16.0)
        self.assertEqual(
            fingerprint["memory_cost_basis"],
            "fixed_upstream_artifact_materialization_floor",
        )
        self.assertEqual(task["upstream_module_id"], "interaction_fingerprints")
        self.assertEqual(task["gap_tolerance_observations"], [0, 1])
        self.assertEqual(task["primary_gap_tolerance_observations"], 0)
        self.assertEqual(task["minimum_complete_events"], 2)
        self.assertIn("censoring", task["event_boundary_policy"])

    def test_spatial_interaction_exact_mode_gates_are_planned(self):
        project = {
            "requested_modules": [
                "interaction_fingerprints", "spatial_interaction_ensembles",
            ],
            "definitions": {
                "spatial_interaction_ensembles": {
                    "point_construction_policy": "endpoint_partner_coordinates_v1",
                    "alignment_selection": "alignment",
                    "mode_k_values": [2, 3],
                    "minimum_distinct_frames": 20,
                    "minimum_mode_time_blocks": 2,
                    "minimum_mode_replicas": 1,
                    "maximum_point_observations": 50_000,
                    "maximum_exact_mode_points": 1_000,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _automatic_context_tasks(
                path, [1_000], time_safety_factor=1.5,
                context_id="base", task_namespace="base",
                task_scope="base_automatic_chemistry",
            )
        task = next(
            row for row in tasks
            if row["module_id"] == "spatial_interaction_ensembles"
        )
        self.assertEqual(task["upstream_module_id"], "interaction_fingerprints")
        self.assertEqual(task["mode_k_values"], [2, 3])
        self.assertEqual(task["maximum_exact_mode_points"], 1_000)
        self.assertIn("withhold", task["exact_mode_resource_policy"])

    def test_reactive_path_dtw_limits_are_visible_in_late_stage_plan(self):
        project = {
            "requested_modules": [
                "common_pca", "markov_state_models", "reactive_path_ensembles",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 84,
                    "component_count": 10,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "markov_state_models": {
                    "lag_frames": [1, 2, 5, 10],
                    "minimum_transition_count": 1,
                },
                "reactive_path_ensembles": {
                    "maximum_paths_per_direction": 80,
                    "maximum_path_frames": 400,
                    "maximum_pairwise_dtw_cells": 40_000_000,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-global.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(path, [1_000, 1_000], 423, time_safety_factor=1.5)
        by_module = {row["module_id"]: row for row in tasks}
        reactive = by_module["reactive_path_ensembles"]
        self.assertGreater(
            reactive["dependency_stage"],
            by_module["markov_state_models"]["dependency_stage"],
        )
        self.assertEqual(reactive["maximum_retained_paths"], 160)
        self.assertEqual(reactive["maximum_pairwise_dtw_cells"], 40_000_000)
        self.assertEqual(reactive["provisional_workload_scale"], 2.0)
        self.assertEqual(reactive["calibration_status"], "provisional_complexity_model")

    def test_nemo_calibrated_allosteric_task_scales_quadratically_by_nodes(self):
        base_project = {
            "requested_modules": ["dccm", "allosteric_pathways"],
            "definitions": {
                "allosteric_pathways": {"maximum_nodes": 56},
            },
        }
        direct_tasks = [{
            "task_scope": "direct_trajectory_estimator",
            "module_id": "dccm",
            "source_frames_per_replica": [1_000],
            "minimum_frames_per_replica": 20,
            "maximum_frames_per_replica": 1_000,
            "balance_group": "dccm",
            "replica_sampling_mode": "balanced_pooled",
        }]
        tasks = _base_derived_tasks(
            base_project, direct_tasks, time_safety_factor=1.5
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["module_id"], "allosteric_pathways")
        self.assertEqual(task["provisional_workload_scale"], 4.0)
        self.assertAlmostEqual(
            task["cpu_seconds_per_physical_frame"],
            (18.698369 / 1_000) * 4.0 * 1.5,
        )
        self.assertEqual(
            task["calibration_status"],
            "completed_single_fixture_provisional_scaling",
        )

    def test_measured_hbond_overlay_uses_spatial_work_rate(self):
        calibration = {
            "hydrogen_bond_discovery": {
                "conservative_cpu_seconds_per_frame": 999.0,
                "conservative_cpu_seconds_per_spatial_neighbor_pair": 0.01,
                "conservative_spatial_neighbor_pairs_per_selected_frame": 2_000.0,
                "maximum_resident_memory_mib": 1024.0,
                "catalog_sha256": "e" * 64,
                "measurement_count": 1,
                "complete_measurement_count": 1,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 12,
                "maximum_measured_observation_count": 12,
                "maximum_measured_spatial_neighbor_pair_count": 24_000,
                "maximum_measured_spatial_endpoint_count_per_system": 2_000,
            }
        }
        task = {
            "task_id": "direct:hydrogen_bond_discovery",
            "module_id": "hydrogen_bond_discovery",
            "cpu_seconds_per_physical_frame": 1.0,
            "estimated_peak_memory_gib": 1.0,
            "measured_cpu_rate_multiplier": 2.0,
        }
        _apply_measured_resource_calibrations(
            [task], calibration, time_safety_factor=1.5,
        )
        self.assertEqual(task["cpu_seconds_per_physical_frame"], 60.0)
        self.assertEqual(
            task["measured_resource_calibration"]["runtime_work_unit"],
            "spatial_neighbor_pairs_v1",
        )

    def test_direct_hbond_gate_tracks_postallocation_selected_frames(self):
        project = {
            "definitions": {
                "hydrogen_bond_discovery": {
                    "frame_selection": {"mode": "fixed_stride_v1"},
                    "maximum_feature_observations": 125_001 * 2_028,
                }
            }
        }
        sampling_plan = {
            "dimensions": {
                "hydrogen_bond_candidate_planning": {
                    "status": "complete",
                    "common_candidate_count": 125_001,
                }
            },
            "method_plans": [{
                "module_id": "hydrogen_bond_discovery",
                "selected_frame_count": 4_134,
                "frame_selection": {
                    "mode": "integer_stride_per_replica_v1",
                    "stride": 29,
                },
            }],
        }

        _apply_direct_project_sampling(project, sampling_plan)

        definition = project["definitions"]["hydrogen_bond_discovery"]
        self.assertEqual(
            definition["maximum_feature_observations"],
            125_001 * 4_134,
        )
        self.assertEqual(
            definition["frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 29},
        )

    def test_convergence_block_tracks_final_campaign_allocation(self):
        project = {
            "definitions": {
                "convergence_uncertainty": {
                    "block_size_frames": 500,
                    "include_partial_final_block": True,
                    "minimum_blocks": 4,
                }
            }
        }
        sampling_plan = {
            "method_plans": [{
                "module_id": "replica_rmsd_rg",
                "campaign_resource_allocation": {
                    "selected_physical_frames_per_replica": [1_000, 1_200],
                },
            }]
        }

        _apply_direct_project_sampling(project, sampling_plan)

        convergence = project["definitions"]["convergence_uncertainty"]
        self.assertEqual(convergence["block_size_frames"], 100)
        self.assertLessEqual(
            convergence["block_size_frames"]
            * (convergence["minimum_blocks"] - 1) + 1,
            1_000,
        )

    def test_small_system_memory_scaling_retains_floor_and_headroom(self):
        tasks = [{
            "task_id": "small:sasa",
            "estimated_peak_memory_gib": 24.0,
            "power_law_cost_model": {
                "calibration_memory_gib": 10.0,
            },
        }]
        _apply_system_memory_scaling(tasks, 423)
        self.assertEqual(tasks[0]["memory_atom_scale"], 0.1)
        self.assertAlmostEqual(tasks[0]["estimated_peak_memory_gib"], 2.4)
        self.assertEqual(
            tasks[0]["power_law_cost_model"]["calibration_memory_gib"], 1.0
        )
        self.assertEqual(tasks[0]["reference_peak_memory_gib"], 24.0)

    def test_artifact_materialization_memory_floor_is_not_atom_scaled_away(self):
        tasks = [{
            "task_id": "small:derived-join",
            "estimated_peak_memory_gib": 16.0,
            "minimum_materialized_working_set_gib": 16.0,
        }]
        _apply_system_memory_scaling(tasks, 423)
        self.assertEqual(tasks[0]["memory_atom_scale"], 0.1)
        self.assertEqual(tasks[0]["estimated_peak_memory_gib"], 16.0)

    def test_measured_overlay_preserves_declared_workload_scaling(self):
        calibration = {
            "coordinate_cache": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 1024.0,
                "catalog_sha256": "a" * 64,
                "measurement_count": 1,
                "complete_measurement_count": 1,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "complete_only",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100,
                "maximum_measured_observation_count": 100,
            }
        }
        scaled = {
            "task_id": "preprocessing:coordinate_cache",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 1.0,
            "measured_cpu_rate_multiplier": 0.1,
            "measured_memory_multiplier": 0.1,
        }
        reference = {
            "task_id": "reference",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 1.0,
        }
        _apply_measured_resource_calibrations(
            [scaled, reference], calibration,
            time_safety_factor=1.5,
        )
        self.assertAlmostEqual(scaled["cpu_seconds_per_physical_frame"], 0.075)
        self.assertAlmostEqual(reference["cpu_seconds_per_physical_frame"], 0.75)
        self.assertEqual(
            scaled["measured_resource_calibration"]["cpu_rate_workload_multiplier"],
            0.1,
        )

    def test_qualified_memory_measurement_replaces_legacy_default(self):
        calibration = {
            "coordinate_cache": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 1024.0,
                "maximum_completed_resident_memory_mib": 1024.0,
                "catalog_sha256": "b" * 64,
                "measurement_count": 2,
                "complete_measurement_count": 2,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100,
                "maximum_measured_observation_count": 100,
                "memory_replacement_qualified": True,
                "memory_replacement_policy": (
                    "replace_legacy_baseline_with_conservative_completed_measurement"
                ),
            }
        }
        task = {
            "task_id": "preprocessing:coordinate_cache",
            "module_id": "coordinate_cache",
            "cpu_seconds_per_physical_frame": 0.01,
            "estimated_peak_memory_gib": 24.0,
        }
        _apply_measured_resource_calibrations(
            [task], calibration,
            time_safety_factor=1.5,
        )
        self.assertAlmostEqual(task["estimated_peak_memory_gib"], 1.0)
        self.assertAlmostEqual(
            task["measured_memory_cost_model"]["calibration_memory_gib"],
            1.0,
        )
        self.assertTrue(
            task["measured_resource_calibration"]["memory_replacement_qualified"]
        )

    def test_qualified_memory_recalibrates_power_model_at_measured_workload(self):
        calibration = {
            "alternative_clustering": {
                "conservative_cpu_seconds_per_frame": 0.5,
                "maximum_resident_memory_mib": 16384.0,
                "maximum_completed_resident_memory_mib": 16384.0,
                "catalog_sha256": "c" * 64,
                "measurement_count": 2,
                "complete_measurement_count": 2,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "censored_timeout_safety_factor": 1.5,
                "maximum_measured_selected_frame_count": 100_000,
                "maximum_measured_observation_count": 100_000,
                "memory_replacement_qualified": True,
            }
        }
        task = {
            "task_id": "clustering:alternative",
            "module_id": "alternative_clustering",
            "fixed_cpu_hours": 1.0,
            "estimated_peak_memory_gib": 32.0,
            "power_law_cost_model": {
                "calibration_observations": 3_000,
                "calibration_cpu_hours": 1.0,
                "time_exponent": 2.0,
                "calibration_memory_gib": 32.0,
                "memory_exponent": 1.0,
            },
        }
        _apply_measured_resource_calibrations(
            [task], calibration,
            time_safety_factor=1.5,
        )
        model = task["measured_memory_cost_model"]
        self.assertEqual(model["calibration_observations"], 100_000)
        self.assertEqual(model["calibration_memory_gib"], 16.0)
        self.assertEqual(model["memory_exponent"], 1.0)

    def test_base_automatic_chemistry_tasks_are_budgeted_in_dependency_order(self):
        project = {
            "requested_modules": [
                "trajectory_features", "scalar_feature_distributions",
                "scalar_threshold_states",
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _automatic_context_tasks(
                path, [1_000], time_safety_factor=1.5,
                context_id="base", task_namespace="base",
                task_scope="base_automatic_chemistry",
            )
        self.assertEqual(
            {row["task_id"] for row in tasks},
            {
                "base:trajectory_features",
                "base:scalar_feature_distributions",
                "base:scalar_threshold_states",
            },
        )
        stages = {row["module_id"]: row["dependency_stage"] for row in tasks}
        self.assertLess(
            stages["trajectory_features"], stages["scalar_feature_distributions"]
        )
        scalar = next(
            row for row in tasks
            if row["module_id"] == "scalar_feature_distributions"
        )
        self.assertFalse(scalar["measured_calibration_eligible"])
        self.assertIn(
            "validated upstream report",
            scalar["measured_calibration_exclusion_reason"],
        )

    def test_infeasible_message_reports_actual_shortfall_and_retry_bound(self):
        detail = _campaign_infeasibility_detail({
            "infeasibility_reasons": ["minimum wall budget exceeded"],
            "maximum_wall_hours_input": 24.0,
            "science_budget_wall_hours": 16.8,
            "minimum_wall_hours_lower_bound": 20.0,
            "minimum_known_cpu_hours": 200.0,
            "science_budget_cpu_hours": 537.6,
            "maximum_parallel_cpus_input": 32,
        })
        self.assertIn("minimum calibrated critical path 20.000 h", detail)
        self.assertIn("science wall allowance 16.800 h", detail)
        self.assertIn("--target-wall-hours 29", detail)

    def test_grouped_ml_rejects_insufficient_time_block_groups(self):
        project = {
            "requested_modules": ["common_pca", "grouped_ml"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "grouped_ml": {
                    "group_block_size_frames": 10,
                    "minimum_groups": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-groups.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 3 segment/time-block groups"
            ):
                _view_tasks(path, [30], 10_000, time_safety_factor=1.5)

    def test_grouped_ml_carries_group_minimum_into_plan(self):
        project = {
            "requested_modules": ["common_pca", "grouped_ml"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "grouped_ml": {
                    "group_block_size_frames": 10,
                    "minimum_groups": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-minimum-groups.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [31], 10_000, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks if row["module_id"] == "grouped_ml"
        )
        self.assertEqual(task["minimum_frames_per_replica"], 50)
        self.assertEqual(task["minimum_frames_per_replica_for_groups"], 31)
        self.assertEqual(task["maximum_available_groups"], 4)

    def test_information_dynamics_rejects_insufficient_lag_pairs(self):
        project = {
            "requested_modules": ["common_pca", "information_dynamics"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "information_dynamics": {
                    "analyses": ["transfer_entropy", "coskewness"],
                    "lag_frames": 1,
                    "minimum_pairs": 20,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 19 segment-safe lag pairs"
            ):
                _view_tasks(path, [20], 10_000, time_safety_factor=1.5)

    def test_information_dynamics_carries_lag_pair_minimum_into_plan(self):
        project = {
            "requested_modules": ["common_pca", "information_dynamics"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "information_dynamics": {
                    "analyses": ["transfer_entropy", "coskewness"],
                    "lag_frames": 1,
                    "minimum_pairs": 20,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-minimum-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [21], 10_000, time_safety_factor=1.5
            )
        task = next(
            row for row in tasks
            if row["module_id"] == "information_dynamics"
        )
        self.assertEqual(task["minimum_frames_per_replica"], 50)
        self.assertEqual(
            task["minimum_frames_per_replica_for_lag_pairs"], 21
        )
        self.assertEqual(task["minimum_lag_pairs"], 20)
        self.assertEqual(task["maximum_available_lag_pairs"], 20)

    def test_tica_requires_configured_pairs_in_every_segment(self):
        project = {
            "requested_modules": [
                "common_pca",
                "time_lagged_independent_component_analysis",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "time_lagged_independent_component_analysis": {
                    "lag_frames": 3,
                    "minimum_pairs_per_segment": 10,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-tica.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "at least 13 projected frames"
            ):
                _view_tasks(path, [12], 10_000, time_safety_factor=1.5)

    def test_tica_carries_configured_lag_pair_minimum_into_plan(self):
        project = {
            "requested_modules": [
                "common_pca",
                "time_lagged_independent_component_analysis",
            ],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "time_lagged_independent_component_analysis": {
                    "lag_frames": 3,
                    "minimum_pairs_per_segment": 10,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-tica.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(path, [20], 10_000, time_safety_factor=1.5)
        task = next(
            row for row in tasks
            if row["module_id"]
            == "time_lagged_independent_component_analysis"
        )
        self.assertEqual(task["lag_frames"], 3)
        self.assertEqual(task["minimum_lag_pairs_per_segment"], 10)
        self.assertEqual(
            task["minimum_frames_per_replica_for_lag_pairs"], 13
        )
        self.assertEqual(task["maximum_available_lag_pairs"], 17)

    def test_msm_uses_largest_configured_lag_and_transition_minimum(self):
        project = {
            "requested_modules": ["common_pca", "markov_state_models"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "markov_state_models": {
                    "lag_frames": [1, 3, 5],
                    "minimum_transition_count": 4,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-msm.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(path, [10], 10_000, time_safety_factor=1.5)
        task = next(
            row for row in tasks if row["module_id"] == "markov_state_models"
        )
        self.assertEqual(task["largest_configured_lag_frames"], 5)
        self.assertEqual(task["minimum_transition_count"], 4)
        self.assertEqual(
            task["minimum_frames_per_replica_for_transition_pairs"], 9
        )
        self.assertEqual(
            task["maximum_available_transition_pairs_at_largest_lag"], 5
        )

    def test_msm_rejects_insufficient_configured_transition_pairs(self):
        project = {
            "requested_modules": ["common_pca", "markov_state_models"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "markov_state_models": {
                    "lag_frames": [1, 5],
                    "minimum_transition_count": 2,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-short-msm.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only 0 segment-safe transition pairs"
            ):
                _view_tasks(path, [5], 10_000, time_safety_factor=1.5)

    def test_alternative_families_are_independent_logical_tasks_in_one_bundle(self):
        project = {
            "requested_modules": ["common_pca", "alternative_clustering"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 5616,
                    "symmetry_expansion": {"member_count": 2},
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "alternative_clustering": {
                    "algorithms": ["pam", "gaussian_mixture", "ward"],
                    "component_indices": list(range(1, 11)),
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-member-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [10_000, 10_000, 10_000], 100_000,
                time_safety_factor=1.5,
            )
        algorithms = {
            row["algorithm_id"]: row for row in tasks
            if row.get("task_scope") == "conformational_view_algorithm_fit"
        }
        self.assertEqual(set(algorithms), {"pam", "gaussian_mixture"})
        self.assertEqual(
            algorithms["pam"]["execution_bundle_id"],
            algorithms["gaussian_mixture"]["execution_bundle_id"],
        )
        self.assertNotEqual(
            algorithms["pam"]["balance_group"],
            algorithms["gaussian_mixture"]["balance_group"],
        )
        self.assertEqual(
            algorithms["pam"]["power_law_cost_model"]["time_exponent"], 2.0
        )
        self.assertEqual(
            algorithms["gaussian_mixture"]["power_law_cost_model"][
                "time_exponent"
            ],
            1.0,
        )
        self.assertEqual(
            algorithms["pam"]["algorithm_memory_model"],
            "feature_aware_pairwise_matrix",
        )
        self.assertEqual(
            algorithms["pam"]["projection_feature_count"], 10
        )
        self.assertAlmostEqual(
            algorithms["pam"]["measured_memory_cost_model"][
                "calibration_memory_gib"
            ],
            0.125 + 20_000 ** 2 * 186 / 2 ** 30,
        )
        self.assertEqual(
            algorithms["gaussian_mixture"]["measured_memory_cost_model"][
                "calibration_memory_gib"
            ],
            1.0,
        )
        self.assertTrue(algorithms["pam"]["memory_replacement_qualified"])
        self.assertFalse(algorithms["pam"]["measured_calibration_eligible"])
        self.assertEqual(
            algorithms["pam"]["performance_calibration_provenance"][
                "measurement_count"
            ],
            120,
        )
        self.assertFalse(
            algorithms["pam"]["performance_calibration_provenance"][
                "scientific_observations_saved"
            ]
        )

    def test_pald_uses_separate_measured_cubic_observation_model(self):
        project = {
            "requested_modules": ["common_pca", "pald_community_analysis"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 5616,
                    "symmetry_expansion": {"member_count": 2},
                },
                "pald_community_analysis": {"maximum_observations": 500},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-member-view.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [100_000, 100_000, 100_000], 100_000,
                time_safety_factor=1.5,
            )
        pald = next(
            row for row in tasks
            if row["module_id"] == "pald_community_analysis"
        )
        self.assertEqual(pald["fit_observation_limit"], 500)
        self.assertEqual(
            pald["calibration_status"], "completed_bounded_cubic_calibration"
        )
        self.assertAlmostEqual(
            pald["fixed_cpu_hours"], (0.5 + 28.637) * 1.5 / 3600.0
        )
        self.assertEqual(pald["cpu_seconds_per_physical_frame"], 0.0)
        self.assertEqual(pald["estimated_peak_memory_gib"], 1.0)

    def test_representative_only_state_exports_do_not_scale_as_trajectories(self):
        project = {
            "requested_modules": ["common_pca", "state_coordinate_exports"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "state_coordinate_exports": {"write_trajectories": False},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-representative-exports.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [1_000, 1_000], 10_000, time_safety_factor=1.5
            )
        export = next(
            row for row in tasks
            if row["module_id"] == "state_coordinate_exports"
        )
        self.assertEqual(export["coordinate_export_mode"], "representatives_only")
        self.assertFalse(export["state_trajectory_exports_enabled"])
        self.assertFalse(export["measured_calibration_eligible"])
        self.assertEqual(export["cpu_seconds_per_physical_frame"], 0.0)
        self.assertEqual(export["maximum_coordinate_writes"], 250)
        self.assertEqual(export["fixed_cpu_hours"], 0.20)
        self.assertIn(
            "predate indexed requested-frame reads",
            export["measured_calibration_exclusion_reason"],
        )

    def test_state_trajectory_export_cost_uses_bounded_output_count(self):
        project = {
            "requested_modules": ["common_pca", "state_coordinate_exports"],
            "definitions": {
                "common_pca": {
                    "maximum_features": 100,
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                },
                "state_coordinate_exports": {
                    "write_trajectories": True,
                    "maximum_states": 250,
                    "representatives_per_state": 1,
                    "maximum_total_frames": 500,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project-state-trajectories.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            tasks = _view_tasks(
                path, [10_000, 10_000], 10_000, time_safety_factor=1.5
            )
        export = next(
            row for row in tasks
            if row["module_id"] == "state_coordinate_exports"
        )
        self.assertEqual(
            export["coordinate_export_mode"],
            "state_trajectories_and_representatives",
        )
        self.assertTrue(export["state_trajectory_exports_enabled"])
        self.assertEqual(export["maximum_coordinate_writes"], 750)
        self.assertFalse(export["measured_calibration_eligible"])
        self.assertAlmostEqual(
            export["fixed_cpu_hours"], (300.0 + 10.0 * 750) * 1.5 / 3600.0
        )


if __name__ == "__main__":
    unittest.main()
