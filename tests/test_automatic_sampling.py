import json
import struct
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.automatic_sampling import (
    AutomaticSamplingError,
    REFERENCE_ATOM_COUNT,
    _campaign_direct_resource_plan,
    _module_plan,
    _runtime_workload_multiplier,
    automatic_sampling_plan,
    plan_cartesian_pca_basis,
    sampling_profile,
)


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _write_dcd(path: Path, atom_count: int, frames: int) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, frames, 0, 1)
    title = struct.pack("<i", 1) + b"automatic sampling fixture".ljust(80)
    path.write_bytes(
        _record(bytes(header))
        + _record(title)
        + _record(struct.pack("<i", atom_count))
    )


def _write_system(root: Path, frames_per_replica: int = 12_000) -> Path:
    (root / "one.pdb").write_text(
        "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        encoding="utf-8",
    )
    replicas = []
    for index in range(3):
        trajectory = root / f"replica-{index + 1}.dcd"
        _write_dcd(trajectory, atom_count=1, frames=frames_per_replica)
        replicas.append({
            "replica_id": f"r{index + 1}",
            "topology": "one.pdb",
            "segments": [{
                "segment_id": "production",
                "trajectory": trajectory.name,
                "timing": {
                    "first_frame_time": 0,
                    "frame_interval": 10,
                    "unit": "ps",
                },
            }],
        })
    path = root / "system.json"
    path.write_text(json.dumps({
        "systems": [{"system_id": "authoritative-trex", "replicas": replicas}]
    }), encoding="utf-8")
    return path


class AutomaticSamplingTests(unittest.TestCase):
    def test_hydrogen_bond_runtime_uses_candidate_dimension(self):
        dimensions = self._reference_dimensions(frames_per_replica=100)
        dimensions["hydrogen_bond_candidate_planning"] = {
            "status": "complete",
            "common_candidate_count": 129_280,
            "mean_candidate_count_per_replica": 129_280.0,
        }
        multiplier, basis = _runtime_workload_multiplier(
            sampling_profile("hydrogen_bond_discovery"), dimensions
        )
        self.assertAlmostEqual(multiplier, 2.0 ** 0.5)
        self.assertEqual(basis["implicit_candidate_count"], 129_280)
        self.assertNotIn("observed_candidate_count", basis)
        self.assertEqual(
            basis["dimension"],
            "lazy spatial donor/acceptor endpoint proxy from the square root "
            "of the implicit common candidate universe",
        )

    def test_feature_aware_pca_basis_plan_separates_basis_and_projection(self):
        trace = plan_cartesian_pca_basis(1_422, [10_000, 10_000, 10_000])
        self.assertEqual(trace["solver_method"], "dense_covariance_v1")
        self.assertEqual(trace["basis_frame_count"], 1_500)
        self.assertEqual(trace["projection_frame_count"], 30_000)
        self.assertEqual(
            trace["basis_frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 20},
        )

        common_heavy = plan_cartesian_pca_basis(11_226, [10_000, 10_000, 10_000])
        self.assertEqual(
            common_heavy["solver_method"], "randomized_truncated_svd_v1"
        )
        self.assertEqual(common_heavy["basis_frame_count"], 1_500)
        self.assertEqual(common_heavy["projection_frame_count"], 30_000)
        self.assertLessEqual(
            common_heavy["estimated_sample_matrix_elements"],
            common_heavy["maximum_sample_matrix_elements"],
        )
        self.assertGreater(
            common_heavy["estimated_dense_covariance_bytes_float64"],
            trace["estimated_dense_covariance_bytes_float64"],
        )
        self.assertEqual(
            common_heavy["randomized_solver_subspace_policy"],
            "fixed_leading_subspace",
        )
        self.assertEqual(common_heavy["randomized_solver_oversampling"], 12)

        bounded = plan_cartesian_pca_basis(7_950, [100])
        self.assertEqual(
            bounded["randomized_solver_subspace_policy"],
            "full_bounded_sample_space",
        )
        self.assertEqual(bounded["randomized_solver_oversampling"], 90)
        self.assertEqual(bounded["randomized_solver_subspace_size"], 100)

    def test_feature_aware_pca_basis_fails_when_minimum_cannot_fit(self):
        with self.assertRaisesRegex(AutomaticSamplingError, "required minimum"):
            plan_cartesian_pca_basis(
                100_000,
                [100, 100, 100],
                maximum_sample_matrix_elements=1_000_000,
            )

    @staticmethod
    def _reference_dimensions(frames_per_replica: int = 12_000):
        replicas = [
            {
                "system_id": "reference",
                "replica_id": f"r{index + 1}",
                "atom_count": REFERENCE_ATOM_COUNT,
                "source_frame_count": frames_per_replica,
                "segments": [{
                    "segment_id": "production",
                    "source_frame_count": frames_per_replica,
                }],
            }
            for index in range(3)
        ]
        return {
            "system_ids": ["reference"],
            "replica_count": 3,
            "maximum_atom_count": REFERENCE_ATOM_COUNT,
            "minimum_atom_count": REFERENCE_ATOM_COUNT,
            "total_source_frame_count": 3 * frames_per_replica,
            "minimum_source_frames_per_replica": frames_per_replica,
            "maximum_source_frames_per_replica": frames_per_replica,
            "replicas": replicas,
        }

    def test_default_plan_accounts_for_every_registry_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_system(Path(temporary), frames_per_replica=2)
            report = automatic_sampling_plan(path, simulation_kind="unbiased_md")
            self.assertEqual(len(report["method_plans"]), 58)
        self.assertEqual(
            len({row["module_id"] for row in report["method_plans"]}), 58
        )

    def test_assigns_method_ceiling_and_balanced_sampling(self):
        dimensions = self._reference_dimensions()
        plans = {
            module_id: _module_plan(
                sampling_profile(module_id),
                dimensions,
                b_vs_2b=False,
                replica_diagnostics=False,
                target_wall_seconds=14_400.0,
                time_safety_factor=1.5,
            )
            for module_id in (
                "replica_rmsd_rg", "dccm", "individual_pca",
                "solvent_accessible_surface_area",
            )
        }
        self.assertTrue(plans["replica_rmsd_rg"]["subsampling_triggered"])
        self.assertIn("no random draw", plans["replica_rmsd_rg"]["sampling_strategy"])
        self.assertEqual(
            plans["individual_pca"]["frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 3},
        )
        self.assertEqual(plans["individual_pca"]["selected_frame_count"], 12_000)
        self.assertTrue(plans["dccm"]["subsampling_triggered"])
        self.assertEqual(plans["dccm"]["selected_frame_count"], 18_000)
        self.assertEqual(plans["dccm"]["resolved_maximum_frames_per_replica"], 6_000)
        self.assertEqual(
            plans["solvent_accessible_surface_area"]["frame_selection"],
            {"mode": "integer_stride_per_replica_v1", "stride": 12},
        )
        self.assertEqual(plans["solvent_accessible_surface_area"]["selected_frame_count"], 3_000)
        self.assertEqual(
            plans["solvent_accessible_surface_area"]["resource_time_budget"]
            ["target_wall_hours"],
            4.0,
        )
        self.assertLessEqual(
            plans["solvent_accessible_surface_area"]["resource_time_budget"]
            ["estimated_selected_wall_hours"],
            4.0,
        )
        self.assertFalse(plans["individual_pca"]["replica_diagnostics"]["recommended"])

    def test_owner_flags_are_explicit_and_downstream_inherits(self):
        dimensions = self._reference_dimensions()
        water = _module_plan(
                sampling_profile("water_mediated_hydrogen_bond_networks"),
                dimensions,
                b_vs_2b=True,
                replica_diagnostics=True,
                target_wall_seconds=14_400.0,
                time_safety_factor=1.5,
        )
        self.assertEqual(water["b_vs_2b"]["status"], "planned")
        self.assertEqual(water["b_vs_2b"]["doubled_maximum_total_frames"], 12_000)
        self.assertEqual(water["b_vs_2b"]["doubled_maximum_frames_per_replica"], 4_000)
        self.assertEqual(water["replica_diagnostics"]["status"], "optional_exploratory")
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_system(Path(temporary))
            report = automatic_sampling_plan(
                path,
                simulation_kind="unbiased_md",
                module_ids=["pca_fes_basins"],
                b_vs_2b=True,
                replica_diagnostics=True,
            )
        plans = {row["module_id"]: row for row in report["method_plans"]}
        self.assertEqual(plans["pca_fes_basins"]["inherited_from"], "common_pca")
        self.assertFalse(plans["pca_fes_basins"]["independent_resampling_allowed"])
        self.assertEqual(
            plans["pca_fes_basins"]["postprocessing_limits"]
            ["silhouette_focal_observation_ceiling"],
            1_000,
        )

    def test_runtime_pilots_are_small_method_specific_and_size_scaled(self):
        reference = self._reference_dimensions()
        common = _module_plan(
            sampling_profile("common_pca"),
            reference,
            b_vs_2b=False,
            replica_diagnostics=False,
            target_wall_seconds=14_400.0,
            time_safety_factor=1.5,
        )
        self.assertEqual(common["technical_pilot_frames_per_replica"], 50)
        self.assertIn("not a scientific minimum", common["technical_pilot_role"])

        large = self._reference_dimensions()
        large["maximum_atom_count"] = 10 * REFERENCE_ATOM_COUNT
        large["minimum_atom_count"] = 10 * REFERENCE_ATOM_COUNT
        for replica in large["replicas"]:
            replica["atom_count"] = 10 * REFERENCE_ATOM_COUNT
        large_common = _module_plan(
            sampling_profile("common_pca"),
            large,
            b_vs_2b=False,
            replica_diagnostics=False,
            target_wall_seconds=14_400.0,
            time_safety_factor=1.5,
        )
        self.assertEqual(large_common["technical_pilot_frames_per_replica"], 5)

    def test_configured_campaign_envelope_replaces_per_method_allowances(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_system(Path(temporary))
            report = automatic_sampling_plan(
                path,
                simulation_kind="unbiased_md",
                module_ids=["common_pca", "solvent_accessible_surface_area"],
                target_wall_seconds=360.0,
                campaign_execution={
                    "maximum_parallel_cpus": 1,
                    "maximum_hours_per_cpu": 0.1,
                    "maximum_memory_gib": 64.0,
                    "planning_utilization": 0.85,
                    "pilot_budget_fraction": 0.05,
                },
            )
        campaign = report["campaign_resource_plan"]
        self.assertEqual(campaign["raw_capacity_cpu_hours"], 0.1)
        self.assertEqual(campaign["planning_scope"], "enabled direct trajectory estimators")
        plans = {row["module_id"]: row for row in report["method_plans"]}
        self.assertIn("campaign_resource_allocation", plans["common_pca"])
        self.assertTrue(any(
            plans[module_id]["subsampling_triggered"]
            for module_id in plans
        ))

    def test_structural_qc_uses_one_cache_backed_worker_per_replica(self):
        dimensions = self._reference_dimensions(frames_per_replica=12_000)
        plan = _campaign_direct_resource_plan(
            dimensions,
            ["structural_integrity_qc"],
            {
                "maximum_parallel_cpus": 8,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 185,
                "planning_utilization": 0.85,
                "pilot_budget_fraction": 0.05,
                "coordinate_cache": "auto",
            },
            time_safety_factor=1.5,
        )
        row = plan["tasks"][0]
        self.assertEqual(
            row["parallel_execution_model"],
            "replica_worker_exact_global_reducer_v1",
        )
        self.assertEqual(row["intrinsic_cpu_cap"], 3)
        self.assertEqual(row["effective_cpu_cap"], 3)
        self.assertEqual(row["parallel_worker_count"], 3)
        self.assertEqual(row["estimated_peak_memory_gib_per_parallel_worker"], 5.0)
        self.assertEqual(row["estimated_peak_memory_gib"], 15.0)
        self.assertAlmostEqual(
            row["estimated_wall_hours_at_effective_cpu_cap"],
            row["estimated_cpu_hours"] / 3.0,
        )

    def test_structural_qc_parallelism_respects_aggregate_memory(self):
        dimensions = self._reference_dimensions(frames_per_replica=12_000)
        plan = _campaign_direct_resource_plan(
            dimensions,
            ["structural_integrity_qc"],
            {
                "maximum_parallel_cpus": 8,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 7.5,
                "planning_utilization": 0.85,
                "pilot_budget_fraction": 0.05,
                "coordinate_cache": "auto",
            },
            time_safety_factor=1.5,
        )
        row = plan["tasks"][0]
        self.assertEqual(row["effective_cpu_cap"], 3)
        self.assertEqual(row["parallel_worker_count"], 3)
        self.assertEqual(
            row["active_parallel_workers_at_selected_observations"], 1
        )
        self.assertEqual(
            row["parallel_node_layout_at_selected_observations"][
                "worker_wave_count"
            ],
            3,
        )
        self.assertEqual(
            row["estimated_peak_memory_gib_at_selected_observations"], 5.0
        )

    def test_experimental_replica_reducers_are_priced_by_active_workers(self):
        dimensions = self._reference_dimensions(frames_per_replica=1_000)
        plan = _campaign_direct_resource_plan(
            dimensions,
            ["multivalent_molecular_bridges", "nucleic_acid_structure"],
            {
                "maximum_parallel_cpus": 8,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 185,
                "planning_utilization": 0.85,
                "pilot_budget_fraction": 0.05,
                "memory_safety_factor": 1.25,
                "coordinate_cache": "auto",
            },
            time_safety_factor=1.5,
        )
        rows = {row["module_id"]: row for row in plan["tasks"]}
        for module_id in (
            "multivalent_molecular_bridges", "nucleic_acid_structure",
        ):
            row = rows[module_id]
            self.assertEqual(row["parallel_worker_count"], 3)
            self.assertEqual(row["effective_cpu_cap"], 3)
            self.assertEqual(
                row["parallel_execution_model"],
                "replica_worker_exact_global_reducer_v1",
            )
            self.assertAlmostEqual(
                row["estimated_wall_hours_at_effective_cpu_cap"],
                row["estimated_cpu_hours"] / 3.0,
            )

    def test_censored_only_memory_does_not_invent_observation_scaling(self):
        dimensions = self._reference_dimensions(frames_per_replica=1000)
        measured = {
            "structural_integrity_qc": {
                "catalog_sha256": "a" * 64,
                "conservative_cpu_seconds_per_frame": 1.0,
                "maximum_resident_memory_mib": 700.0,
                "maximum_measured_selected_frame_count": 0,
                "maximum_measured_observation_count": 0,
                "measurement_count": 1,
                "complete_measurement_count": 0,
                "censored_timeout_count": 1,
                "calibration_evidence_status": "censored_lower_bound_only",
            }
        }
        plan = _campaign_direct_resource_plan(
            dimensions,
            ["structural_integrity_qc"],
            {
                "maximum_parallel_cpus": 1,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 185,
                "planning_utilization": 0.85,
                "pilot_budget_fraction": 0.05,
            },
            time_safety_factor=1.5,
            measured_calibrations=measured,
        )
        row = plan["tasks"][0]
        self.assertIsNone(row["measured_memory_cost_model"])
        self.assertLessEqual(
            row["estimated_peak_memory_gib_at_selected_observations"], 5.0
        )

    def test_measured_memory_model_keeps_system_size_scaling(self):
        dimensions = self._reference_dimensions(frames_per_replica=1000)
        dimensions["maximum_atom_count"] = REFERENCE_ATOM_COUNT // 4
        measured = {
            "replica_rmsd_rg": {
                "catalog_sha256": "b" * 64,
                "conservative_cpu_seconds_per_frame": 0.1,
                "maximum_resident_memory_mib": 8192.0,
                "maximum_completed_resident_memory_mib": 8192.0,
                "maximum_measured_selected_frame_count": 3000,
                "maximum_measured_observation_count": 3000,
                "measurement_count": 2,
                "complete_measurement_count": 2,
                "censored_timeout_count": 0,
                "calibration_evidence_status": "completed_execution",
                "memory_replacement_qualified": True,
            }
        }
        plan = _campaign_direct_resource_plan(
            dimensions,
            ["replica_rmsd_rg"],
            {
                "maximum_parallel_cpus": 1,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 185,
                "planning_utilization": 0.85,
                "pilot_budget_fraction": 0.05,
            },
            time_safety_factor=1.5,
            measured_calibrations=measured,
        )
        row = plan["tasks"][0]
        self.assertAlmostEqual(
            row["measured_memory_cost_model"]["calibration_memory_gib"],
            4.0,
            places=3,
        )

    def test_rejects_unknown_simulation_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_system(Path(temporary), frames_per_replica=2)
            with self.assertRaises(AutomaticSamplingError):
                automatic_sampling_plan(path, simulation_kind="unknown")


if __name__ == "__main__":
    unittest.main()
