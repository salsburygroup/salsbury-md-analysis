import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.resource_calibrations import (
    SCHEMA, TIMEOUT_SCHEMA, build_resource_calibration_catalog,
    load_resource_calibration_catalog, redact_resource_calibration_catalog,
)
from salsbury_md_analysis.planner_calibration_models import (
    MODEL_SCHEMA, validate_size_length_models,
)


class ResourceCalibrationTests(unittest.TestCase):
    def test_distributed_v5_catalog_and_model_are_valid_and_path_redacted(self):
        root = Path(__file__).resolve().parents[1]
        model_path = root / "profiles" / "apollo_planner_size_length_cpu_models_v1.json"
        catalog_path = root / "profiles" / "apollo_measured_resource_calibrations_v5.json"
        model = validate_size_length_models(json.loads(model_path.read_text()))
        resolved = load_resource_calibration_catalog(catalog_path)
        self.assertEqual(set(model["models"]), {
            "structural_integrity_qc",
            "hydrogen_bond_discovery",
            "ion_atmosphere",
        })
        self.assertTrue(all(
            resolved[module_id]["size_length_cpu_model"][
                "heldout_validation_passed"
            ]
            for module_id in model["models"]
        ))
        rendered = model_path.read_text() + catalog_path.read_text()
        self.assertNotIn("/deac/", rendered)
        self.assertNotIn("/private/tmp/", rendered)
        self.assertNotIn("/Users/", rendered)

    def test_catalog_carries_validated_size_length_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text('{"technical_status":"complete"}\n', encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete",
                "module_id": "structural_integrity_qc",
                "report_path": str(report),
                "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 10,
                    "symmetry_expanded_observations": 10,
                    "execution_resources": {
                        "total_cpu_seconds": 1.0,
                        "wall_seconds": 1.0,
                        "maximum_resident_memory_mib": 10.0,
                    },
                },
            }), encoding="utf-8")
            model = root / "model.json"
            model.write_text(json.dumps({
                "model_schema": MODEL_SCHEMA,
                "technical_status": "complete",
                "scientific_status": "runtime evidence only",
                "source_evidence_sha256": "a" * 64,
                "models": {"structural_integrity_qc": {
                    "module_id": "structural_integrity_qc",
                    "intercept_cpu_seconds": 1.0,
                    "cpu_seconds_per_topology_atom_source_frame": 0.001,
                    "cpu_seconds_per_selected_work_unit": 0.01,
                    "selected_work_unit": "topology_atom_selected_frames_v1",
                    "planning_proxy": "topology_atoms_per_selected_frame_v1",
                    "selected_work_units_per_proxy_unit": 1.0,
                    "residual_safety_factor": 1.5,
                    "heldout_validation_passed": True,
                    "training_point_count": 6,
                    "heldout_point_count": 3,
                    "measured_ranges": {},
                    "extrapolation_policy": "pilot outside measured range",
                }},
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog(
                [sidecar], work_model_paths=[model]
            )
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(catalog_path)[
                "structural_integrity_qc"
            ]
        self.assertEqual(
            resolved["size_length_cpu_model"]["planning_proxy"],
            "topology_atoms_per_selected_frame_v1",
        )

    def test_spatial_hbond_work_is_aggregated_separately_from_dense_universe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text('{"technical_status":"complete"}\n', encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete",
                "module_id": "hydrogen_bond_discovery",
                "report_path": str(report),
                "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 12,
                    "symmetry_expanded_observations": 12,
                    "conceptual_candidate_frame_count": 16_512_252,
                    "spatial_neighbor_pair_count": 25_164,
                    "explicit_geometry_evaluation_count": 21_095,
                    "present_event_count": 7_322,
                    "maximum_spatial_endpoint_count_per_system": 2_348,
                    "execution_resources": {
                        "total_cpu_seconds": 301.46992,
                        "wall_seconds": 213.152388,
                        "maximum_resident_memory_mib": 190.86328125,
                    },
                },
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog([sidecar])
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(catalog_path)[
                "hydrogen_bond_discovery"
            ]
        self.assertEqual(resolved["runtime_work_unit"], "spatial_neighbor_pairs_v1")
        self.assertEqual(
            resolved["maximum_measured_spatial_neighbor_pair_count"], 25_164
        )
        self.assertEqual(
            resolved["maximum_measured_spatial_endpoint_count_per_system"], 2_348
        )
        self.assertAlmostEqual(
            resolved["conservative_spatial_neighbor_pairs_per_selected_frame"],
            25_164 / 12,
        )
        self.assertAlmostEqual(
            resolved["conservative_cpu_seconds_per_spatial_neighbor_pair"],
            301.46992 / 25_164,
        )

    def test_hash_bound_cpu_memory_and_coverage_are_aggregated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text('{"technical_status":"complete"}\n', encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete", "module_id": "ion_atmosphere",
                "report_path": str(report), "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 20,
                    "symmetry_expanded_observations": 20,
                    "execution_resources": {
                        "total_cpu_seconds": 10.0, "wall_seconds": 11.0,
                        "maximum_resident_memory_mib": 100.0,
                        "computer_hostname": "test", "platform": "test",
                        "requested_cpu_count": 1,
                    },
                },
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog([sidecar])
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(catalog_path)["ion_atmosphere"]
        self.assertEqual(resolved["maximum_measured_selected_frame_count"], 20)
        self.assertEqual(resolved["maximum_resident_memory_mib"], 100.0)
        self.assertAlmostEqual(resolved["conservative_cpu_seconds_per_frame"], 0.5)
        self.assertEqual(catalog["catalog_schema"], SCHEMA)
        self.assertEqual(resolved["complete_measurement_count"], 1)
        self.assertEqual(resolved["censored_timeout_count"], 0)
        self.assertFalse(resolved["memory_replacement_qualified"])
        self.assertEqual(
            resolved["memory_replacement_policy"],
            "retain_legacy_baseline_and_use_measurement_as_lower_bound",
        )

    def test_two_complete_measurements_qualify_memory_to_replace_legacy_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecars = []
            for index, memory in enumerate((100.0, 120.0), start=1):
                report = root / f"report-{index}.json"
                report.write_text(
                    '{"technical_status":"complete"}\n', encoding="utf-8"
                )
                digest = hashlib.sha256(report.read_bytes()).hexdigest()
                sidecar = root / f"report-{index}.summary.json"
                sidecar.write_text(json.dumps({
                    "technical_status": "complete",
                    "module_id": "ion_atmosphere",
                    "report_path": str(report),
                    "report_sha256": digest,
                    "resource_evidence": {
                        "selected_source_physical_frames": index * 20,
                        "symmetry_expanded_observations": index * 20,
                        "execution_resources": {
                            "total_cpu_seconds": float(index * 10),
                            "wall_seconds": float(index * 11),
                            "maximum_resident_memory_mib": memory,
                        },
                    },
                }), encoding="utf-8")
                sidecars.append(sidecar)
            catalog = build_resource_calibration_catalog(sidecars)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(
                catalog_path
            )["ion_atmosphere"]
        self.assertTrue(resolved["memory_replacement_qualified"])
        self.assertEqual(resolved["maximum_completed_resident_memory_mib"], 120.0)
        self.assertEqual(resolved["minimum_measured_observation_count"], 20)
        self.assertEqual(resolved["maximum_measured_observation_count"], 40)

    def test_timeout_is_censored_lower_bound_not_completed_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeout = root / "timeout.json"
            timeout.write_text(json.dumps({
                "evidence_schema": TIMEOUT_SCHEMA,
                "technical_status": "timeout",
                "scientific_status": "not evaluated",
                "module_id": "ion_atmosphere",
                "scheduler_job_id": 123,
                "selected_source_physical_frames": 60_000,
                "symmetry_expanded_observations": 60_000,
                "elapsed_seconds": 28_800.0,
                "allocated_cpu_count": 1,
                "maximum_resident_memory_mib": 300.0,
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog(
                [], timeout_records=[timeout]
            )
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(
                catalog_path, censored_timeout_safety_factor=2.0
            )["ion_atmosphere"]
        self.assertEqual(resolved["complete_measurement_count"], 0)
        self.assertEqual(resolved["censored_timeout_count"], 1)
        self.assertEqual(resolved["maximum_measured_selected_frame_count"], 0)
        self.assertEqual(resolved["maximum_timeout_target_frame_count"], 60_000)
        self.assertEqual(
            resolved["calibration_evidence_status"], "censored_lower_bound_only"
        )
        self.assertAlmostEqual(
            resolved["maximum_censored_cpu_seconds_per_frame_lower_bound"],
            28_800.0 / 60_000,
        )
        self.assertAlmostEqual(
            resolved["conservative_cpu_seconds_per_frame"],
            2.0 * 28_800.0 / 60_000,
        )
        self.assertEqual(resolved["maximum_resident_memory_mib"], 300.0)
        self.assertEqual(
            resolved["maximum_observed_resident_memory_mib_all_records"],
            300.0,
        )

    def test_parallel_timeout_memory_is_not_replayed_as_per_worker_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timeout = root / "parallel-timeout.json"
            timeout.write_text(json.dumps({
                "evidence_schema": TIMEOUT_SCHEMA,
                "technical_status": "timeout",
                "module_id": "pooled_rmsf",
                "selected_source_physical_frames": 30_000,
                "symmetry_expanded_observations": 30_000,
                "elapsed_seconds": 1_800.0,
                "allocated_cpu_count": 12,
                "maximum_resident_memory_mib": 116_000.0,
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog(
                [], timeout_records=[timeout]
            )
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(catalog_path)[
                "pooled_rmsf"
            ]
        self.assertEqual(resolved["maximum_resident_memory_mib"], 0.0)
        self.assertEqual(
            resolved["maximum_observed_resident_memory_mib_all_records"],
            116_000.0,
        )
        self.assertIn(
            "not treated as per-worker",
            resolved["memory_timeout_evidence_policy"],
        )

    def test_completed_and_timeout_rates_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text('{"technical_status":"complete"}\n', encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete", "module_id": "structural_integrity_qc",
                "report_path": str(report), "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 100,
                    "symmetry_expanded_observations": 100,
                    "execution_resources": {
                        "total_cpu_seconds": 100.0, "wall_seconds": 101.0,
                        "maximum_resident_memory_mib": 100.0,
                    },
                },
            }), encoding="utf-8")
            timeout = root / "timeout.json"
            timeout.write_text(json.dumps({
                "evidence_schema": TIMEOUT_SCHEMA,
                "technical_status": "timeout",
                "module_id": "structural_integrity_qc",
                "selected_source_physical_frames": 100,
                "symmetry_expanded_observations": 100,
                "elapsed_seconds": 200.0,
                "allocated_cpu_count": 1,
            }), encoding="utf-8")
            catalog = build_resource_calibration_catalog(
                [sidecar], timeout_records=[timeout]
            )
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            resolved = load_resource_calibration_catalog(
                catalog_path, censored_timeout_safety_factor=1.5
            )["structural_integrity_qc"]
        self.assertEqual(resolved["complete_measurement_count"], 1)
        self.assertEqual(resolved["censored_timeout_count"], 1)
        self.assertEqual(resolved["maximum_measured_selected_frame_count"], 100)
        self.assertAlmostEqual(resolved["conservative_cpu_seconds_per_frame"], 3.0)

    def test_base_catalog_can_be_extended_without_losing_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete", "module_id": "old_module",
                "report_path": str(report), "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 10,
                    "symmetry_expanded_observations": 10,
                    "execution_resources": {
                        "total_cpu_seconds": 10.0, "wall_seconds": 10.0,
                        "maximum_resident_memory_mib": 10.0,
                    },
                },
            }), encoding="utf-8")
            base = build_resource_calibration_catalog([sidecar])
            base_path = root / "base.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            timeout = root / "timeout.json"
            timeout.write_text(json.dumps({
                "evidence_schema": TIMEOUT_SCHEMA,
                "technical_status": "timeout",
                "module_id": "new_module",
                "selected_source_physical_frames": 20,
                "symmetry_expanded_observations": 20,
                "elapsed_seconds": 20.0,
                "allocated_cpu_count": 1,
            }), encoding="utf-8")
            extended = build_resource_calibration_catalog(
                [], timeout_records=[timeout], base_catalogs=[base_path]
            )
        self.assertEqual(extended["entry_count"], 2)
        self.assertEqual(extended["complete_execution_count"], 1)
        self.assertEqual(extended["censored_timeout_count"], 1)
        self.assertEqual(extended["base_catalogs"][0]["entry_count"], 1)

    def test_overlapping_base_catalogs_deduplicate_identical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete", "module_id": "dccm",
                "report_path": str(report), "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 10,
                    "symmetry_expanded_observations": 10,
                    "execution_resources": {
                        "total_cpu_seconds": 20.0, "wall_seconds": 21.0,
                        "maximum_resident_memory_mib": 30.0,
                    },
                },
            }), encoding="utf-8")
            first = build_resource_calibration_catalog([sidecar])
            first_path = root / "first.json"
            second_path = root / "second.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second = redact_resource_calibration_catalog(first)
            second_path.write_text(json.dumps(second), encoding="utf-8")
            merged = build_resource_calibration_catalog(
                [], base_catalogs=[first_path, second_path]
            )
        self.assertEqual(merged["entry_count"], 1)
        self.assertEqual(merged["duplicate_evidence_entry_count"], 1)

    def test_redacted_catalog_retains_planner_values_without_private_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            sidecar = root / "report.json.summary.json"
            sidecar.write_text(json.dumps({
                "technical_status": "complete", "module_id": "ion_atmosphere",
                "report_path": str(report), "report_sha256": digest,
                "resource_evidence": {
                    "selected_source_physical_frames": 10,
                    "symmetry_expanded_observations": 10,
                    "execution_resources": {
                        "total_cpu_seconds": 20.0, "wall_seconds": 21.0,
                        "maximum_resident_memory_mib": 30.0,
                        "computer_hostname": "private-node",
                        "requested_cpu_count": 1,
                    },
                },
            }), encoding="utf-8")
            full = build_resource_calibration_catalog([sidecar])
            redacted = redact_resource_calibration_catalog(full)
            redacted_path = root / "redacted.json"
            redacted_path.write_text(json.dumps(redacted), encoding="utf-8")
            resolved = load_resource_calibration_catalog(redacted_path)[
                "ion_atmosphere"
            ]
        rendered = json.dumps(redacted)
        self.assertNotIn(str(root), rendered)
        self.assertNotIn("private-node", rendered)
        self.assertIn(full["entries"][0]["source_sidecar_sha256"], rendered)
        self.assertEqual(resolved["maximum_measured_selected_frame_count"], 10)
        self.assertAlmostEqual(
            resolved["conservative_cpu_seconds_per_frame"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
