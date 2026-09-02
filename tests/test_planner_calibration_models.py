import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.planner_calibration_models import (
    PlannerCalibrationModelError,
    fit_size_length_models,
    validate_runtime_holdouts,
    validate_size_length_models,
)


def _evidence_points():
    rows = []
    definitions = {
        "structural_integrity_qc": ("topology_atom_frame_count", 1.0),
        "hydrogen_bond_discovery": ("spatial_neighbor_pair_count", 2.0),
        "ion_atmosphere": ("ion_target_minimum_image_pair_count", 3.0),
    }
    for module_id, (work_field, work_multiplier) in definitions.items():
        for atom_count, label in ((10, "small"), (20, "middle"), (40, "large")):
            for selected_frames in (5, 10, 20):
                source_frames = 100
                topology_atom_frames = atom_count * selected_frames
                work = int(work_multiplier * topology_atom_frames)
                row = {
                    "point_id": f"{module_id}-{label}-{selected_frames}",
                    "label": label,
                    "module_id": module_id,
                    "topology_atom_count": atom_count,
                    "source_frame_count": source_frames,
                    "selected_source_physical_frames": selected_frames,
                    "topology_atom_source_frame_count": atom_count * source_frames,
                    "topology_atom_frame_count": topology_atom_frames,
                    "total_cpu_seconds": 1.0 + 0.0001 * atom_count * source_frames + 0.01 * work,
                    "stderr_nonempty": False,
                    work_field: work,
                }
                if module_id == "hydrogen_bond_discovery":
                    row["maximum_spatial_endpoint_count_per_system"] = atom_count
                rows.append(row)
    return rows


class PlannerCalibrationModelTests(unittest.TestCase):
    def test_fit_reserves_middle_system_and_passes_heldout_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "evidence.json"
            source.write_text(json.dumps({
                "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
                "technical_status": "complete",
                "unexpected_error_count": 0,
                "points": _evidence_points(),
            }), encoding="utf-8")
            result = fit_size_length_models(source)
        self.assertEqual(set(result["models"]), {
            "structural_integrity_qc",
            "hydrogen_bond_discovery",
            "ion_atmosphere",
        })
        for model in result["models"].values():
            self.assertTrue(model["heldout_validation_passed"])
            self.assertEqual(model["heldout_topology_atom_count"], 20)
            self.assertLessEqual(
                model["maximum_heldout_observed_to_planning_upper_ratio"], 1.0
            )

    def test_validation_rejects_failed_heldout_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "evidence.json"
            source.write_text(json.dumps({
                "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
                "technical_status": "complete",
                "unexpected_error_count": 0,
                "points": _evidence_points(),
            }), encoding="utf-8")
            result = fit_size_length_models(source)
        result["models"]["ion_atmosphere"]["heldout_validation_passed"] = False
        with self.assertRaises(PlannerCalibrationModelError):
            validate_size_length_models(result)

    def test_independent_runtime_holdout_uses_precoordinate_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "evidence.json"
            source.write_text(json.dumps({
                "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
                "technical_status": "complete",
                "unexpected_error_count": 0,
                "points": _evidence_points(),
            }), encoding="utf-8")
            model = fit_size_length_models(source)
        holdout = {
            "holdout_schema": "salsbury-planner-runtime-holdouts-v1",
            "technical_status": "complete",
            "unexpected_error_count": 0,
            "content_sha256": "a" * 64,
            "points": [{
                "point_id": "external-hbond",
                "module_id": "hydrogen_bond_discovery",
                "source_topology_atom_frame_count": 4_000,
                "selected_source_physical_frames": 20,
                "selected_work_proxy_count_per_frame": 40,
                "observed_selected_work_units": 1,
                "observed_total_cpu_seconds": 10.0,
                "stderr_nonempty": False,
                "report_sha256": "b" * 64,
                "project_manifest_sha256": "c" * 64,
                "input_content_signature_sha256": "d" * 64,
                "contract_signature_sha256": "e" * 64,
            }],
        }
        accepted = validate_runtime_holdouts(model, holdout)
        self.assertTrue(accepted["all_holdouts_passed"])
        self.assertFalse(accepted["prediction_coordinate_data_used"])
        self.assertFalse(
            accepted["prediction_dense_candidate_universe_materialized"]
        )
        self.assertEqual(
            accepted["points"][0]["estimated_selected_work_units"], 1600.0
        )

    def test_independent_runtime_holdout_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "evidence.json"
            source.write_text(json.dumps({
                "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
                "technical_status": "complete",
                "unexpected_error_count": 0,
                "points": _evidence_points(),
            }), encoding="utf-8")
            model = fit_size_length_models(source)
        holdout = {
            "holdout_schema": "salsbury-planner-runtime-holdouts-v1",
            "technical_status": "complete",
            "unexpected_error_count": 0,
            "points": [{
                "point_id": "too-slow",
                "module_id": "hydrogen_bond_discovery",
                "source_topology_atom_frame_count": 4_000,
                "selected_source_physical_frames": 20,
                "selected_work_proxy_count_per_frame": 40,
                "observed_total_cpu_seconds": 1_000_000.0,
                "stderr_nonempty": False,
                "report_sha256": "b" * 64,
                "project_manifest_sha256": "c" * 64,
                "input_content_signature_sha256": "d" * 64,
                "contract_signature_sha256": "e" * 64,
            }],
        }
        with self.assertRaisesRegex(
            PlannerCalibrationModelError, "exceeded planning upper bound"
        ):
            validate_runtime_holdouts(model, holdout)


if __name__ == "__main__":
    unittest.main()
