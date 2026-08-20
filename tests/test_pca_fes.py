import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.pca_fes import (
    PCAFESAnalysisError,
    build_landscape,
    basin_silhouette_report,
    pca_fes_basins_project,
    select_bin_counts,
)


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _frame(x, y, label):
    return (
        f"4\n{label}\n"
        "C 0 0 0\n"
        "N 1 0 0\n"
        "O 0 1 0\n"
        f"C {x} {y} 0\n"
    )


def _write_ai_project(root: Path) -> Path:
    (root / "reference.pdb").write_text(
        "".join([
            _pdb_atom(1, "C", 0, 0, 0, "C"),
            _pdb_atom(2, "N", 1, 0, 0, "N"),
            _pdb_atom(3, "O", 0, 1, 0, "O"),
            _pdb_atom(4, "CB", 0, 2, 0, "C"),
        ]) + "END\n",
        encoding="utf-8",
    )
    points = [(-2, -2), (-2, -1), (-1, -2), (2, 2), (2, 1), (1, 2)]
    (root / "samples.xyz").write_text(
        "".join(_frame(x, y, f"sample-{index}") for index, (x, y) in enumerate(points)),
        encoding="utf-8",
    )
    system = {"systems": [{"system_id": "ai", "replicas": [{
        "replica_id": "r1", "topology": "reference.pdb", "segments": [{
            "segment_id": "samples", "trajectory": "samples.xyz",
            "sample_axis": {"first_sample_index": 0, "sample_interval": 1},
        }],
    }]}]}
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "ai-fes-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "AI_ENSEMBLE",
        "coordinate_unit": "angstrom",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "reference_system": "ai",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"atom_names": ["C", "N", "O"]},
            "analysis": {"atom_names": ["CB"]},
        },
        "definitions": {
            "common_pca": {
                "alignment_selection": "alignment",
                "analysis_selection": "analysis",
                "minimum_reference_coverage": 1.0,
                "frame_stride": 1,
                "maximum_features": 3,
                "component_count": 2,
                "minimum_evaluated_frames_per_replica": 2,
                "basis_weighting": "frame",
            },
            "pca_fes_basins": {
                "x_component": 1,
                "y_component": 2,
                "bins_x": 6,
                "bins_y": 6,
                "padding_fraction": 0.1,
                "minimum_bin_count": 1,
                "population_block_size_frames": 3,
                "include_partial_final_block": True,
                "maximum_grid_cells": 100,
                "density_estimator": "histogram",
                "smoothing_sigmas_bins": [0.0, 1.0],
                "primary_smoothing_sigma_bins": 0.0,
            },
        },
        "requested_modules": ["common_pca", "pca_fes_basins"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class PCAFESAnalysisTests(unittest.TestCase):
    def test_fes_basin_silhouette_is_secondary_and_handles_one_basin(self):
        report = basin_silhouette_report(
            [(0.0, 0.0), (0.1, 0.0), (10.0, 10.0), (10.1, 10.0)],
            [1, 1, 2, 2],
            maximum_exact_observations=100,
        )
        self.assertEqual(report["status"], "complete")
        self.assertGreater(report["score"], 0.9)
        self.assertFalse(report["defines_fes_basins"])
        one = basin_silhouette_report([(0.0, 0.0), (0.1, 0.0)], [1, 1], 100)
        self.assertEqual(one["status"], "not_calculable")
        self.assertIsNone(one["score"])

    def test_scott_freedman_diaconis_and_rice_bin_axes_independently(self):
        points = [(float(index), float(index * index)) for index in range(1, 101)]
        reports = {
            rule: select_bin_counts(points, rule, 0.0, 2, 100)
            for rule in ("scott", "freedman_diaconis", "rice")
        }
        for rule in ("scott", "freedman_diaconis"):
            report = reports[rule]
            self.assertNotEqual(report["rule_width_x"], report["rule_width_y"])
            self.assertEqual(
                report["raw_bins_x"],
                math.ceil(99.0 / report["rule_width_x"]),
            )
            self.assertEqual(
                report["raw_bins_y"],
                math.ceil((10000.0 - 1.0) / report["rule_width_y"]),
            )
        x_values = [float(index) for index in range(1, 101)]
        x_mean = sum(x_values) / len(x_values)
        expected_scott_x = (
            3.5
            * math.sqrt(sum((value - x_mean) ** 2 for value in x_values) / 100)
            * 100 ** (-1.0 / 3.0)
        )
        self.assertAlmostEqual(reports["scott"]["rule_width_x"], expected_scott_x)
        self.assertEqual(reports["rice"]["bins_x"], reports["rice"]["bins_y"])
        with self.assertRaises(PCAFESAnalysisError):
            select_bin_counts([(1.0, 1.0), (1.0, 2.0)], "scott", 0.0, 2, 20)

    def test_landscape_keeps_occupancy_and_thermodynamic_energy_distinct(self):
        points = [(0.0, 0.0)] * 3 + [(0.1, 0.1)] + [(10.0, 10.0)] * 2
        occupancy = build_landscape(
            points, bins_x=6, bins_y=6, padding_fraction=0.1,
            minimum_bin_count=1, temperature_kelvin=None,
        )
        self.assertEqual(len(occupancy["basins"]), 2)
        occupied = [row for row in occupancy["grid"] if row["count"]]
        self.assertTrue(all("relative_occupancy_score" in row for row in occupied))
        self.assertTrue(all("relative_free_energy_kcal_per_mol" not in row for row in occupied))

        fes = build_landscape(
            points, bins_x=6, bins_y=6, padding_fraction=0.1,
            minimum_bin_count=1, temperature_kelvin=300.0,
        )
        occupied_fes = [row for row in fes["grid"] if row["count"]]
        self.assertAlmostEqual(
            min(row["relative_free_energy_kcal_per_mol"] for row in occupied_fes),
            0.0,
        )

    def test_smoothing_changes_minima_without_changing_raw_histogram_counts(self):
        points = []
        for x, count in enumerate([10, 2, 8, 2, 10]):
            points.extend([(float(x), 0.0)] * count)
            points.extend([(float(x), 1.0)] * count)
        raw = build_landscape(
            points, bins_x=5, bins_y=2, padding_fraction=0.0,
            minimum_bin_count=1, temperature_kelvin=300.0,
            smoothing_sigma_bins=0.0,
        )
        smooth = build_landscape(
            points, bins_x=5, bins_y=2, padding_fraction=0.0,
            minimum_bin_count=1, temperature_kelvin=300.0,
            smoothing_sigma_bins=2.0,
        )
        self.assertEqual(
            [row["count"] for row in raw["grid"]],
            [row["count"] for row in smooth["grid"]],
        )
        self.assertEqual((len(raw["basins"]), len(smooth["basins"])), (3, 1))
        self.assertEqual(len(raw["point_assignments"]), len(points))
        self.assertEqual(len(smooth["point_assignments"]), len(points))

    def test_ai_project_emits_sample_assignments_and_no_fake_fes(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = pca_fes_basins_project(_write_ai_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["landscape_kind"], "nonthermodynamic_relative_occupancy")
        self.assertIsNone(report["temperature_kelvin"])
        self.assertEqual(
            [row["sample_index"] for row in report["frame_assignments"]],
            list(range(6)),
        )
        self.assertEqual(len(report["block_populations"]), 2)
        self.assertEqual(
            [row["smoothing_sigma_bins"] for row in report["smoothing_landscapes"]],
            [0.0, 1.0],
        )
        self.assertEqual(len(report["smoothing_sensitivity"]), 1)
        self.assertIn(report["basin_silhouette"]["status"], {"complete", "not_calculable"})
        self.assertEqual(
            len(report["smoothing_landscapes"][1]["frame_assignments"]), 6
        )
        self.assertEqual(report["issues"][0]["code"], "AI_ENSEMBLE_NOT_THERMODYNAMIC")

    def test_per_system_landscapes_use_shared_basis_grid_and_local_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = _write_ai_project(root)
            system_path = root / "system.json"
            system = json.loads(system_path.read_text(encoding="utf-8"))
            second = json.loads(json.dumps(system["systems"][0]))
            second["system_id"] = "ai-variant"
            second["replicas"][0]["replica_id"] = "r2"
            system["systems"].append(second)
            system_path.write_text(json.dumps(system), encoding="utf-8")
            report = pca_fes_basins_project(project_path)

        surfaces = report["per_system_landscapes"]
        self.assertEqual([row["system_id"] for row in surfaces], ["ai", "ai-variant"])
        pooled_bounds = report["landscape"]["bounds"]
        pooled_centers = [
            (row["x_center_angstrom"], row["y_center_angstrom"])
            for row in report["landscape"]["grid"]
        ]
        for row in surfaces:
            self.assertEqual(row["normalization_scope"], "within_system")
            self.assertTrue(row["common_grid_with_pooled_landscape"])
            self.assertEqual(row["landscape"]["bounds"], pooled_bounds)
            self.assertEqual(row["landscape"]["bounds_source"], "fixed_shared_grid")
            self.assertEqual(
                [
                    (cell["x_center_angstrom"], cell["y_center_angstrom"])
                    for cell in row["landscape"]["grid"]
                ],
                pooled_centers,
            )
            self.assertEqual(sum(cell["count"] for cell in row["landscape"]["grid"]), 6)

    def test_pooled_surface_survives_when_per_system_occupancy_gate_is_not_met(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = _write_ai_project(root)
            system_path = root / "system.json"
            system = json.loads(system_path.read_text(encoding="utf-8"))
            second = json.loads(json.dumps(system["systems"][0]))
            second["system_id"] = "ai-variant"
            second["replicas"][0]["replica_id"] = "r2"
            system["systems"].append(second)
            system_path.write_text(json.dumps(system), encoding="utf-8")
            project = json.loads(project_path.read_text(encoding="utf-8"))
            project["definitions"]["pca_fes_basins"]["minimum_bin_count"] = 2
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = pca_fes_basins_project(project_path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertTrue(all(
            row["technical_status"] == "not_constructed"
            for row in report["per_system_landscapes"]
        ))
        self.assertEqual(
            sum(issue["code"] == "PER_SYSTEM_LANDSCAPE_GATE_NOT_MET" for issue in report["issues"]),
            4,
        )


if __name__ == "__main__":
    unittest.main()
