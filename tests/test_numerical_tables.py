import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.numerical_tables import (
    validate_dccm_matrix, write_dccm_matrix, write_fes_grid,
)
from salsbury_md_analysis.presentation_artifacts import generate_presentation_artifacts


def grid(offset=0.0):
    return {
        "bin_widths_angstrom": {"x": 0.5, "y": 1.0},
        "bounds": {"x_min_angstrom": -0.5, "x_max_angstrom": 0.5,
                   "y_min_angstrom": -1.5, "y_max_angstrom": 1.5},
        "grid": [
            {"x_bin": i, "y_bin": j, "x_center_angstrom": -0.25 + 0.5 * i,
             "y_center_angstrom": -1.0 + j, "count": 3 * i + j,
             "probability": (3 * i + j) / 15.0,
             "surface_count": 3 * i + j + 0.125,
             "surface_probability": 0.12345678901234567,
             "relative_free_energy_kcal_per_mol": offset + 3 * i + j,
             "basin_id": 1 if j else None}
            for i in range(2) for j in range(3)
        ],
        "basins": [],
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_report(root, folder, payload):
    path = root / "results" / folder / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class NumericalTableTests(unittest.TestCase):
    def test_fes_rectangular_coordinates_precision_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.csv"
            landscape = grid()
            before = copy.deepcopy(landscape)
            count = write_fes_grid(path, landscape, {"system_id": "A, B", "smoothing_sigma_bins": 1.0})
            rows = read_csv(path)
            self.assertEqual(count, 6)
            for row, source in zip(rows, landscape["grid"]):
                for key, value in source.items():
                    if value is None:
                        self.assertEqual(row[key], "")
                    else:
                        self.assertEqual(float(row[key]), value)
                self.assertEqual(row["system_id"], "A, B")
                self.assertEqual(float(row["x_bin_width_angstrom"]), 0.5)
                self.assertEqual(float(row["y_min_angstrom"]), -1.5)
            self.assertEqual(landscape, before)
            with self.assertRaises(FileExistsError):
                write_fes_grid(path, landscape, {})

    def test_fes_nonthermodynamic_and_nonfinite_are_not_zero_energy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.csv"
            landscape = grid()
            values = [0.0, None, math.nan, math.inf, -math.inf, 2.5]
            for cell, value in zip(landscape["grid"], values):
                del cell["relative_free_energy_kcal_per_mol"]
                cell["relative_occupancy_score"] = value
            write_fes_grid(path, landscape, {"landscape_kind": "nonthermodynamic_relative_occupancy"})
            rows = read_csv(path)
            self.assertEqual([r["occupancy_score_status"] for r in rows],
                             ["finite", "missing", "nan", "positive_infinity", "negative_infinity", "finite"])
            self.assertTrue(all(r["free_energy_status"] == "not_reported" for r in rows))
            self.assertTrue(all(r["relative_free_energy_kcal_per_mol"] == "" for r in rows))
            self.assertEqual(rows[0]["relative_occupancy_score"], "0.0")
            self.assertEqual(rows[1]["relative_occupancy_score"], "")

    def test_fes_rejects_duplicate_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            for duplicate in (False, True):
                landscape = grid()
                if duplicate:
                    landscape["grid"].append(dict(landscape["grid"][0]))
                else:
                    landscape["grid"].pop(1)
                with self.assertRaises(ValueError):
                    write_fes_grid(Path(temporary) / "bad.csv", landscape, {})

    def test_dccm_all_entries_labels_and_missing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dccm.csv"
            matrix = [[1.0, 0.0, None], [0.0, math.nan, math.inf], [None, -math.inf, 1.0]]
            atoms = [{"chain_id": "A", "residue_name": "GLY", "residue_number": i + 1,
                      "insertion_code": "B", "atom_name": "C,A"} for i in range(3)]
            self.assertEqual(write_dccm_matrix(path, matrix, atoms, "test"), 9)
            rows = read_csv(path)
            self.assertEqual([(int(r["atom_i"]), int(r["atom_j"])) for r in rows],
                             [(i, j) for i in range(3) for j in range(3)])
            self.assertEqual(rows[0]["atom_i_label"], "A:GLY1B:C,A")
            self.assertEqual(rows[1]["correlation"], "0.0")
            self.assertEqual(rows[2]["correlation"], "")
            self.assertEqual([r["value_status"] for r in rows],
                             ["finite", "finite", "missing", "finite", "nan",
                              "positive_infinity", "missing", "negative_infinity", "finite"])

    def test_dccm_rejects_shapes_and_identity_mismatches(self):
        for matrix, atoms in (([], []), ([[1, 0]], []), ([[1]], [{}, {}]), ([[True]], [])):
            with self.subTest(matrix=matrix, atoms=atoms), self.assertRaises(ValueError):
                validate_dccm_matrix(matrix, atoms)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "different dimensions"):
                write_dccm_matrix(Path(temporary) / "bad.csv", [[1.0]], [], "A",
                                  right_matrix=[[1.0, 0.0], [0.0, 1.0]], right_system_id="B")

    def test_dccm_difference_keeps_operand_missing_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "difference.csv"
            write_dccm_matrix(path, [[math.nan]], [], "A",
                              right_matrix=[[0.25]], right_system_id="B")
            row = read_csv(path)[0]
            self.assertEqual(row["left_system_id"], "A")
            self.assertEqual(row["right_system_id"], "B")
            self.assertEqual(row["left_value_status"], "nan")
            self.assertEqual(row["right_value_status"], "finite")
            self.assertEqual(row["left_minus_right"], "")

    def test_all_fes_surfaces_export_once_with_hashes_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variants = [{
                "smoothing_sigma_bins": sigma, "landscape": grid(sigma),
                "per_system_landscapes": [
                    {"system_id": system_id, "technical_status": "complete",
                     "normalization_scope": "within_system",
                     "common_grid_with_pooled_landscape": True, "landscape": grid(sigma + i)}
                    for i, system_id in enumerate(("A/B", "A B"))
                ] + [{"system_id": "empty", "technical_status": "not_constructed"}],
            } for sigma in (1.0, 2.0)]
            report = {
                "module_id": "pca_fes_basins", "technical_status": "complete",
                "primary_smoothing_sigma_bins": 1.0,
                "landscape_kind": "thermodynamic_relative_free_energy", "temperature_kelvin": 300.0,
                "pca_basis": {"x_component": 2, "y_component": 3, "basis_weighting": "frame_pooled"},
                "landscape": variants[0]["landscape"],
                "per_system_landscapes": variants[0]["per_system_landscapes"],
                "smoothing_landscapes": variants,
            }
            source = save_report(root, "conformational-views/global_common_heavy/fes", report)
            before = (source.read_bytes(), source.stat().st_mtime_ns)
            manifest = generate_presentation_artifacts(root)
            tables = [a for a in manifest["artifacts"] if a["artifact_type"] == "table"]
            self.assertEqual(len(tables), 6)  # Two pooled grids and four per-system grids.
            self.assertEqual(len({a["relative_path"] for a in tables}), 6)
            for artifact in tables:
                path = root / "presentation-artifacts" / artifact["relative_path"]
                rows = read_csv(path)
                self.assertEqual(len(rows), 6)
                self.assertEqual(artifact["artifact_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(artifact["source_report_sha256"], [hashlib.sha256(before[0]).hexdigest()])
                self.assertEqual(rows[0]["x_pca_component"], "2")
                self.assertEqual(rows[0]["y_pca_component"], "3")
                self.assertEqual(rows[0]["temperature_kelvin"], "300.0")
                sigma = float(rows[0]["smoothing_sigma_bins"])
                index = 1 if rows[0]["system_id"] == "A B" else 0
                self.assertEqual(float(rows[0]["relative_free_energy_kcal_per_mol"]), sigma + index)
                if artifact["purpose"] == "per_system_fes_grid":
                    self.assertEqual(rows[0]["normalization_scope"], "within_system")
                    self.assertEqual(rows[0]["common_grid_with_pooled_landscape"], "True")
            self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)

    def test_per_system_fes_reports_have_separate_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, system in enumerate(("A+B", "A B")):
                save_report(root, f"per-system/{system}/conformational-views/global_common_heavy/fes",
                            {"module_id": "pca_fes_basins", "technical_status": "complete",
                             "landscape": grid(i), "primary_smoothing_sigma_bins": 1.0})
            manifest = generate_presentation_artifacts(root)
            artifacts = manifest["artifacts"]
            self.assertEqual(len({a["relative_path"] for a in artifacts}), len(artifacts))
            tables = [a for a in artifacts if a["artifact_type"] == "table"]
            self.assertEqual(len(tables), 2)
            for table in tables:
                rows = read_csv(root / "presentation-artifacts" / table["relative_path"])
                self.assertEqual(rows[0]["system_id"], table["context"]["system_id"])
                self.assertEqual(rows[0]["normalization_scope"], "within_system")

    def test_full_dccm_difference_not_limited_to_top_50(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = [[1.0 if i == j else (i + j) / 20 for j in range(9)] for i in range(9)]
            right = [[1.0 if i == j else -(i + j) / 20 for j in range(9)] for i in range(9)]
            source = save_report(root, "dccm", {
                "module_id": "dccm", "technical_status": "complete",
                "systems": [
                    {"system_id": "A", "frame_pooled_dccm": {"matrix": left}},
                    {"system_id": "B", "frame_pooled_dccm": {"matrix": right}},
                ],
            })
            original = source.read_bytes()
            manifest = generate_presentation_artifacts(root)
            tables = {a["purpose"] + ":" + a["context"].get("system_id", ""):
                      read_csv(root / "presentation-artifacts" / a["relative_path"])
                      for a in manifest["artifacts"] if a["artifact_type"] == "table"}
            self.assertEqual(len(tables["pairwise_difference:"]), 50)
            full = tables["pairwise_difference_matrix:"]
            self.assertEqual(len(full), 81)
            self.assertEqual(len(tables["system_matrix:A"]), 81)
            for row in full:
                i, j = int(row["atom_i"]), int(row["atom_j"])
                self.assertEqual(float(row["left_minus_right"]), left[i][j] - right[i][j])
            self.assertEqual(source.read_bytes(), original)

    def test_presentation_rejects_inconsistent_matrix_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_report(root, "dccm", {"module_id": "dccm", "technical_status": "complete",
                "systems": [
                    {"system_id": "A", "frame_pooled_dccm": {"matrix": [[1.0]]}},
                    {"system_id": "B", "frame_pooled_dccm": {"matrix": [[1.0, 0.0], [0.0, 1.0]]}},
                ]})
            with self.assertRaisesRegex(ValueError, "different dimensions"):
                generate_presentation_artifacts(root)


if __name__ == "__main__":
    unittest.main()
