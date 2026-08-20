import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.pca import (
    _mapping_sets,
    common_pca_project,
    common_pca_project_safe,
    individual_pca_project,
)
from salsbury_md_analysis.atom_mapping import AtomRecord


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _frame(cb_x, label):
    return (
        f"4\n{label}\n"
        "C 0 0 0\n"
        "N 1 0 0\n"
        "O 0 1 0\n"
        f"C {cb_x} 2 0\n"
    )


def _write_project(root: Path, common: bool, weighting: str = "replica_equal", maximum_features: int = 3) -> Path:
    atoms = [
        _pdb_atom(1, "C", 0, 0, 0, "C"),
        _pdb_atom(2, "N", 1, 0, 0, "N"),
        _pdb_atom(3, "O", 0, 1, 0, "O"),
        _pdb_atom(4, "CB", 0, 2, 0, "C"),
    ]
    (root / "reference.pdb").write_text("".join(atoms) + "END\n", encoding="utf-8")
    (root / "short.xyz").write_text(
        _frame(-1, "short-0") + _frame(1, "short-1"), encoding="utf-8"
    )
    systems = [{
        "system_id": "short",
        "replicas": [{
            "replica_id": "r1",
            "topology": "reference.pdb",
            "segments": [{
                "segment_id": "s1",
                "trajectory": "short.xyz",
                "timing": {"first_frame_time": 5, "frame_interval": 2, "unit": "ps"},
            }],
        }],
    }]
    if common:
        (root / "long.xyz").write_text(
            "".join(_frame(10, f"long-{index}") for index in range(4)),
            encoding="utf-8",
        )
        systems.append({
            "system_id": "long",
            "replicas": [{
                "replica_id": "r1",
                "topology": "reference.pdb",
                "segments": [{
                    "segment_id": "s1",
                    "trajectory": "long.xyz",
                    "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                }],
            }],
        })
    (root / "system.json").write_text(json.dumps({"systems": systems}), encoding="utf-8")
    module_id = "common_pca" if common else "individual_pca"
    definition = {
        "alignment_selection": "alignment",
        "analysis_selection": "analysis",
        "minimum_reference_coverage": 1.0,
        "frame_stride": 1,
        "maximum_features": maximum_features,
        "component_count": 1,
        "minimum_evaluated_frames_per_replica": 2,
    }
    if common:
        definition["basis_weighting"] = weighting
    project = {
        "project_id": "pca-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "reference_system": "short",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"atom_names": ["C", "N", "O"]},
            "analysis": {"atom_names": ["CB"]},
        },
        "definitions": {module_id: definition},
        "requested_modules": [module_id],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class PCAAnalysisTests(unittest.TestCase):
    def test_global_common_analysis_uses_intersection_while_alignment_keeps_gate(self):
        reference = (
            AtomRecord(0, 1, "P", "", "DA", "A", 1, "", "P"),
            AtomRecord(1, 2, "C1'", "", "DA", "A", 1, "", "C"),
            AtomRecord(2, 3, "N6", "", "DA", "A", 1, "", "N"),
        )
        target = (
            AtomRecord(0, 1, "P", "", "DG", "A", 1, "", "P"),
            AtomRecord(1, 2, "C1'", "", "DG", "A", 1, "", "C"),
        )
        mappings = _mapping_sets(
            reference,
            (target,),
            {
                "alignment": {"atom_names": ["P"]},
                "analysis": {"preset": "solute_heavy"},
            },
            {
                "alignment_selection": "alignment",
                "analysis_selection": "analysis",
                "minimum_reference_coverage": 0.95,
            },
            "position",
            True,
        )
        self.assertEqual(mappings[0]["alignment"].reference_coverage, 1.0)
        self.assertAlmostEqual(mappings[0]["analysis"].reference_coverage, 2.0 / 3.0)
        self.assertEqual(len(mappings[0]["analysis"].reference_indices), 2)

    def test_individual_pca_returns_local_basis_and_physical_time_projections(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = individual_pca_project(_write_project(Path(temporary), False))
        self.assertEqual(report["technical_status"], "complete")
        replica = report["systems"][0]["replicas"][0]
        pca = replica["pca"]
        self.assertAlmostEqual(pca["total_variance_angstrom2"], 1.0)
        component = pca["components"][0]
        self.assertAlmostEqual(component["eigenvalue_angstrom2"], 1.0)
        self.assertAlmostEqual(component["loadings"][0]["loading_x"], 1.0)
        projections = replica["segments"][0]["projections"]
        self.assertEqual([row["time"] for row in projections], [5.0, 7.0])
        self.assertEqual([row["scores_angstrom"][0] for row in projections], [-1.0, 1.0])

    def test_ai_ensemble_pca_reports_sample_indices_not_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root, False)
            system = json.loads((root / "system.json").read_text(encoding="utf-8"))
            segment = system["systems"][0]["replicas"][0]["segments"][0]
            segment.pop("timing")
            segment["sample_axis"] = {"first_sample_index": 10, "sample_interval": 3}
            (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
            project = json.loads(path.read_text(encoding="utf-8"))
            project["sampling_mode"] = "AI_ENSEMBLE"
            project.pop("time_unit")
            path.write_text(json.dumps(project), encoding="utf-8")
            report = individual_pca_project(path)
        self.assertEqual(report["frame_axis_kind"], "sample_index")
        self.assertIsNone(report["time_unit"])
        segment_report = report["systems"][0]["replicas"][0]["segments"][0]
        projections = segment_report["projections"]
        self.assertEqual([row["sample_index"] for row in projections], [10, 13])
        self.assertTrue(all("time" not in row for row in projections))
        self.assertEqual(segment_report["evaluated_axis_range"]["unit"], "sample")

    def test_common_pca_equal_replica_and_frame_weighting_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            equal = common_pca_project(_write_project(root, True, "replica_equal"))
            project_path = _write_project(root, True, "frame")
            frame = common_pca_project(project_path)
        equal_mean = equal["basis"]["pca"]["mean_structure"][0]["mean_x_angstrom"]
        frame_mean = frame["basis"]["pca"]["mean_structure"][0]["mean_x_angstrom"]
        self.assertAlmostEqual(equal_mean, 5.0)
        self.assertAlmostEqual(frame_mean, 40.0 / 6.0)
        self.assertEqual(equal["basis"]["basis_weighting"], "replica_equal")
        contributions = equal["basis"]["replica_contributions"]
        self.assertEqual([row["basis_weight"] for row in contributions], [0.5, 0.5])
        by_system = {row["system_id"]: row for row in equal["systems"]}
        self.assertAlmostEqual(
            by_system["long"]["projection_mean_difference_from_reference_angstrom"][0],
            10.0,
        )

    def test_one_frame_replica_contributes_to_pooled_pca_but_has_no_local_pca(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            individual_path = _write_project(root, False)
            (root / "short.xyz").write_text(_frame(-1, "single"), encoding="utf-8")
            individual = individual_pca_project(individual_path)
            local_replica = individual["systems"][0]["replicas"][0]
            self.assertEqual(individual["technical_status"], "complete")
            self.assertEqual(local_replica["technical_status"], "insufficient_local_frames")
            self.assertIsNone(local_replica["pca"])

            common_path = _write_project(root, True, "frame")
            (root / "long.xyz").write_text(_frame(10, "single"), encoding="utf-8")
            common = common_pca_project(common_path)
        self.assertEqual(common["technical_status"], "complete")
        self.assertEqual(common["basis"]["evaluated_frame_count"], 3)
        self.assertIn(
            "LOW_REPLICA_FRAME_COUNT_POOLED_PCA",
            {issue["code"] for issue in common["issues"]},
        )

    def test_budgeted_basis_can_project_every_source_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root, True, "replica_equal")
            project = json.loads(path.read_text(encoding="utf-8"))
            settings = project["definitions"]["common_pca"]
            settings["frame_selection"] = {
                "mode": "uniform_per_replica_budget_v1",
                "maximum_frames_per_replica": 2,
            }
            settings["projection_frame_stride"] = 1
            settings["projection_frame_selection"] = {"mode": "fixed_stride_v1"}
            path.write_text(json.dumps(project), encoding="utf-8")
            report = common_pca_project(path)
        self.assertEqual(report["basis"]["evaluated_frame_count"], 4)
        self.assertEqual(report["basis_frame_selection"]["selected_frame_count"], 4)
        self.assertEqual(report["projection_frame_selection"]["selected_frame_count"], 6)
        replicas = [
            replica
            for system in report["systems"]
            for replica in system["replicas"]
        ]
        self.assertEqual(
            [replica["basis_evaluated_frame_count"] for replica in replicas], [2, 2]
        )
        self.assertEqual(
            [replica["projection_evaluated_frame_count"] for replica in replicas], [2, 4]
        )
        self.assertEqual(
            sum(
                len(segment["projections"])
                for replica in replicas
                for segment in replica["segments"]
            ),
            6,
        )

    def test_feature_gate_fails_machine_readably(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = common_pca_project_safe(
                _write_project(Path(temporary), True, maximum_features=2)
            )
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("quadratically", report["issues"][0]["message"])

    def test_common_pca_randomized_solver_projects_all_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root, True, "replica_equal")
            project = json.loads(path.read_text(encoding="utf-8"))
            settings = project["definitions"]["common_pca"]
            settings["solver"] = {
                "method": "randomized_truncated_svd_v1",
                "oversampling": 2,
                "power_iterations": 2,
                "power_iteration_schedule": [2, 4],
                "random_seed": 11,
                "maximum_sample_matrix_elements": 1_000,
                "maximum_relative_residual": 1.0e-8,
            }
            path.write_text(json.dumps(project), encoding="utf-8")
            report = common_pca_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(
            report["basis"]["solver_diagnostics"]["method"],
            "randomized_truncated_svd_v1",
        )
        self.assertEqual(
            report["basis"]["solver_diagnostics"]["power_iteration_schedule"],
            [2, 4],
        )
        self.assertEqual(report["basis"]["evaluated_frame_count"], 6)
        self.assertEqual(
            sum(
                replica["projection_evaluated_frame_count"]
                for system in report["systems"]
                for replica in system["replicas"]
            ),
            6,
        )

    def test_cli_exposes_both_pca_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            individual_path = _write_project(root, False)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["individual-pca", str(individual_path)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["module_id"], "individual_pca")


if __name__ == "__main__":
    unittest.main()
