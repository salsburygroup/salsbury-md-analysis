import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.dccm import dccm_project, dccm_project_safe


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _frame(cb_x, cg_x, label):
    return (
        f"5\n{label}\n"
        "C 0 0 0\n"
        "N 1 0 0\n"
        "O 0 1 0\n"
        f"C {cb_x} 2 0\n"
        f"C {cg_x} 2 0\n"
    )


def _write_project(
    root: Path,
    maximum_atoms: int = 10,
    frame_stride: int = 1,
    minimum_frames: int = 3,
    values=(-1.0, 0.0, 1.0),
) -> Path:
    atoms = [
        _pdb_atom(1, "C", 0, 0, 0, "C"),
        _pdb_atom(2, "N", 1, 0, 0, "N"),
        _pdb_atom(3, "O", 0, 1, 0, "O"),
        _pdb_atom(4, "CB", 0, 2, 0, "C"),
        _pdb_atom(5, "CG", 1, 2, 0, "C"),
    ]
    (root / "reference.pdb").write_text("".join(atoms) + "END\n", encoding="utf-8")
    positive = "".join(
        _frame(value, 2 * value, f"p{index}")
        for index, value in enumerate(values)
    )
    negative = "".join(
        _frame(value, -2 * value, f"n{index}")
        for index, value in enumerate(values)
    )
    (root / "positive.xyz").write_text(positive, encoding="utf-8")
    (root / "negative.xyz").write_text(negative, encoding="utf-8")
    system = {
        "systems": [
            {
                "system_id": "positive",
                "replicas": [{
                    "replica_id": "r1",
                    "topology": "reference.pdb",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "positive.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 5, "unit": "ps"},
                    }],
                }],
            },
            {
                "system_id": "negative",
                "replicas": [{
                    "replica_id": "r1",
                    "topology": "reference.pdb",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "negative.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 5, "unit": "ps"},
                    }],
                }],
            },
        ]
    }
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "dccm-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "reference_system": "positive",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"atom_names": ["C", "N", "O"]},
            "analysis": {"atom_names": ["CB", "CG"]},
        },
        "definitions": {
            "dccm": {
                "alignment_selection": "alignment",
                "analysis_selection": "analysis",
                "minimum_reference_coverage": 1.0,
                "frame_stride": frame_stride,
                "maximum_atoms": maximum_atoms,
                "minimum_evaluated_frames_per_replica": minimum_frames,
                "minimum_variance_angstrom2": 1.0e-12,
            }
        },
        "requested_modules": ["dccm"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class DCCMTests(unittest.TestCase):
    def test_positive_negative_and_reference_difference_matrices(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = dccm_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        by_system = {system["system_id"]: system for system in report["systems"]}
        positive = by_system["positive"]["frame_pooled_dccm"]["matrix"]
        negative = by_system["negative"]["frame_pooled_dccm"]["matrix"]
        difference = by_system["negative"]["difference_from_reference_dccm"]["matrix"]
        self.assertAlmostEqual(positive[0][1], 1.0)
        self.assertAlmostEqual(negative[0][1], -1.0)
        self.assertAlmostEqual(difference[0][1], -2.0)

    def test_zero_variance_entries_are_null_not_zero(self):
        report = dccm_project(EXAMPLE / "project.json", hash_content=True)
        matrix = report["systems"][0]["frame_pooled_dccm"]["matrix"]
        self.assertIsNone(matrix[0][0])
        self.assertIn(
            "UNDEFINED_ZERO_VARIANCE_CORRELATIONS",
            {issue["code"] for issue in report["issues"]},
        )

    def test_quadratic_atom_gate_fails_before_trajectory_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = dccm_project_safe(_write_project(Path(temporary), maximum_atoms=1))
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("quadratically", report["issues"][0]["message"])

    def test_frame_stride_is_reported_and_applied_per_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = dccm_project(
                _write_project(
                    Path(temporary), frame_stride=2, minimum_frames=2,
                    values=(-1.0, 0.0, 1.0, 2.0),
                )
            )
        first_replica = report["systems"][0]["replicas"][0]
        self.assertEqual(first_replica["dccm"]["evaluated_frame_count"], 2)
        self.assertEqual(first_replica["segments"][0]["evaluated_frame_count"], 2)
        self.assertEqual(
            first_replica["segments"][0]["evaluated_time_range"],
            {"start": 0.0, "end": 10.0, "unit": "ps"},
        )
        self.assertIn("FRAME_SUBSAMPLING", {issue["code"] for issue in report["issues"]})

    def test_uniform_budget_balances_system_replicas_and_reports_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root, minimum_frames=2)
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["dccm"]["frame_selection"] = {
                "mode": "uniform_per_replica_budget_v1",
                "maximum_frames_per_replica": 2,
            }
            path.write_text(json.dumps(project), encoding="utf-8")
            report = dccm_project(path)
        self.assertEqual(report["frame_selection"]["source_frame_count"], 6)
        self.assertEqual(report["frame_selection"]["selected_frame_count"], 4)
        self.assertAlmostEqual(report["frame_selection"]["coverage_fraction"], 2 / 3)
        self.assertEqual(
            [row["selected_frame_count"] for row in report["frame_selection"]["replicas"]],
            [2, 2],
        )
        for system in report["systems"]:
            replica = system["replicas"][0]
            self.assertEqual(replica["dccm"]["evaluated_frame_count"], 2)
            self.assertEqual(
                replica["segments"][0]["evaluated_time_range"],
                {"start": 0.0, "end": 10.0, "unit": "ps"},
            )
        self.assertIn("FRAME_SUBSAMPLING", {issue["code"] for issue in report["issues"]})

    def test_one_frame_replica_is_retained_only_in_system_pooled_dccm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root, minimum_frames=2)
            (root / "single.xyz").write_text(
                _frame(0.5, 1.0, "single"), encoding="utf-8"
            )
            system = json.loads((root / "system.json").read_text(encoding="utf-8"))
            system["systems"][0]["replicas"].append({
                "replica_id": "rare-event-r2",
                "topology": "reference.pdb",
                "segments": [{
                    "segment_id": "s1",
                    "trajectory": "single.xyz",
                    "timing": {
                        "first_frame_time": 0,
                        "frame_interval": 5,
                        "unit": "ps",
                    },
                }],
            })
            (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
            report = dccm_project(path)
        self.assertEqual(report["technical_status"], "complete")
        positive = next(row for row in report["systems"] if row["system_id"] == "positive")
        rare = next(
            row for row in positive["replicas"] if row["replica_id"] == "rare-event-r2"
        )
        self.assertIsNone(rare["dccm"])
        self.assertEqual(positive["frame_pooled_dccm"]["evaluated_frame_count"], 4)
        self.assertIn(
            "INSUFFICIENT_REPLICA_FRAMES_FOR_REPLICA_DCCM",
            {issue["code"] for issue in report["issues"]},
        )

    def test_cli_emits_machine_readable_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["dccm", str(EXAMPLE / "project.json")])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "dccm")


if __name__ == "__main__":
    unittest.main()
