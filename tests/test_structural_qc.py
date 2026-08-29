import io
import gzip
import json
import math
import random
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import salsbury_md_analysis.structural_qc as structural_qc_module
from salsbury_md_analysis.cli import main
from salsbury_md_analysis.structural_qc import (
    _maximum_rigid_body_aligned_displacement,
    _near_coincident_pairs,
    structural_qc_project,
    structural_qc_project_safe,
)
from salsbury_md_analysis.coordinates import CoordinateFrame


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


def _atom(serial: int, x: float, name: str = "C") -> str:
    return (
        f"ATOM  {serial:5d} {name:^4s} UNK A   1    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    )


def _write_project(
    root: Path,
    xyz_text: str,
    atom_count: int,
    displacement_gate: float = 10.0,
) -> Path:
    topology = root / "system.pdb"
    topology.write_text(
        "".join(_atom(index + 1, float(index)) for index in range(atom_count)) + "END\n",
        encoding="utf-8",
    )
    trajectory = root / "trajectory.xyz"
    trajectory.write_text(xyz_text, encoding="utf-8")
    system = {
        "systems": [{
            "system_id": "system",
            "replicas": [{
                "replica_id": "replica-1",
                "topology": "system.pdb",
                "segments": [{
                    "segment_id": "segment-1", "trajectory": "trajectory.xyz",
                    "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
                }],
            }],
        }]
    }
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "qc-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "selections": {
            "alignment": {"preset": "backbone"},
            "analysis": {"preset": "heavy"},
        },
        "definitions": {
            "structural_qc": {
                "near_coincident_distance_angstrom": 0.5,
                "maximum_near_coincident_pairs_per_frame": 0,
                "maximum_absolute_coordinate_angstrom": 1000.0,
                "maximum_frame_atom_displacement_angstrom": displacement_gate,
                "frame_stride": 1,
            }
        },
        "requested_modules": ["structural_integrity_qc"],
        "protected_locations": ["/protected/example"],
    }
    project_path = root / "project.json"
    project_path.write_text(json.dumps(project), encoding="utf-8")
    return project_path


class StructuralQCTests(unittest.TestCase):
    def test_displacement_removes_global_translation_and_rotation(self):
        previous = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        current = tuple(
            (-point[1] + 12.0, point[0] - 7.0, point[2] + 4.0)
            for point in previous
        )
        self.assertAlmostEqual(
            _maximum_rigid_body_aligned_displacement(previous, current),
            0.0,
            places=12,
        )

    def test_displacement_retains_internal_coordinate_jump(self):
        previous = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        current = list(previous)
        current[-1] = (0.0, 0.0, 4.0)
        self.assertGreater(
            _maximum_rigid_body_aligned_displacement(previous, tuple(current)),
            1.0,
        )

    def test_kdtree_neighbor_search_matches_brute_force(self):
        generator = random.Random(20260813)
        coordinates = [
            (generator.uniform(-3, 3), generator.uniform(-3, 3), generator.uniform(-3, 3))
            for _ in range(80)
        ]
        coordinates.extend([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (math.nan, 1.0, 2.0)])
        threshold = 0.75
        expected = []
        for left in range(len(coordinates)):
            if not all(math.isfinite(value) for value in coordinates[left]):
                continue
            for right in range(left + 1, len(coordinates)):
                if not all(math.isfinite(value) for value in coordinates[right]):
                    continue
                distance = math.dist(coordinates[left], coordinates[right])
                if distance <= threshold:
                    expected.append((left, right, distance))
        frame = CoordinateFrame(0, coordinates, "angstrom", False, None)
        count, minimum, examples = _near_coincident_pairs(frame, threshold, 7)
        self.assertEqual(count, len(expected))
        self.assertAlmostEqual(minimum, min(row[2] for row in expected))
        self.assertEqual(
            [(row["atom_index_1"], row["atom_index_2"]) for row in examples],
            [(left, right) for left, right, _ in expected[:7]],
        )

    def test_teaching_fixture_passes_technical_gates_only(self):
        with patch.object(
            structural_qc_module, "_write_structural_qc_checkpoint"
        ):
            report = structural_qc_project(
                EXAMPLE / "project.json", hash_content=True
            )
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        self.assertTrue(report["input_content_signature_sha256"])
        segment = report["systems"][0]["replicas"][0]["segments"][0]
        self.assertEqual(segment["observed_frame_count"], 2)
        self.assertEqual(segment["evaluated_frame_count"], 2)

    def test_near_coincident_pair_is_nonblocking_review_finding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root, "2\nframe\nC 0 0 0\nC 0 0 0\n", atom_count=2
            )
            report = structural_qc_project(project)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["qc_status"], "review_required")
            self.assertEqual(report["human_review_status"], "pending")
            self.assertEqual(report["scientific_status"], "not evaluated")
            self.assertIn(
                "NEAR_COINCIDENT_PAIR_THRESHOLD_EXCEEDED",
                {issue["code"] for issue in report["issues"]},
            )

    def test_raw_frame_displacement_gate_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root,
                "1\nf0\nC 0 0 0\n1\nf1\nC 2 0 0\n",
                atom_count=1,
                displacement_gate=0.5,
            )
            report = structural_qc_project(project)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["qc_status"], "review_required")
            self.assertIn(
                "FRAME_DISPLACEMENT_EXCEEDED",
                {issue["code"] for issue in report["issues"]},
            )

    def test_displacement_selection_excludes_diffusing_bulk_water(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root,
                "5\nf0\nN 0 0 0\nC 1 0 0\nC 0 1 0\nO 0 0 1\nO 10 10 10\n"
                "5\nf1\nN 0 0 0\nC 1 0 0\nC 0 1 0\nO 0 0 1\nO 500 10 10\n",
                atom_count=5,
                displacement_gate=10.0,
            )
            topology = root / "system.pdb"
            topology.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      3  C   ALA A   1       0.000   1.000   0.000  1.00  0.00           C\n"
                "ATOM      4  O   ALA A   1       0.000   0.000   1.000  1.00  0.00           O\n"
                "HETATM    5  O   HOH W   2      10.000  10.000  10.000  1.00  0.00           O\n"
                "END\n",
                encoding="utf-8",
            )
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload["selections"]["solute_heavy"] = {"preset": "solute_heavy"}
            payload["definitions"]["structural_qc"][
                "frame_displacement_selection"
            ] = "solute_heavy"
            project.write_text(json.dumps(payload), encoding="utf-8")
            report = structural_qc_project(project)
        self.assertEqual(report["technical_status"], "complete")
        segment = report["systems"][0]["replicas"][0]["segments"][0]
        self.assertEqual(segment["frame_displacement_selection"], "solute_heavy")
        self.assertEqual(segment["frame_displacement_atom_count"], 4)
        self.assertAlmostEqual(segment["maximum_frame_atom_displacement_angstrom"], 0.0)

    def test_frame_stride_reports_source_and_decoded_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root,
                "1\nf0\nC 0 0 0\n1\nf1\nC 1 0 0\n1\nf2\nC 2 0 0\n",
                atom_count=1,
            )
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload["definitions"]["structural_qc"]["frame_stride"] = 2
            project.write_text(json.dumps(payload), encoding="utf-8")
            report = structural_qc_project(project)
        segment = report["systems"][0]["replicas"][0]["segments"][0]
        self.assertEqual(segment["observed_frame_count"], 3)
        self.assertEqual(segment["decoded_frame_count"], 2)
        self.assertEqual(segment["evaluated_frame_count"], 2)
        self.assertIn("FRAME_SUBSAMPLING", {issue["code"] for issue in report["issues"]})

    def test_segment_completion_checkpoint_is_reused_without_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root,
                "1\nf0\nC 0 0 0\n1\nf1\nC 1 0 0\n",
                atom_count=1,
            )
            first = structural_qc_project(project)
            checkpoint_files = list(
                root.glob("outputs/structural-qc/checkpoints/*/*.json.gz")
            )
            self.assertEqual(len(checkpoint_files), 1)
            with gzip.open(checkpoint_files[0], "rt", encoding="utf-8") as handle:
                wrapper = json.load(handle)
            self.assertEqual(wrapper["payload"]["status"], "complete")
            self.assertEqual(
                first["checkpointing"]["within_segment_interval_seconds"],
                7200.0,
            )

            with patch.object(
                structural_qc_module,
                "iter_coordinate_frames",
                side_effect=AssertionError("completed segment was decoded again"),
            ):
                second = structural_qc_project(project)
            self.assertEqual(first, second)

    def test_within_segment_checkpoint_resumes_equivalently_after_two_hours(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(
                root,
                "1\nf0\nC 0 0 0\n1\nf1\nC 1 0 0\n1\nf2\nC 2 0 0\n",
                atom_count=1,
            )
            original_writer = structural_qc_module._write_structural_qc_checkpoint
            observed_statuses = []

            def write_then_interrupt(path, identity, payload):
                original_writer(path, identity, payload)
                observed_statuses.append(payload["status"])
                if payload["status"] == "in_progress":
                    raise KeyboardInterrupt("simulated scheduler interruption")

            with patch.object(
                structural_qc_module.time,
                "monotonic",
                side_effect=[0.0, 7201.0],
            ), patch.object(
                structural_qc_module,
                "_write_structural_qc_checkpoint",
                side_effect=write_then_interrupt,
            ):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "simulated scheduler interruption"
                ):
                    structural_qc_project(project)
            self.assertEqual(observed_statuses, ["in_progress"])

            checkpoint_files = list(
                root.glob("outputs/structural-qc/checkpoints/*/*.json.gz")
            )
            self.assertEqual(len(checkpoint_files), 1)
            with gzip.open(checkpoint_files[0], "rt", encoding="utf-8") as handle:
                interrupted_wrapper = json.load(handle)
            self.assertEqual(interrupted_wrapper["payload"]["status"], "in_progress")
            self.assertEqual(
                interrupted_wrapper["payload"]["segment_state"][
                    "last_decoded_frame_index"
                ],
                0,
            )

            resumed = structural_qc_project(project)
            with gzip.open(checkpoint_files[0], "rt", encoding="utf-8") as handle:
                completed_wrapper = json.load(handle)
            self.assertEqual(completed_wrapper["payload"]["status"], "complete")

            shutil.rmtree(root / "outputs" / "structural-qc" / "checkpoints")
            uninterrupted = structural_qc_project(project)
            self.assertEqual(resumed, uninterrupted)

    def test_missing_explicit_thresholds_returns_machine_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = _write_project(root, "1\nf0\nC 0 0 0\n", atom_count=1)
            data = json.loads(project.read_text(encoding="utf-8"))
            data.pop("definitions")
            project.write_text(json.dumps(data), encoding="utf-8")
            report = structural_qc_project_safe(project)
            self.assertEqual(report["technical_status"], "failed")
            self.assertEqual(report["issues"][0]["code"], "STRUCTURAL_QC_INVALID")

    def test_cli_emits_machine_readable_qc(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(
            structural_qc_module, "_write_structural_qc_checkpoint"
        ):
            status = main(["structural-qc", str(EXAMPLE / "project.json")])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "structural_integrity_qc")


if __name__ == "__main__":
    unittest.main()
