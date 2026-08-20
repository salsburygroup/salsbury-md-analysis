import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.rmsf import pooled_rmsf_project, pooled_rmsf_project_safe


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _xyz_frame(cb_x, label):
    return (
        f"4\n{label}\n"
        "C 0 0 0\n"
        "N 1 0 0\n"
        "O 0 1 0\n"
        f"C {cb_x} 1 0\n"
    )


def _write_two_replica_project(
    root: Path,
    block_size: int = 2,
    include_partial: bool = True,
    extra_frame: bool = False,
    frame_stride: int = 1,
) -> Path:
    atoms = [
        _pdb_atom(1, "C", 0, 0, 0, "C"),
        _pdb_atom(2, "N", 1, 0, 0, "N"),
        _pdb_atom(3, "O", 0, 1, 0, "O"),
        _pdb_atom(4, "CB", 1, 1, 0, "C"),
    ]
    (root / "reference.pdb").write_text("".join(atoms) + "END\n", encoding="utf-8")
    r1 = _xyz_frame(1, "r1-f0") + _xyz_frame(3, "r1-f1")
    r2 = _xyz_frame(2, "r2-f0") + _xyz_frame(6, "r2-f1")
    if extra_frame:
        r1 += _xyz_frame(5, "r1-f2")
        r2 += _xyz_frame(10, "r2-f2")
    (root / "r1.xyz").write_text(r1, encoding="utf-8")
    (root / "r2.xyz").write_text(r2, encoding="utf-8")
    system = {
        "systems": [{
            "system_id": "system",
            "replicas": [
                {
                    "replica_id": "r1",
                    "topology": "reference.pdb",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "r1.xyz",
                        "timing": {"first_frame_time": 5, "frame_interval": 2, "unit": "ps"},
                    }],
                },
                {
                    "replica_id": "r2",
                    "topology": "reference.pdb",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "r2.xyz",
                        "timing": {"first_frame_time": 5, "frame_interval": 2, "unit": "ps"},
                    }],
                },
            ],
        }]
    }
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "rmsf-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"atom_names": ["C", "N", "O"]},
            "analysis": {"atom_names": ["CB"]},
        },
        "definitions": {
            "pooled_rmsf": {
                "alignment_selection": "alignment",
                "analysis_selection": "analysis",
                "minimum_reference_coverage": 1.0,
                "frame_stride": frame_stride,
                "time_block_size_frames": block_size,
                "include_partial_final_block": include_partial,
                "minimum_replicas_for_uncertainty": 2,
            }
        },
        "requested_modules": ["pooled_rmsf"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class RMSFTests(unittest.TestCase):
    def test_teaching_fixture_reports_zero_fluctuation_without_fake_uncertainty(self):
        report = pooled_rmsf_project(EXAMPLE / "project.json", hash_content=True)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        system = report["systems"][0]
        self.assertEqual(system["atom_statistics"][0]["frame_pooled_rmsf_angstrom"], 0.0)
        summary = system["atom_statistics"][0]["replica_rmsf_summary_angstrom"]
        self.assertEqual(summary["count"], 1)
        self.assertIsNone(summary["sem"])
        self.assertIn(
            "REPLICA_UNCERTAINTY_UNAVAILABLE",
            {issue["code"] for issue in report["issues"]},
        )

    def test_frame_pooled_and_replica_balanced_estimators_remain_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = pooled_rmsf_project(
                _write_two_replica_project(Path(temporary))
            )
        row = report["systems"][0]["atom_statistics"][0]
        self.assertAlmostEqual(row["frame_pooled_rmsf_angstrom"], math.sqrt(3.5))
        replica = row["replica_rmsf_summary_angstrom"]
        self.assertAlmostEqual(replica["mean"], 1.5)
        self.assertAlmostEqual(replica["sample_sd"], math.sqrt(0.5))
        self.assertAlmostEqual(replica["sem"], 0.5)
        self.assertEqual(report["systems"][0]["included_time_block_count"], 2)

    def test_partial_blocks_are_explicitly_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = pooled_rmsf_project(
                _write_two_replica_project(
                    Path(temporary), block_size=2, include_partial=False, extra_frame=True
                )
            )
        system = report["systems"][0]
        self.assertEqual(system["included_time_block_count"], 2)
        discarded = [
            replica["segments"][0]["discarded_partial_block_frame_count"]
            for replica in system["replicas"]
        ]
        self.assertEqual(discarded, [1, 1])

    def test_frame_stride_retains_source_frame_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = pooled_rmsf_project(
                _write_two_replica_project(
                    Path(temporary), extra_frame=True, frame_stride=2
                )
            )
        first_replica = report["systems"][0]["replicas"][0]
        self.assertEqual(first_replica["segments"][0]["decoded_frame_count"], 2)
        block = first_replica["segments"][0]["time_blocks"][0]
        self.assertEqual(first_replica["evaluated_frame_count"], 2)
        self.assertEqual((block["start_frame_index"], block["end_frame_index"]), (0, 2))
        self.assertEqual((block["start_time"], block["end_time"]), (5.0, 9.0))
        self.assertIn("FRAME_SUBSAMPLING", {issue["code"] for issue in report["issues"]})

    def test_missing_configuration_fails_machine_readably(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_two_replica_project(Path(temporary))
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"].pop("pooled_rmsf")
            path.write_text(json.dumps(project), encoding="utf-8")
            report = pooled_rmsf_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertEqual(report["issues"][0]["code"], "RMSF_INVALID")

    def test_cli_emits_machine_readable_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["rmsf", str(EXAMPLE / "project.json")])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "pooled_rmsf")


if __name__ == "__main__":
    unittest.main()
