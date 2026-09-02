import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _write_dcd(path: Path, atom_count: int, frames: int) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, frames, 0, 1)
    title = struct.pack("<i", 1) + b"calibration fixture".ljust(80)
    path.write_bytes(
        _record(bytes(header))
        + _record(title)
        + _record(struct.pack("<i", atom_count))
    )


class PlannerCalibrationMatrixTests(unittest.TestCase):
    def test_prepare_and_collect_exact_one_replica_points(self):
        prepare = _load_script("prepare_planner_calibration_matrix.py")
        collect = _load_script("collect_planner_calibration_matrix.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb = root / "system.pdb"
            pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
                "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
                "END\n",
                encoding="utf-8",
            )
            trajectory = root / "trajectory.dcd"
            _write_dcd(trajectory, atom_count=2, frames=20)
            system = root / "system-source.json"
            system.write_text(json.dumps({"systems": [{
                "system_id": "system-a",
                "replicas": [{
                    "replica_id": "replica-1",
                    "topology": str(pdb),
                    "segments": [{
                        "segment_id": "production",
                        "trajectory": str(trajectory),
                        "timing": {
                            "first_frame_time": 0.0,
                            "frame_interval": 1.0,
                            "unit": "ps",
                        },
                    }],
                }],
            }]}), encoding="utf-8")
            project = root / "project-source.json"
            project.write_text(json.dumps({
                "project_id": "source",
                "system_manifest": str(system),
                "analysis_output_root": "results",
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "reject",
                "reference_system": "system-a",
                "reference_structure": str(pdb),
                "common_atom_policy": "strict",
                "selections": {
                    "alignment": {"preset": "backbone"},
                    "analysis": {"preset": "heavy"},
                },
                "definitions": {"structural_qc": {
                    "near_coincident_distance_angstrom": 0.5,
                    "maximum_near_coincident_pairs_per_frame": 0,
                    "maximum_absolute_coordinate_angstrom": 1_000_000.0,
                    "maximum_frame_atom_displacement_angstrom": 10.0,
                    "frame_stride": 1,
                    "checkpointing": {
                        "enabled": True,
                        "within_segment_interval_seconds": 60.0,
                    },
                }},
                "requested_modules": ["structural_integrity_qc"],
                "protected_locations": [str(root)],
            }), encoding="utf-8")
            matrix_root = root / "matrix"
            matrix = prepare.prepare_matrix(
                project,
                system,
                system_id="system-a",
                replica_id="replica-1",
                module_id="structural_integrity_qc",
                frame_budgets=[5, 10],
                output=matrix_root,
                label="small",
            )
            self.assertEqual(matrix["point_count"], 2)
            self.assertEqual(
                [row["selected_source_physical_frames"] for row in matrix["points"]],
                [5, 10],
            )
            self.assertTrue(all(row["topology_atom_count"] == 2 for row in matrix["points"]))
            for point in matrix["points"]:
                report_path = matrix_root / point["report"]
                report_path.write_text(json.dumps({
                    "module_id": "structural_integrity_qc",
                    "technical_status": "complete",
                    "scientific_status": "not evaluated",
                    "error_count": 0,
                }), encoding="utf-8")
                report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
                sidecar_path = matrix_root / point["sidecar"]
                sidecar_path.write_text(json.dumps({
                    "technical_status": "complete",
                    "report_sha256": report_hash,
                    "resource_evidence": {
                        "selected_source_physical_frames": point[
                            "selected_source_physical_frames"
                        ],
                        "execution_resources": {
                            "total_cpu_seconds": 1.0,
                            "wall_seconds": 1.0,
                            "maximum_resident_memory_mib": 10.0,
                            "requested_cpu_count": 1,
                            "measurement_scope": (
                                "one fresh child process for one analysis command"
                            ),
                            "stderr_nonempty": False,
                        },
                    },
                }), encoding="utf-8")
            evidence = collect.collect_matrix(matrix_root / "matrix.json")
            self.assertEqual(evidence["technical_status"], "complete")
            self.assertEqual(evidence["point_count"], 2)
            self.assertEqual(evidence["unexpected_error_count"], 0)
            self.assertEqual(evidence["points"][1]["topology_atom_frame_count"], 20)
            combined = collect.collect_matrices([matrix_root / "matrix.json"])
            self.assertEqual(combined["matrix_count"], 1)
            self.assertEqual(combined["point_count"], 2)


if __name__ == "__main__":
    unittest.main()
