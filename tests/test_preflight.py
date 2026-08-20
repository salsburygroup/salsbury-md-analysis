import json
import struct
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.manifests import load_json
from salsbury_md_analysis.preflight import (
    FileProbeError,
    preflight_system,
    probe_dcd,
    probe_bond_json,
    probe_connectivity,
    probe_gro,
    probe_prmtop,
    probe_psf,
    probe_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


def _record(payload: bytes, endian: str = "<") -> bytes:
    marker = struct.pack(f"{endian}i", len(payload))
    return marker + payload + marker


def _write_dcd(
    path: Path,
    atom_count: int = 1,
    frames: int = 2,
    start: int = 0,
    interval: int = 100,
    endian: str = "<",
) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into(f"{endian}3i", header, 4, frames, start, interval)
    title = struct.pack(f"{endian}i", 1) + b"synthetic DCD header fixture".ljust(80)
    path.write_bytes(
        _record(bytes(header), endian)
        + _record(title, endian)
        + _record(struct.pack(f"{endian}i", atom_count), endian)
    )


class PreflightTests(unittest.TestCase):
    def test_teaching_fixture_preflight_is_complete_but_not_scientific(self):
        manifest = EXAMPLE / "system.json"
        report = preflight_system(load_json(manifest), manifest, hash_content=True)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        self.assertEqual(report["error_count"], 0)
        replica = report["systems"][0]["replicas"][0]
        self.assertEqual(replica["topology"]["atom_count"], 1)
        self.assertEqual(replica["segments"][0]["trajectory"]["observed_frame_count"], 2)

    def test_atom_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.pdb").write_text(
                "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            (root / "two.xyz").write_text(
                "2\nframe\nC 0 0 0\nC 1 0 0\n", encoding="utf-8"
            )
            data = {
                "systems": [{
                    "system_id": "s",
                    "replicas": [{
                        "replica_id": "r",
                        "topology": "one.pdb",
                        "segments": [{
                            "segment_id": "1", "trajectory": "two.xyz",
                            "timing": {
                                "first_frame_time": 0,
                                "frame_interval": 1,
                                "unit": "ps",
                            },
                        }],
                    }],
                }]
            }
            manifest = root / "system.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            report = preflight_system(data, manifest)
            self.assertEqual(report["technical_status"], "failed")
            self.assertEqual(report["issues"][0]["code"], "ATOM_COUNT_MISMATCH")

    def test_dcd_header_and_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "one.pdb"
            topology.write_text(
                "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            first = root / "first.dcd"
            second = root / "second.dcd"
            _write_dcd(first, frames=2, start=0, interval=100)
            _write_dcd(second, frames=2, start=100, interval=100)
            probe = probe_dcd(first)
            self.assertEqual(probe["atom_count"], 1)
            self.assertEqual(probe["declared_frame_count"], 2)
            big_endian = root / "big-endian.dcd"
            _write_dcd(big_endian, endian=">")
            self.assertEqual(probe_dcd(big_endian)["byte_order"], "big")
            data = {
                "systems": [{
                    "system_id": "s",
                    "replicas": [{
                        "replica_id": "r",
                        "topology": "one.pdb",
                        "segments": [
                            {
                                "segment_id": "1", "trajectory": "first.dcd",
                                "timing": {
                                    "first_frame_time": 0,
                                    "frame_interval": 2,
                                    "unit": "ps",
                                },
                            },
                            {
                                "segment_id": "2", "trajectory": "second.dcd",
                                "continuous_with_previous": True,
                                "timing": {
                                    "first_frame_time": 4,
                                    "frame_interval": 2,
                                    "unit": "ps",
                                },
                            },
                        ],
                    }],
                }]
            }
            manifest = root / "system.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            report = preflight_system(data, manifest)
            self.assertEqual(report["technical_status"], "failed")
            self.assertIn("DCD_CONTINUITY_MISMATCH", {issue["code"] for issue in report["issues"]})
            _write_dcd(second, frames=2, start=200, interval=100)
            corrected = preflight_system(data, manifest)
            self.assertEqual(corrected["technical_status"], "complete")
            self.assertNotIn(
                "DCD_CONTINUITY_MISMATCH", {issue["code"] for issue in corrected["issues"]}
            )
            _write_dcd(second, frames=2, start=0, interval=100)
            reset = preflight_system(data, manifest)
            self.assertEqual(reset["technical_status"], "complete")
            self.assertIn(
                "DCD_HEADER_STEP_RESET", {issue["code"] for issue in reset["issues"]}
            )
            data["systems"][0]["replicas"][0]["segments"][1]["timing"][
                "first_frame_time"
            ] = 3
            time_mismatch = preflight_system(data, manifest)
            self.assertIn(
                "PHYSICAL_TIME_CONTINUITY_MISMATCH",
                {issue["code"] for issue in time_mismatch["issues"]},
            )

    def test_other_topology_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            psf = root / "one.psf"
            psf.write_text("PSF\n\n       1 !NATOM\n", encoding="utf-8")
            prmtop = root / "one.prmtop"
            prmtop.write_text(
                "%VERSION VERSION_STAMP = V0001.000\n%FLAG POINTERS\n%FORMAT(10I8)\n       1\n",
                encoding="utf-8",
            )
            gro = root / "one.gro"
            gro.write_text(
                "synthetic\n1\n    1UNK      C    1   0.000   0.000   0.000\n1.0 1.0 1.0\n",
                encoding="utf-8",
            )
            self.assertEqual(probe_psf(psf)["atom_count"], 1)
            self.assertEqual(probe_prmtop(prmtop)["atom_count"], 1)
            self.assertEqual(probe_gro(gro, "topology")["atom_count"], 1)

    def test_connectivity_is_probed_hashed_and_cardinality_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "one.pdb"
            topology.write_text(
                "ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            trajectory = root / "one.xyz"
            trajectory.write_text("1\nframe\nC 0 0 0\n", encoding="utf-8")
            connectivity = root / "one.psf"
            connectivity.write_text("PSF\n\n       1 !NATOM\n", encoding="utf-8")
            data = {
                "systems": [{
                    "system_id": "s",
                    "replicas": [{
                        "replica_id": "r",
                        "topology": "one.pdb",
                        "connectivity": "one.psf",
                        "segments": [{
                            "segment_id": "1",
                            "trajectory": "one.xyz",
                            "timing": {
                                "first_frame_time": 0,
                                "frame_interval": 1,
                                "unit": "ps",
                            },
                        }],
                    }],
                }]
            }
            manifest = root / "system.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            report = preflight_system(data, manifest, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            observed = report["systems"][0]["replicas"][0]["connectivity"]
            self.assertEqual(observed["role"], "connectivity")
            self.assertEqual(observed["atom_count"], 1)
            self.assertEqual(len(observed["sha256"]), 64)

            connectivity.write_text("PSF\n\n       2 !NATOM\n", encoding="utf-8")
            mismatch = preflight_system(data, manifest)
            self.assertEqual(mismatch["technical_status"], "failed")
            self.assertIn(
                "CONNECTIVITY_ATOM_COUNT_MISMATCH",
                {issue["code"] for issue in mismatch["issues"]},
            )

    def test_portable_bond_json_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bonds.json"
            path.write_text(
                json.dumps({
                    "format": "salsbury-bonds-v1",
                    "atom_count": 2,
                    "index_base": 0,
                    "bonds": [[0, 1]],
                }),
                encoding="utf-8",
            )
            self.assertEqual(probe_bond_json(path)["bond_count"], 1)
            self.assertEqual(probe_connectivity(path)["role"], "connectivity")

    def test_unsupported_trajectory_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.xtc"
            path.write_bytes(b"not-an-xtc")
            with self.assertRaises(FileProbeError) as context:
                probe_trajectory(path)
            self.assertIn("unsupported trajectory format", str(context.exception))


if __name__ == "__main__":
    unittest.main()
