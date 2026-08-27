import json
import struct
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.coordinate_cache import (
    CoordinateCacheError,
    build_coordinate_cache,
    build_coordinate_cache_safe,
    validate_reusable_coordinate_cache,
)
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.manifests import load_json, validate_system
from salsbury_md_analysis.preflight import probe_trajectory


def record(payload: bytes) -> bytes:
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def write_dcd(path: Path) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, 3, 0, 1)
    struct.pack_into("<i", header, 44, 1)
    struct.pack_into("<i", header, 80, 24)
    payload = record(bytes(header))
    payload += record(struct.pack("<i", 1) + b"cache fixture".ljust(80))
    payload += record(struct.pack("<i", 5))
    frames = (
        ((9.5, 0.5, 5.0, 5.5, 8.0), (0.0,) * 5, (0.0,) * 5),
        # The ligand crosses x=10 between saved frames.  Its cached image must
        # remain continuous at 10.2/11.2 rather than jumping to 0.2/1.2.
        ((0.2, 1.2, 5.0, 5.5, 8.0), (0.0,) * 5, (0.0,) * 5),
        ((0.7, 1.7, 5.0, 5.5, 8.0), (0.0,) * 5, (0.0,) * 5),
    )
    for axes in frames:
        payload += record(struct.pack("<6d", 10.0, 90.0, 10.0, 90.0, 90.0, 10.0))
        for values in axes:
            payload += record(struct.pack("<5f", *values))
    path.write_bytes(payload)


class CoordinateCacheTests(unittest.TestCase):
    def test_cache_is_made_whole_solute_only_and_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb = root / "system.pdb"
            pdb.write_text(
                "ATOM      1  C1  LIG A   1       9.500   0.000   0.000  1.00  0.00           C\n"
                "ATOM      2  C2  LIG A   1       0.500   0.000   0.000  1.00  0.00           C\n"
                "HETATM    3  O   TIP W   2       5.000   0.000   0.000  1.00  0.00           O\n"
                "HETATM    4  H1  TIP W   2       5.500   0.000   0.000  1.00  0.00           H\n"
                "HETATM    5 MG   MG  M   3       8.000   0.000   0.000  1.00  0.00          MG\n"
                "END\n",
                encoding="utf-8",
            )
            bonds = root / "system.bonds.json"
            bonds.write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 5,
                "index_base": 0, "bonds": [[0, 1], [2, 3]],
            }), encoding="utf-8")
            dcd = root / "segment.dcd"
            write_dcd(dcd)
            system = {
                "systems": [{
                    "system_id": "test",
                    "replicas": [{
                        "replica_id": "replica-1",
                        "topology": str(pdb),
                        "connectivity": str(bonds),
                        "segments": [{
                            "segment_id": "production",
                            "trajectory": str(dcd),
                            "timing": {
                                "first_frame_time": 0.0,
                                "frame_interval": 1.0,
                                "unit": "ps",
                            },
                        }],
                    }],
                }],
            }
            manifest = root / "system.json"
            manifest.write_text(json.dumps(system), encoding="utf-8")
            output = root / "cache"
            report = build_coordinate_cache(manifest, output)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(
                report["rows"][0]["periodic_reconstruction"]["policy"],
                "unwrap_continuous",
            )
            cached_manifest_path = output / "system-cache.json"
            cached = load_json(cached_manifest_path)
            validate_system(cached, source_path=cached_manifest_path, check_paths=True)
            replica = cached["systems"][0]["replicas"][0]
            trajectory = output / replica["segments"][0]["trajectory"]
            self.assertEqual(probe_trajectory(trajectory)["atom_count"], 3)
            frames = list(iter_coordinate_frames(trajectory, "angstrom"))
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(
                frame.coordinate_representation
                == "made_whole_molecular_payload_cache"
                for frame in frames
            ))
            self.assertAlmostEqual(frames[0].coordinates_angstrom[1][0], 10.5)
            self.assertAlmostEqual(frames[0].coordinates_angstrom[2][0], 8.0)
            self.assertAlmostEqual(
                frames[1].coordinates_angstrom[0][0], 10.2, places=5
            )
            self.assertAlmostEqual(
                frames[1].coordinates_angstrom[1][0], 11.2, places=5
            )
            self.assertFalse((output / "partial").exists())
            per_system = report["cached_per_system_manifests"]["test"]
            self.assertTrue((output / per_system["path"]).is_file())
            reuse = validate_reusable_coordinate_cache(output, manifest)
            self.assertEqual(reuse["technical_status"], "complete")
            self.assertEqual(reuse["replica_count"], 1)
            failed = build_coordinate_cache_safe(manifest, output)
            self.assertEqual(failed["technical_status"], "failed")

            strided_output = root / "strided-cache"
            strided = build_coordinate_cache(
                manifest, strided_output, cache_stride=2
            )
            self.assertEqual(strided["cache_stride"], 2)
            self.assertEqual(strided["source_frame_scan"], "all source frames decoded in order")
            strided_row = strided["rows"][0]
            self.assertEqual(strided_row["decoded_frame_count"], 3)
            self.assertEqual(strided_row["retained_frame_count"], 2)
            strided_manifest = load_json(strided_output / "system-cache.json")
            strided_replica = strided_manifest["systems"][0]["replicas"][0]
            strided_segment = strided_replica["segments"][0]
            self.assertEqual(strided_segment["timing"]["frame_interval"], 2.0)
            strided_frames = list(iter_coordinate_frames(
                strided_output / strided_segment["trajectory"], "angstrom"
            ))
            self.assertEqual(len(strided_frames), 2)
            self.assertAlmostEqual(
                strided_frames[1].coordinates_angstrom[0][0], 10.7, places=5
            )

            second_replica = json.loads(json.dumps(
                system["systems"][0]["replicas"][0]
            ))
            second_replica["replica_id"] = "replica-2"
            system["systems"][0]["replicas"].append(second_replica)
            manifest.write_text(json.dumps(system), encoding="utf-8")
            parallel_output = root / "parallel-cache"
            parallel = build_coordinate_cache(
                manifest, parallel_output, maximum_workers=2
            )
            self.assertEqual(parallel["maximum_workers_used"], 2)
            parallel_manifest = load_json(parallel_output / "system-cache.json")
            self.assertEqual(
                len(parallel_manifest["systems"][0]["replicas"]), 2
            )
            self.assertEqual(
                len(list(parallel_output.glob("*-segment-00.dcd"))), 2
            )
            with self.assertRaisesRegex(
                CoordinateCacheError, "stride 1"
            ):
                validate_reusable_coordinate_cache(strided_output, manifest)
            pdb.write_text(
                pdb.read_text(encoding="utf-8") + "REMARK source changed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CoordinateCacheError, "topology changed"
            ):
                validate_reusable_coordinate_cache(parallel_output, manifest)


if __name__ == "__main__":
    unittest.main()
