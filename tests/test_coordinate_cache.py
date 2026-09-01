import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.coordinate_cache import (
    CoordinateCacheError,
    _execute_coordinate_cache_workers,
    build_coordinate_cache,
    build_coordinate_cache_safe,
    validate_reusable_coordinate_cache,
)
from salsbury_md_analysis.cache_routing import (
    cache_compatibility,
    materialize_cache_backed_base_project,
)
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.manifests import load_json, validate_system
from salsbury_md_analysis.preflight import probe_trajectory
from salsbury_md_analysis.structural_qc import structural_qc_project


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
    def test_distributed_cache_workers_use_the_complete_slurm_allocation(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "SMA_DISTRIBUTED_REPLICA_WORKERS": "1",
                "SLURM_NNODES": "2",
                "SMA_REPLICA_WORKERS_PER_NODE": "40",
            },
        ), patch(
            "salsbury_md_analysis.coordinate_cache.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stderr = ""
            run.return_value.stdout = ""
            task = ("manifest.json", "output", False, 4.0, 0.1, 1)
            _execute_coordinate_cache_workers(
                [task] * 63,
                maximum_workers=63,
                worker_root=Path(temporary),
            )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--nodes") + 1], "2")
            self.assertEqual(command[command.index("--ntasks") + 1], "63")
            self.assertEqual(
                command[command.index("--ntasks-per-node") + 1], "40"
            )

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
            self.assertEqual(strided_row["retained_frame_count"], 1)
            strided_manifest = load_json(strided_output / "system-cache.json")
            strided_replica = strided_manifest["systems"][0]["replicas"][0]
            strided_segment = strided_replica["segments"][0]
            self.assertEqual(strided_segment["timing"]["frame_interval"], 2.0)
            strided_frames = list(iter_coordinate_frames(
                strided_output / strided_segment["trajectory"], "angstrom"
            ))
            self.assertEqual(len(strided_frames), 1)
            self.assertAlmostEqual(
                strided_frames[0].coordinates_angstrom[0][0], 9.5, places=5
            )
            strided_reuse = validate_reusable_coordinate_cache(
                strided_output, manifest
            )
            self.assertEqual(strided_reuse["cache_stride"], 2)
            self.assertIn(
                "Every 2 frame", strided_reuse["reuse_boundary"]
            )
            cache_source_project = root / "cache-source-project.json"
            cache_source_project.write_text(json.dumps({
                "project_id": "cache-routing",
                "analysis_profile": "standard_md_v1",
                "system_manifest": str(manifest),
                "analysis_output_root": str(root / "cache-routing-output"),
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "unwrap_continuous",
                "periodic_reconstruction": {
                    "maximum_bond_length_angstrom": 3.0,
                    "cycle_closure_tolerance_angstrom": 0.25,
                    "maximum_anchor_displacement_angstrom": 20.0,
                },
                "reference_structure": str(pdb),
                "reference_connectivity": str(bonds),
                "selections": {
                    "alignment": {"preset": "heavy"},
                    "analysis": {"preset": "heavy"},
                },
                "definitions": {"replica_rmsd_rg": {}},
                "requested_modules": ["replica_rmsd_rg"],
                "protected_locations": [],
            }), encoding="utf-8")
            cache_project = root / "project-cache-base.json"
            routing = materialize_cache_backed_base_project(
                cache_source_project, strided_output, cache_project
            )
            self.assertEqual(routing["technical_status"], "complete")
            cached_project = load_json(cache_project)
            self.assertEqual(
                cached_project["periodic_coordinate_policy"],
                "preprocessed_make_whole",
            )
            self.assertEqual(cached_project["requested_modules"], [
                "replica_rmsd_rg"
            ])

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
            project = root / "project.json"
            project.write_text(json.dumps({
                "project_id": "parallel-cache-qc",
                "analysis_profile": "standard_md_v1",
                "system_manifest": str(manifest),
                "analysis_output_root": str(root / "qc-output"),
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "unwrap_continuous",
                "periodic_reconstruction": {
                    "maximum_bond_length_angstrom": 3.0,
                    "cycle_closure_tolerance_angstrom": 0.25,
                    "maximum_anchor_displacement_angstrom": 20.0,
                },
                "selections": {
                    "alignment": {"preset": "heavy"},
                    "analysis": {"preset": "heavy"},
                },
                "definitions": {"structural_qc": {
                    "near_coincident_distance_angstrom": 0.2,
                    "maximum_near_coincident_pairs_per_frame": 100,
                    "maximum_absolute_coordinate_angstrom": 1000.0,
                    "frame_stride": 1,
                    "parallel_execution": {
                        "enabled": True,
                        "maximum_workers": 2,
                        "coordinate_cache_system_manifest": str(
                            parallel_output / "system-cache.json"
                        ),
                        "coordinate_cache_report": str(
                            parallel_output / "coordinate-cache-report.json"
                        ),
                    },
                }},
                "requested_modules": ["structural_integrity_qc"],
                "protected_locations": [],
            }), encoding="utf-8")
            qc = structural_qc_project(project, hash_content=True)
            self.assertEqual(qc["technical_status"], "complete", qc)
            self.assertEqual(qc["parallel_execution"]["workers_used"], 2)
            self.assertEqual(qc["parallel_execution"]["shard_count"], 2)
            self.assertEqual(qc["frame_selection"]["selected_frame_count"], 6)
            self.assertEqual(len(qc["systems"][0]["replicas"]), 2)
            self.assertEqual(
                qc["periodic_coordinate_policy"], "preprocessed_make_whole"
            )

            alternate_pdb = root / "alternate-system.pdb"
            alternate_pdb.write_text(
                "HETATM    1  O   TIP W   2       5.000   0.000   0.000  1.00  0.00           O\n"
                "HETATM    2  H1  TIP W   2       5.500   0.000   0.000  1.00  0.00           H\n"
                "ATOM      3  C1  LIG A   1       9.500   0.000   0.000  1.00  0.00           C\n"
                "ATOM      4  C2  LIG A   1       0.500   0.000   0.000  1.00  0.00           C\n"
                "HETATM    5 MG   MG  M   3       8.000   0.000   0.000  1.00  0.00          MG\n"
                "END\n",
                encoding="utf-8",
            )
            alternate_bonds = root / "alternate-system.bonds.json"
            alternate_bonds.write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 5,
                "index_base": 0, "bonds": [[0, 1], [2, 3]],
            }), encoding="utf-8")
            heterogeneous_system = json.loads(json.dumps(system))
            heterogeneous_system["systems"].append({
                "system_id": "alternate",
                "replicas": [{
                    "replica_id": "replica-1",
                    "topology": str(alternate_pdb),
                    "connectivity": str(alternate_bonds),
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
            })
            heterogeneous_manifest = root / "heterogeneous-system.json"
            heterogeneous_manifest.write_text(
                json.dumps(heterogeneous_system), encoding="utf-8"
            )
            heterogeneous_cache = root / "heterogeneous-cache"
            built_heterogeneous = build_coordinate_cache(
                heterogeneous_manifest, heterogeneous_cache, maximum_workers=3
            )
            self.assertEqual(built_heterogeneous["technical_status"], "complete")
            heterogeneous_project = root / "heterogeneous-project.json"
            heterogeneous_project.write_text(json.dumps({
                "project_id": "heterogeneous-cache-routing",
                "analysis_profile": "standard_md_v1",
                "system_manifest": str(heterogeneous_manifest),
                "analysis_output_root": str(root / "heterogeneous-output"),
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "unwrap_continuous",
                "periodic_reconstruction": {
                    "maximum_bond_length_angstrom": 3.0,
                    "cycle_closure_tolerance_angstrom": 0.25,
                    "maximum_anchor_displacement_angstrom": 20.0,
                },
                "reference_structure": str(pdb),
                "reference_connectivity": str(bonds),
                "selections": {
                    "alignment": {"preset": "heavy"},
                    "analysis": {"preset": "heavy"},
                },
                "definitions": {"replica_rmsd_rg": {}},
                "requested_modules": ["replica_rmsd_rg"],
                "protected_locations": [],
            }), encoding="utf-8")
            heterogeneous_routing = materialize_cache_backed_base_project(
                heterogeneous_project,
                heterogeneous_cache,
                root / "heterogeneous-project-cache-base.json",
            )
            self.assertFalse(
                heterogeneous_routing[
                    "source_index_mappings_identical_across_replicas"
                ]
            )
            self.assertEqual(
                len(heterogeneous_routing["per_system_cache_projects"]), 2
            )
            explicit_project = load_json(heterogeneous_project)
            explicit_project["definitions"]["replica_rmsd_rg"] = {
                "atom_indices": [0, 1]
            }
            explicit_decision = cache_compatibility(
                "replica_rmsd_rg", explicit_project
            )
            self.assertFalse(explicit_decision["cache_compatible"])
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
