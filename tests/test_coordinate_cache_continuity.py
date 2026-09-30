"""Synthetic payload regressions for segmented cache timing and provenance."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.coordinate_cache import (
    build_coordinate_cache,
    validate_reusable_coordinate_cache,
)
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.manifests import load_json, sha256_file
from salsbury_md_analysis.preflight import preflight_system, probe_trajectory


def _record(payload):
    marker = struct.pack("<i", len(payload))
    return marker + payload + marker


def _write_dcd(path, frame_count, source_offset, start=5000):
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, frame_count, start, 50000)
    struct.pack_into("<i", header, 44, 1)
    struct.pack_into("<i", header, 80, 24)
    with path.open("wb") as handle:
        handle.write(_record(bytes(header)))
        handle.write(_record(struct.pack("<i", 1) + b"synthetic cache test".ljust(80)))
        handle.write(_record(struct.pack("<i", 4)))
        for local_index in range(frame_count):
            global_index = source_offset + local_index
            x = 9.5 + global_index * 0.001
            handle.write(_record(struct.pack(
                "<6d", 10.0, 90.0, 11.0, 90.0, 90.0,
                12.0 + (global_index % 3) * 0.25,
            )))
            axes = ((x % 10, (x + 1) % 10, 4.0, 8.0),
                    (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
            for values in axes:
                handle.write(_record(struct.pack("<4f", *values)))


def _fixture(root, lengths=(4, 4), replicas=1, policy="reset_per_segment"):
    topology = root / "system.pdb"
    topology.write_text(
        "ATOM      1  C1  LIG A   1       9.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C2  LIG A   1       0.500   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  O   HOH W   2       4.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    4  K   K   K   3       8.000   0.000   0.000  1.00  0.00           K\n"
        "END\n", encoding="utf-8",
    )
    connectivity = root / "system.bonds.json"
    connectivity.write_text(json.dumps({
        "format": "salsbury-bonds-v1", "atom_count": 4,
        "index_base": 0, "bonds": [[0, 1]],
    }), encoding="utf-8")
    replica_rows = []
    for replica_index in range(replicas):
        segments = []
        offset = 0
        for segment_index, length in enumerate(lengths):
            trajectory = root / f"rep{replica_index + 1}-{segment_index + 1}.dcd"
            start = 5000 if policy == "reset_per_segment" else 5000 + offset * 50000
            _write_dcd(trajectory, length, offset, start=start)
            segment = {
                "segment_id": f"chunk_{segment_index + 1:03d}",
                "trajectory": str(trajectory),
                "timing": {"first_frame_time": 100.0 + offset * 100.0,
                           "frame_interval": 100.0, "unit": "ps"},
            }
            if policy is not None:
                segment["dcd_header_step_policy"] = policy
            if segment_index:
                segment["continuous_with_previous"] = True
            segments.append(segment)
            offset += length
        replica_rows.append({
            "replica_id": f"rep{replica_index + 1}",
            "topology": str(topology), "connectivity": str(connectivity),
            "segments": segments,
        })
    data = {"systems": [{"system_id": "synthetic-segmented",
                          "replicas": replica_rows}]}
    manifest = root / "system.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    return data, manifest


class CoordinateCacheContinuityTests(unittest.TestCase):
    def _preflight(self, manifest):
        return preflight_system(load_json(manifest), manifest, hash_content=True)

    def test_four_replicas_seven_restarting_fragments_payload_lineage_and_24_joins(self):
        # Same segment lengths, physical timing and layout as the corrected
        # incident: four 1,000-frame + three 2,000-frame fragments per replica.
        # Coordinates are entirely synthetic and are not scientific fixtures.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = _fixture(
                root, lengths=(1000, 1000, 1000, 1000, 2000, 2000, 2000),
                replicas=4,
            )
            source_hashes = {p: sha256_file(p) for p in root.iterdir() if p.is_file()}
            initial = self._preflight(manifest)
            self.assertEqual(initial["technical_status"], "complete", initial["issues"])
            self.assertEqual(sum(i["code"] == "DCD_HEADER_STEP_RESET"
                                 for i in initial["issues"]), 24)
            output = root / "cache"
            report = build_coordinate_cache(manifest, output, hash_source_content=True)
            self.assertEqual(report["technical_status"], "complete")
            cached_manifest = output / "system-cache.json"
            cached = load_json(cached_manifest)
            validated = self._preflight(cached_manifest)
            self.assertEqual(validated["technical_status"], "complete", validated["issues"])
            self.assertEqual(sum(i["code"] == "DCD_HEADER_STEP_RESET"
                                 for i in validated["issues"]), 24)
            missing_policy = json.loads(json.dumps(cached))
            for replica in missing_policy["systems"][0]["replicas"]:
                for segment in replica["segments"]:
                    segment.pop("dcd_header_step_policy")
            old_behavior = preflight_system(missing_policy, cached_manifest)
            self.assertEqual(old_behavior["technical_status"], "failed")
            self.assertEqual(sum(i["code"] == "DCD_CONTINUITY_MISMATCH"
                                 for i in old_behavior["issues"]), 24)
            source_replicas = data["systems"][0]["replicas"]
            cached_replicas = cached["systems"][0]["replicas"]
            for source_replica, cached_replica, row in zip(
                source_replicas, cached_replicas, report["rows"]
            ):
                self.assertEqual(cached_replica["replica_id"], source_replica["replica_id"])
                self.assertEqual(row["source_atom_indices_in_cache_order"], [0, 1, 3])
                self.assertEqual(row["decoded_frame_count"], 10000)
                self.assertEqual(row["retained_frame_count"], 10000)
                offset = 0
                for source, derived, lineage in zip(
                    source_replica["segments"], cached_replica["segments"], row["segments"]
                ):
                    self.assertEqual(derived["segment_id"], source["segment_id"])
                    self.assertEqual(derived["timing"], source["timing"])
                    self.assertEqual(derived.get("continuous_with_previous"),
                                     source.get("continuous_with_previous"))
                    self.assertEqual(derived["dcd_header_step_policy"], "reset_per_segment")
                    self.assertEqual(lineage["source"]["sha256"], source_hashes[Path(source["trajectory"])])
                    self.assertEqual(lineage["first_retained_source_frame_index"], 0)
                    self.assertEqual(lineage["cache_stride"], 1)
                    path = output / derived["trajectory"]
                    self.assertEqual(lineage["cache"]["sha256"], sha256_file(path))
                    self.assertEqual(probe_trajectory(path)["starting_step"], 5000)
                    observed = list(iter_coordinate_frames(path, "angstrom"))
                    source_frames = list(iter_coordinate_frames(Path(source["trajectory"]), "angstrom"))
                    self.assertEqual(len(observed), lineage["source_frame_count"])
                    # Check every retained payload, not just the header or a
                    # representative. The second atom requires make-whole and
                    # the molecule crosses a periodic boundary in chunk one.
                    expected = np.asarray([
                        [[9.5 + (offset + i) * .001, 0, 0],
                         [10.5 + (offset + i) * .001, 0, 0], [8.0, 0, 0]]
                        for i in range(len(observed))
                    ])
                    np.testing.assert_allclose(
                        [f.coordinates_angstrom for f in observed], expected,
                        rtol=0, atol=2e-6,
                    )
                    np.testing.assert_allclose(
                        [f.cell_vectors_angstrom for f in observed],
                        [f.cell_vectors_angstrom for f in source_frames],
                        rtol=0, atol=1e-12,
                    )
                    offset += len(observed)
            self.assertEqual(validate_reusable_coordinate_cache(output, manifest)["technical_status"], "complete")
            self.assertEqual({p: sha256_file(p) for p in source_hashes}, source_hashes)

    def test_real_time_gaps_and_overlaps_still_fail_after_caching(self):
        for delta in (-100.0, 100.0):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, manifest = _fixture(root)
                data["systems"][0]["replicas"][0]["segments"][1]["timing"]["first_frame_time"] += delta
                manifest.write_text(json.dumps(data), encoding="utf-8")
                output = root / "cache"
                build_coordinate_cache(manifest, output)
                for path in (manifest, output / "system-cache.json"):
                    report = self._preflight(path)
                    self.assertEqual(report["technical_status"], "failed")
                    self.assertIn("PHYSICAL_TIME_CONTINUITY_MISMATCH",
                                  {i["code"] for i in report["issues"]})

    def test_unexpected_counter_change_is_not_treated_as_declared_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = _fixture(root)
            second = data["systems"][0]["replicas"][0]["segments"][1]
            _write_dcd(Path(second["trajectory"]), 4, 4, start=6000)
            output = root / "cache"
            build_coordinate_cache(manifest, output)
            for path in (manifest, output / "system-cache.json"):
                report = self._preflight(path)
                self.assertEqual(report["technical_status"], "failed")
                self.assertIn("DCD_CONTINUITY_MISMATCH", {i["code"] for i in report["issues"]})

    def test_single_file_without_declared_policy_remains_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = _fixture(root, lengths=(6,), policy=None)
            output = root / "cache"
            build_coordinate_cache(manifest, output, cache_stride=2)
            cached = load_json(output / "system-cache.json")
            segment = cached["systems"][0]["replicas"][0]["segments"][0]
            self.assertNotIn("dcd_header_step_policy", segment)
            self.assertEqual(segment["timing"], {
                "first_frame_time": 100.0, "frame_interval": 200.0, "unit": "ps",
            })
            self.assertEqual(self._preflight(output / "system-cache.json")["technical_status"], "complete")

    def test_explicit_continuous_and_restarting_strided_segments(self):
        for policy in ("continuous", "reset_per_segment"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, manifest = _fixture(root, policy=policy)
                output = root / "cache"
                report = build_coordinate_cache(manifest, output, cache_stride=2)
                cached = load_json(output / "system-cache.json")
                segments = cached["systems"][0]["replicas"][0]["segments"]
                self.assertTrue(all(s["dcd_header_step_policy"] == policy for s in segments))
                self.assertEqual([s["timing"]["first_frame_time"] for s in segments], [100.0, 500.0])
                self.assertTrue(all(s["timing"]["frame_interval"] == 200.0 for s in segments))
                self.assertEqual(report["rows"][0]["retained_frame_count"], 4)
                self.assertEqual(self._preflight(output / "system-cache.json")["technical_status"], "complete")
                for source, derived in zip(data["systems"][0]["replicas"][0]["segments"], segments):
                    source_frames = list(iter_coordinate_frames(Path(source["trajectory"]), "angstrom"))
                    frames = list(iter_coordinate_frames(output / derived["trajectory"], "angstrom"))
                    self.assertEqual(len(frames), 2)
                    np.testing.assert_allclose(
                        [f.cell_vectors_angstrom for f in frames],
                        [source_frames[i].cell_vectors_angstrom for i in (0, 2)],
                        rtol=0, atol=1e-12,
                    )

    def test_nondividing_stride_preserves_reset_convention_and_exact_selected_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = _fixture(root, lengths=(5, 7))
            output = root / "cache"
            report = build_coordinate_cache(manifest, output, cache_stride=3)
            cached = load_json(output / "system-cache.json")
            segments = cached["systems"][0]["replicas"][0]["segments"]
            lineage = report["rows"][0]["segments"]
            self.assertEqual([s["first_retained_source_frame_index"] for s in lineage], [0, 1])
            self.assertEqual([s["retained_frame_count"] for s in lineage], [2, 2])
            self.assertEqual([s["timing"]["first_frame_time"] for s in segments], [100.0, 700.0])
            self.assertTrue(all(s["timing"]["frame_interval"] == 300.0 for s in segments))
            self.assertEqual([probe_trajectory(output / s["trajectory"])["starting_step"]
                              for s in segments], [5000, 5000])
            report = self._preflight(output / "system-cache.json")
            self.assertEqual(report["technical_status"], "complete", report["issues"])
            self.assertIn("DCD_HEADER_STEP_RESET", {i["code"] for i in report["issues"]})
            frames = [f for s in segments
                      for f in iter_coordinate_frames(output / s["trajectory"], "angstrom")]
            np.testing.assert_allclose([f.coordinates_angstrom[0][0] for f in frames],
                                       [9.5 + i * .001 for i in (0, 3, 6, 9)],
                                       rtol=0, atol=2e-6)
            source_frames = [f for s in data["systems"][0]["replicas"][0]["segments"]
                             for f in iter_coordinate_frames(Path(s["trajectory"]), "angstrom")]
            np.testing.assert_allclose([f.cell_vectors_angstrom for f in frames],
                                       [source_frames[i].cell_vectors_angstrom for i in (0, 3, 6, 9)],
                                       rtol=0, atol=1e-12)

    def test_replica_worker_merge_preserves_reset_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = _fixture(root, lengths=(4,) * 7, replicas=4)
            output = root / "cache"
            report = build_coordinate_cache(manifest, output, maximum_workers=2)
            self.assertEqual(report["maximum_workers_used"], 2)
            checked = self._preflight(output / "system-cache.json")
            self.assertEqual(checked["technical_status"], "complete", checked["issues"])
            self.assertEqual(sum(i["code"] == "DCD_HEADER_STEP_RESET"
                                 for i in checked["issues"]), 24)

    def test_reset_declaration_also_preserves_valid_continuous_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = _fixture(root, lengths=(5, 7, 6))
            # The source contract accepts both a continuous header and a
            # subsequent restart at that header's baseline under reset policy.
            segments = data["systems"][0]["replicas"][0]["segments"]
            _write_dcd(Path(segments[1]["trajectory"]), 7, 5, start=255000)
            _write_dcd(Path(segments[2]["trajectory"]), 6, 12, start=255000)
            self.assertEqual(self._preflight(manifest)["technical_status"], "complete")
            output = root / "cache"
            build_coordinate_cache(manifest, output, cache_stride=3)
            report = self._preflight(output / "system-cache.json")
            self.assertEqual(report["technical_status"], "complete", report["issues"])
            cached_segments = load_json(output / "system-cache.json")["systems"][0]["replicas"][0]["segments"]
            self.assertEqual([probe_trajectory(output / s["trajectory"])["starting_step"]
                              for s in cached_segments], [5000, 305000, 305000])

    def test_continuous_counter_after_reset_chain_retains_derived_phase(self):
        for lengths in ((5, 7, 6), (5, 7, 5, 7, 6)):
            with self.subTest(lengths=lengths), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data, manifest = _fixture(root, lengths=lengths)
                segments = data["systems"][0]["replicas"][0]["segments"]
                final_start = 5000 + lengths[-2] * 50000
                _write_dcd(Path(segments[-1]["trajectory"]), lengths[-1],
                           sum(lengths[:-1]), start=final_start)
                hashes = {Path(s["trajectory"]): sha256_file(Path(s["trajectory"]))
                          for s in segments}
                self.assertEqual(self._preflight(manifest)["technical_status"], "complete")
                output = root / "cache"
                build_coordinate_cache(manifest, output, cache_stride=3)
                report = self._preflight(output / "system-cache.json")
                self.assertEqual(report["technical_status"], "complete", report["issues"])
                cached_segments = load_json(output / "system-cache.json")["systems"][0]["replicas"][0]["segments"]
                self.assertEqual([probe_trajectory(output / s["trajectory"])["starting_step"]
                                  for s in cached_segments],
                                 [5000] * (len(lengths) - 1) + [305000])
                selected_indices = list(range(0, (sum(lengths) // 3) * 3, 3))
                frames = [f for s in cached_segments for f in iter_coordinate_frames(
                    output / s["trajectory"], "angstrom")]
                source_frames = [f for s in segments for f in iter_coordinate_frames(
                    Path(s["trajectory"]), "angstrom")]
                np.testing.assert_allclose(
                    [f.coordinates_angstrom[0][0] for f in frames],
                    [9.5 + i * .001 for i in selected_indices], rtol=0, atol=2e-6,
                )
                np.testing.assert_allclose(
                    [f.cell_vectors_angstrom for f in frames],
                    [source_frames[i].cell_vectors_angstrom for i in selected_indices],
                    rtol=0, atol=1e-12,
                )
                self.assertEqual({p: sha256_file(p) for p in hashes}, hashes)

    def test_invalid_counter_after_reset_chain_is_never_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = _fixture(root, lengths=(5, 7, 6))
            final = data["systems"][0]["replicas"][0]["segments"][-1]
            _write_dcd(Path(final["trajectory"]), 6, 12, start=356000)
            output = root / "cache"
            build_coordinate_cache(manifest, output, cache_stride=3)
            for path in (manifest, output / "system-cache.json"):
                report = self._preflight(path)
                self.assertEqual(report["technical_status"], "failed")
                self.assertIn("DCD_CONTINUITY_MISMATCH",
                              {i["code"] for i in report["issues"]})
            cached_segments = load_json(output / "system-cache.json")["systems"][0]["replicas"][0]["segments"]
            self.assertEqual(probe_trajectory(output / cached_segments[-1]["trajectory"])["starting_step"], 356000)


if __name__ == "__main__":
    unittest.main()
