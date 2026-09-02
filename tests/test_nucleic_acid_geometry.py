import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.nucleic_acid_geometry import (
    fit_plane,
    nucleic_acid_geometry_project,
    planar_departure_degrees,
    ring_geometry,
)


def _atom(serial, name, residue, resid, x, y, z):
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_project(
    root: Path, *, frame_count: int = 2, frame_stride: int = 1,
    maximum_frames: int = 10, segment_frame_counts=None,
) -> Path:
    first = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    second = [(-1, -1, 3), (1, -1, 3), (1, 1, 3), (-1, 1, 3)]
    topology = "".join(
        _atom(index + 1, f"R{index + 1}", "DA", 1, *point)
        for index, point in enumerate(first)
    )
    topology += "".join(
        _atom(index + 5, f"S{index + 1}", "DT", 2, *point)
        for index, point in enumerate(second)
    ) + "END\n"
    (root / "topology.pdb").write_text(topology, encoding="ascii")
    frames = [
        (first + second)
        if frame_index == 0 else
        (first[:-1] + [(-1, 1, 0.4)] + second)
        for frame_index in range(frame_count)
    ]
    segment_frame_counts = segment_frame_counts or (frame_count,)
    assert sum(segment_frame_counts) == frame_count
    segments = []
    offset = 0
    for segment_index, count in enumerate(segment_frame_counts, start=1):
        trajectory = ""
        for local_index, points in enumerate(frames[offset:offset + count]):
            trajectory += f"8\nframe {local_index}\n"
            trajectory += "".join(f"C {x} {y} {z}\n" for x, y, z in points)
        trajectory_name = f"trajectory-{segment_index}.xyz"
        (root / trajectory_name).write_text(trajectory, encoding="ascii")
        segments.append({
            "segment_id": f"s{segment_index}", "trajectory": trajectory_name,
            "timing": {
                "first_frame_time": offset, "frame_interval": 1, "unit": "ps",
            },
        })
        offset += count
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "dna", "replicas": [{
            "replica_id": "r1", "topology": "topology.pdb",
            "segments": segments,
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "nucleic-geometry-test", "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json", "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
        "time_unit": "ps", "periodic_coordinate_policy": "reject",
        "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
        "definitions": {"nucleic_acid_geometry": {
            "frame_stride": frame_stride, "maximum_frames": maximum_frames,
            "rings": [
                {"ring_id": "base1", "atom_indices": [0, 1, 2, 3]},
                {"ring_id": "base2", "atom_indices": [4, 5, 6, 7]},
            ],
            "plane_pairs": [{
                "pair_id": "stack12", "first_ring_id": "base1",
                "second_ring_id": "base2", "interpretation": "base_stacking",
            }],
            "block_count": 2, "histogram_rule": "scott",
            "histogram_padding_fraction": 0.05,
            "minimum_histogram_bins": 2, "maximum_histogram_bins": 20,
        }},
        "requested_modules": ["nucleic_acid_geometry"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class NucleicAcidGeometryTests(unittest.TestCase):
    def test_plane_and_ring_metrics_distinguish_flat_from_puckered(self):
        planar = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        puckered = planar[:-1] + [(-1, 1, 0.5)]
        self.assertAlmostEqual(fit_plane(planar)["rms_displacement_angstrom"], 0.0)
        self.assertAlmostEqual(ring_geometry(planar)["torsion_rms_planar_departure_degrees"], 0.0)
        self.assertGreater(ring_geometry(puckered)["rms_displacement_angstrom"], 0.0)
        self.assertAlmostEqual(planar_departure_degrees(179.0), -1.0)

    def test_project_reports_independent_scott_distributions_and_stationarity(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = nucleic_acid_geometry_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["evaluated_frame_count"], 2)
        self.assertTrue(
            report["frame_report_encoding"]
            ["raw_scalar_metrics_retained_for_every_selected_frame"]
        )
        self.assertNotIn("rings", report["frame_reports"][0])
        self.assertIn("metrics", report["frame_reports"][0])
        self.assertIn("ring:base1:plane_rms_angstrom", report["metric_ids"])
        distribution = next(
            row for row in report["distribution_reports"]
            if row["metric_id"] == "ring:base1:plane_rms_angstrom"
        )
        self.assertEqual(distribution["status"], "complete")
        self.assertEqual(distribution["binning"]["rule"], "scott")
        self.assertIsNone(distribution["assignments"])
        self.assertFalse(distribution["assignments_retained"])
        self.assertIsNone(distribution["residence_runs"])
        self.assertFalse(distribution["residence_runs_retained"])
        self.assertEqual(sum(row["count"] for row in distribution["histogram"]), 2)
        drift = report["replica_reports"][0]["late_minus_early_metric_means"]
        self.assertGreater(drift["ring:base1:plane_rms_angstrom"], 0.0)

    def test_stride_uses_complete_intervals_and_does_not_exceed_planned_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = nucleic_acid_geometry_project(_write_project(
                Path(temporary), frame_count=30, frame_stride=29,
                maximum_frames=1,
            ))
        self.assertEqual(report["evaluated_frame_count"], 1)
        self.assertEqual(report["frame_reports"][0]["source_frame_index"], 0)

    def test_stride_continues_across_replica_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = nucleic_acid_geometry_project(_write_project(
                Path(temporary), frame_count=60, frame_stride=29,
                maximum_frames=2, segment_frame_counts=(30, 30),
            ))
        self.assertEqual(report["evaluated_frame_count"], 2)
        self.assertEqual(
            [(row["segment_id"], row["source_frame_index"])
             for row in report["frame_reports"]],
            [("s1", 0), ("s1", 29)],
        )


if __name__ == "__main__":
    unittest.main()
