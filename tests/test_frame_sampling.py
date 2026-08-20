import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.frame_sampling import (
    integer_stride_for_budget,
    integer_stride_indices,
    integer_stride_selected_count,
    normalize_frame_selection,
    plan_frame_selection,
    reader_frame_indices,
    uniform_indices,
)


def _xyz(frame_count: int) -> str:
    return "".join(
        f"1\nframe-{index}\nC {index} 0 0\n" for index in range(frame_count)
    )


class FrameSamplingTests(unittest.TestCase):
    def test_uniform_indices_cover_full_timespan_without_duplicates(self):
        self.assertEqual(uniform_indices(10, 4), {0, 3, 6, 9})
        self.assertEqual(uniform_indices(3, 10), {0, 1, 2})
        self.assertEqual(uniform_indices(9, 1), {4})

    def test_uniform_selection_requires_stride_one(self):
        with self.assertRaisesRegex(ValueError, "requires frame_stride = 1"):
            normalize_frame_selection(
                {
                    "mode": "uniform_per_replica_budget_v1",
                    "maximum_frames_per_replica": 3,
                },
                2,
            )

    def test_integer_stride_budget_uses_finest_stride_within_ceiling(self):
        self.assertEqual(integer_stride_for_budget([24_700] * 6, 15_360), 2)
        self.assertEqual(integer_stride_selected_count(24_700, 2), 12_350)
        self.assertEqual(integer_stride_for_budget([10], 4), 3)
        self.assertEqual(integer_stride_indices(10, 3), {0, 3, 6, 9})

    def test_integer_stride_selection_requires_stride_one_wrapper(self):
        with self.assertRaisesRegex(ValueError, "requires frame_stride = 1"):
            normalize_frame_selection(
                {"mode": "integer_stride_per_replica_v1", "stride": 3},
                2,
            )

    def test_automatic_resource_budget_subsamples_and_reports_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("r1.xyz", "r2.xyz"):
                (root / name).write_text(_xyz(10), encoding="ascii")
            manifest = {"systems": [{"system_id": "s", "replicas": [
                {"replica_id": "r1", "segments": [
                    {"segment_id": "a", "trajectory": "r1.xyz"},
                ]},
                {"replica_id": "r2", "segments": [
                    {"segment_id": "a", "trajectory": "r2.xyz"},
                ]},
            ]}]}
            selection = normalize_frame_selection({
                "mode": "auto_resource_budget_v1",
                "target_wall_seconds": 12.0,
                "estimated_seconds_per_frame": 1.0,
                "minimum_frames_per_replica": 2,
                "safety_factor": 1.5,
                "sensitivity_check_policy": "recommend",
                "calibration_id": "unit-test",
            }, 1)
            plan, report = plan_frame_selection(
                manifest, root / "system.json", "angstrom", selection,
                frame_stride=1,
            )
        self.assertEqual(plan[("s", "r1", "a")], {0, 3, 6, 9})
        self.assertEqual(plan[("s", "r2", "a")], {0, 3, 6, 9})
        self.assertEqual(report["mode"], "auto_resource_budget_v1")
        self.assertEqual(report["resolved_mode"], "integer_stride_per_replica_v1")
        self.assertEqual(report["selected_frame_count"], 8)
        self.assertTrue(report["resource_estimate"]["subsampling_triggered"])
        self.assertEqual(
            report["replicas"][0]["selection_spacing"],
            {
                "kind": "exact_integer_stride",
                "random": False,
                "minimum_source_frame_gap": 3,
                "maximum_source_frame_gap": 3,
                "mean_source_frame_gap": 3.0,
                "starts_at_replica_frame_zero": True,
                "last_source_frame_forced": False,
            },
        )
        self.assertEqual(
            report["resource_estimate"]["sensitivity_check_policy"], "recommend"
        )

    def test_automatic_resource_budget_uses_all_frames_when_they_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "r.xyz").write_text(_xyz(4), encoding="ascii")
            manifest = {"systems": [{"system_id": "s", "replicas": [{
                "replica_id": "r", "segments": [
                    {"segment_id": "a", "trajectory": "r.xyz"},
                ],
            }]}]}
            selection = normalize_frame_selection({
                "mode": "auto_resource_budget_v1",
                "target_wall_seconds": 100.0,
                "estimated_seconds_per_frame": 1.0,
                "minimum_frames_per_replica": 2,
                "sensitivity_check_policy": "off",
            }, 1)
            plan, report = plan_frame_selection(
                manifest, root / "system.json", "angstrom", selection,
                frame_stride=1,
            )
        self.assertIsNone(plan[("s", "r", "a")])
        self.assertEqual(report["resolved_mode"], "fixed_stride_v1")
        self.assertFalse(report["resource_estimate"]["subsampling_triggered"])

    def test_plan_balances_each_replica_over_concatenated_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, count in (("r1a.xyz", 2), ("r1b.xyz", 6), ("r2.xyz", 5)):
                (root / name).write_text(_xyz(count), encoding="ascii")
            manifest = {"systems": [{"system_id": "s", "replicas": [
                {"replica_id": "r1", "segments": [
                    {"segment_id": "a", "trajectory": "r1a.xyz"},
                    {"segment_id": "b", "trajectory": "r1b.xyz"},
                ]},
                {"replica_id": "r2", "segments": [
                    {"segment_id": "a", "trajectory": "r2.xyz"},
                ]},
            ]}]}
            plan, report = plan_frame_selection(
                manifest,
                root / "system.json",
                "angstrom",
                {
                    "mode": "uniform_per_replica_budget_v1",
                    "maximum_frames_per_replica": 3,
                },
                frame_stride=1,
            )
        self.assertEqual(plan[("s", "r1", "a")], {0})
        self.assertEqual(plan[("s", "r1", "b")], {1, 5})
        self.assertEqual(plan[("s", "r2", "a")], {0, 2, 4})
        self.assertEqual(report["source_frame_count"], 13)
        self.assertEqual(report["selected_frame_count"], 6)
        self.assertEqual(
            [row["selected_frame_count"] for row in report["replicas"]], [3, 3]
        )

    def test_continuous_unwrap_disables_reader_level_skipping(self):
        indices = {0, 9}
        self.assertIsNone(reader_frame_indices(indices, "unwrap_continuous"))
        self.assertIs(reader_frame_indices(indices, "make_whole"), indices)

    def test_integer_stride_continues_across_segment_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.xyz").write_text(_xyz(4), encoding="ascii")
            (root / "b.xyz").write_text(_xyz(4), encoding="ascii")
            manifest = {"systems": [{"system_id": "s", "replicas": [{
                "replica_id": "r", "segments": [
                    {"segment_id": "a", "trajectory": "a.xyz"},
                    {"segment_id": "b", "trajectory": "b.xyz"},
                ],
            }]}]}
            plan, report = plan_frame_selection(
                manifest, root / "system.json", "angstrom",
                {"mode": "integer_stride_per_replica_v1", "stride": 3},
                frame_stride=1,
            )
        self.assertEqual(plan[("s", "r", "a")], {0, 3})
        self.assertEqual(plan[("s", "r", "b")], {2})
        self.assertEqual(report["selected_frame_count"], 3)
        self.assertEqual(report["resolved_integer_stride"], 3)

    def test_fixed_stride_restarts_at_each_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.xyz").write_text(_xyz(3), encoding="ascii")
            (root / "b.xyz").write_text(_xyz(4), encoding="ascii")
            manifest = {"systems": [{"system_id": "s", "replicas": [{
                "replica_id": "r", "segments": [
                    {"segment_id": "a", "trajectory": "a.xyz"},
                    {"segment_id": "b", "trajectory": "b.xyz"},
                ],
            }]}]}
            plan, report = plan_frame_selection(
                manifest, root / "system.json", "angstrom",
                {"mode": "fixed_stride_v1"}, frame_stride=2,
            )
        self.assertEqual(plan[("s", "r", "a")], {0, 2})
        self.assertEqual(plan[("s", "r", "b")], {0, 2})
        self.assertEqual(report["selected_frame_count"], 4)


if __name__ == "__main__":
    unittest.main()
