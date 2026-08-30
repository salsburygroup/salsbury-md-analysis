import unittest

from salsbury_md_analysis.trajectory_contracts import (
    TrajectoryContractError,
    enforce_periodic_policy,
    frame_axis_value,
    frame_time,
    normalize_sample_axis,
    normalize_segment_axis,
    normalize_segment_timing,
    require_periodic_policy,
)


class TrajectoryContractTests(unittest.TestCase):
    def test_physical_time_axis_converts_units_and_indexes_frames(self):
        segment = {
            "timing": {
                "first_frame_time": 1.0,
                "frame_interval": 0.5,
                "unit": "ns",
            }
        }
        timing = normalize_segment_timing(segment, "ps")
        self.assertEqual(timing["first_frame_time"], 1000.0)
        self.assertEqual(timing["frame_interval"], 500.0)
        self.assertEqual(frame_time(timing, 3), 2500.0)
        axis = normalize_segment_axis(segment, "ps")
        self.assertEqual(frame_axis_value(axis, 3), 2500.0)

    def test_sample_axis_is_integer_and_mutually_exclusive_with_time(self):
        sample = {"sample_axis": {"first_sample_index": 4, "sample_interval": 3}}
        self.assertEqual(
            normalize_sample_axis(sample),
            {"first_sample_index": 4, "sample_interval": 3, "unit": "sample"},
        )
        self.assertEqual(frame_axis_value(normalize_segment_axis(sample, None), 2), 10)
        with self.assertRaisesRegex(TrajectoryContractError, "exactly one"):
            normalize_segment_axis({**sample, "timing": {"unit": "ps"}}, "ps")

    def test_invalid_timing_and_indices_fail_closed(self):
        with self.assertRaises(TrajectoryContractError):
            normalize_segment_timing(
                {"timing": {"first_frame_time": 0, "frame_interval": 0, "unit": "ps"}},
                "ps",
            )
        with self.assertRaises(TrajectoryContractError):
            frame_time({"first_frame_time": 0, "frame_interval": 1}, -1)
        with self.assertRaises(TrajectoryContractError):
            normalize_sample_axis(
                {"sample_axis": {"first_sample_index": True, "sample_interval": 1}}
            )

    def test_periodic_policy_rejects_wrapped_periodic_input_by_default(self):
        self.assertEqual(require_periodic_policy("unwrap_continuous"), "unwrap_continuous")
        with self.assertRaises(TrajectoryContractError):
            require_periodic_policy("guess")
        with self.assertRaisesRegex(TrajectoryContractError, "blocks periodic analysis"):
            enforce_periodic_policy(True, "reject", "replica-1")
        enforce_periodic_policy(True, "make_whole", "replica-1")


if __name__ == "__main__":
    unittest.main()
