import math
import unittest

from salsbury_md_analysis.scalar_distributions import analyze_scalar_distribution


def _records(values, start=0):
    return [
        {
            "source_frame_index": start + index,
            "axis_kind": "physical_time",
            "axis_value": float(start + index),
            "value": float(value),
        }
        for index, value in enumerate(values)
    ]


class ScalarDistributionTests(unittest.TestCase):
    def test_scott_rule_matches_formula_and_assigns_every_observation(self):
        values = [float(index) for index in range(1, 101)]
        report = analyze_scalar_distribution(
            [({"system_id": "s", "replica_id": "r", "segment_id": "a"}, _records(values))],
            binning_rule="scott", padding_fraction=0.0,
            minimum_bins=2, maximum_bins=100,
        )
        mean = sum(values) / len(values)
        population_sd = math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        )
        self.assertAlmostEqual(
            report["binning"]["rule_width"],
            3.5 * population_sd * len(values) ** (-1.0 / 3.0),
        )
        self.assertEqual(
            sum(row["count"] for row in report["histogram"]), len(values)
        )
        self.assertEqual(len(report["assignments"]), len(values))

    def test_residence_runs_do_not_cross_segments_and_mark_censoring(self):
        segments = [
            ({"system_id": "s", "replica_id": "r", "segment_id": "a"}, _records([0, 0, 1])),
            ({"system_id": "s", "replica_id": "r", "segment_id": "b"}, _records([1, 1, 0], 10)),
        ]
        report = analyze_scalar_distribution(
            segments, binning_rule="explicit", padding_fraction=0.0, bin_count=2
        )
        self.assertEqual(
            [(row["segment_id"], row["length_frames"]) for row in report["residence_runs"]],
            [("a", 2), ("a", 1), ("b", 2), ("b", 1)],
        )
        self.assertTrue(report["residence_runs"][1]["right_boundary_censored"])
        self.assertTrue(report["residence_runs"][2]["left_boundary_censored"])


if __name__ == "__main__":
    unittest.main()
