import unittest

from salsbury_md_analysis.scalar_threshold_states import analyze_threshold_state


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


class ScalarThresholdStateTests(unittest.TestCase):
    def test_ion_binding_threshold_has_segment_safe_runs_and_sensitivity(self):
        segments = [
            ({"system_id": "bound", "replica_id": "r1", "segment_id": "a"}, _records([2.8, 3.0, 4.2])),
            ({"system_id": "bound", "replica_id": "r1", "segment_id": "b"}, _records([4.1, 2.9], 10)),
            ({"system_id": "unbound", "replica_id": "r1", "segment_id": "a"}, _records([4.5, 4.2, 3.8])),
        ]
        report = analyze_threshold_state(
            segments,
            operator="less_than_or_equal",
            threshold=3.2,
            sensitivity_thresholds=[3.0, 3.2, 3.5],
            meets_threshold_label="ion_bound",
            does_not_meet_threshold_label="ion_unbound",
        )
        self.assertEqual(sum(row["meets_threshold"] for row in report["assignments"]), 3)
        self.assertEqual(
            [(row["segment_id"], row["length_frames"]) for row in report["residence_runs"][:4]],
            [("a", 2), ("a", 1), ("b", 1), ("b", 1)],
        )
        self.assertEqual(
            [row["threshold"] for row in report["threshold_sensitivity"]],
            [3.0, 3.2, 3.5],
        )
        systems = report["state_population_comparison"]["system_populations"]
        self.assertEqual([row["system_id"] for row in systems], ["bound", "unbound"])
        self.assertTrue(all(
            row["from_state_id"] in {1, 2} and row["to_state_id"] in {1, 2}
            for row in report["transition_counts_within_segments"]
        ))


if __name__ == "__main__":
    unittest.main()
