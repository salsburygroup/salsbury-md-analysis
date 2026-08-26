import unittest

from salsbury_md_analysis.interaction_persistence import (
    build_interaction_persistence,
)


class InteractionPersistenceTests(unittest.TestCase):
    def settings(self):
        return {
            "source_module": "interaction_fingerprints",
            "gap_tolerance_observations": [0, 1],
            "minimum_observations_per_series": 5,
            "minimum_complete_events": 2,
            "maximum_features": 20,
            "maximum_event_records": 100,
            "maximum_interval_relative_deviation": 0.01,
        }

    def test_complete_and_boundary_censored_events_remain_distinct(self):
        states = [1, 1, 0, 1, 1, 0, 0, 1, 0, 0]
        report = {
            "availability_status": "available",
            "feature_dictionary": [{
                "feature_id": "hb|direct|one",
                "source_module": "hydrogen_bond_discovery",
                "interaction_type": "direct_hydrogen_bond",
            }],
            "frame_fingerprints": [{
                "system_id": "K-retained", "replica_id": "r1",
                "segment_id": "production", "source_frame_index": index,
                "available_source_modules": ["hydrogen_bond_discovery"],
                "present_feature_ids": ["hb|direct|one"] if present else [],
            } for index, present in enumerate(states)],
        }
        result = build_interaction_persistence(
            report, self.settings(), {
                ("K-retained", "r1", "production"): {
                    "kind": "physical_time", "first": 0.0,
                    "interval": 0.5, "unit": "ns",
                }
            },
        )
        self.assertEqual(result["availability_status"], "available")
        primary = next(
            row for row in result["feature_persistence_summaries"]
            if row["gap_tolerance_observations"] == 0
        )
        self.assertEqual(primary["event_count"], 3)
        self.assertEqual(primary["complete_event_count"], 2)
        self.assertEqual(primary["boundary_censored_event_count"], 1)
        self.assertEqual(primary["persistence_summary_gate"], "passed")
        self.assertAlmostEqual(
            primary["complete_event_duration_summary"]["median"], 0.75
        )
        intermittent = next(
            row for row in result["feature_persistence_summaries"]
            if row["gap_tolerance_observations"] == 1
        )
        self.assertEqual(intermittent["event_count"], 2)
        bridged = [
            row for row in result["event_records"]
            if row["gap_tolerance_observations"] == 1
            and row["start_source_frame_index"] == 0
        ][0]
        self.assertEqual(bridged["bridged_absent_observation_count"], 1)
        self.assertEqual(bridged["duration"], 2.5)

    def test_missing_source_frame_is_not_encoded_as_negative(self):
        report = {
            "availability_status": "available",
            "feature_dictionary": [{
                "feature_id": "ion|shell|one", "source_module": "ion_atmosphere",
                "interaction_type": "ion_shell_presence",
            }],
            "frame_fingerprints": [{
                "system_id": "system", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": index,
                "available_source_modules": (
                    ["other"] if index == 2 else ["ion_atmosphere"]
                ),
                "present_feature_ids": (
                    ["ion|shell|one"] if index in {0, 1, 3, 4} else []
                ),
            } for index in range(6)],
        }
        settings = self.settings()
        settings["minimum_observations_per_series"] = 4
        settings["maximum_interval_relative_deviation"] = 1.0
        result = build_interaction_persistence(
            report, settings, {("system", "r1", "s1"): {
                "kind": "physical_time", "first": 0.0,
                "interval": 1.0, "unit": "ps",
            }},
        )
        summary = next(
            row for row in result["feature_persistence_summaries"]
            if row["gap_tolerance_observations"] == 0
        )
        self.assertEqual(summary["evaluated_observation_count"], 5)
        self.assertEqual(summary["present_observation_count"], 4)

    def test_irregular_source_schedule_fails_closed_for_that_series(self):
        report = {
            "availability_status": "available",
            "feature_dictionary": [{
                "feature_id": "f", "source_module": "ion_atmosphere",
                "interaction_type": "ion_shell_presence",
            }],
            "frame_fingerprints": [{
                "system_id": "system", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": index,
                "available_source_modules": ["ion_atmosphere"],
                "present_feature_ids": ["f"],
            } for index in [0, 1, 2, 8, 9]],
        }
        result = build_interaction_persistence(
            report, self.settings(), {("system", "r1", "s1"): {
                "kind": "physical_time", "first": 0.0,
                "interval": 1.0, "unit": "ps",
            }},
        )
        self.assertEqual(result["availability_status"], "not_available")
        self.assertEqual(
            result["unavailable_series"][0]["reason"],
            "irregular_source_observation_interval",
        )


if __name__ == "__main__":
    unittest.main()
