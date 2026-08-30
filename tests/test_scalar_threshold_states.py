import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.scalar_threshold_states import (
    analyze_threshold_state,
    scalar_threshold_states_project,
)


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
    def test_project_reuses_validated_trajectory_feature_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project.json"
            project.write_text(json.dumps({
                "definitions": {"scalar_threshold_states": {
                    "source": "trajectory_features",
                    "maximum_observations": 2,
                    "states": [{
                        "state_analysis_id": "s", "question": "q",
                        "feature_id": "f", "value_index": 0,
                        "operator": "less_than_or_equal", "threshold": 3.0,
                        "sensitivity_thresholds": [2.5, 3.0, 3.5],
                        "meets_threshold_label": "bound",
                        "does_not_meet_threshold_label": "unbound",
                    }],
                }},
            }), encoding="utf-8")
            upstream = {
                "project_manifest_sha256": "a" * 64,
                "system_manifest_path": "/system.json",
                "system_manifest_sha256": "b" * 64,
                "input_content_signature_sha256": "c" * 64,
                "segments": [{
                    "system_id": "x", "replica_id": "r", "segment_id": "g",
                    "features": [{
                        "feature_id": "f", "dimension": 1,
                        "records": [
                            {"source_frame_index": 0, "axis_kind": "physical_time", "axis_value": 0.0, "values": [2.0]},
                            {"source_frame_index": 1, "axis_kind": "physical_time", "axis_value": 1.0, "values": [4.0]},
                        ],
                    }],
                }],
                "issues": [],
            }
            with patch(
                "salsbury_md_analysis.scalar_threshold_states.load_cached_project_report",
                return_value=upstream,
            ), patch(
                "salsbury_md_analysis.scalar_threshold_states.trajectory_features_project",
                side_effect=AssertionError("trajectory must not be recomputed"),
            ), patch.dict(os.environ, {
                "SALSBURY_MD_ANALYSIS_COLUMNAR_ARTIFACT_ROOT": str(
                    Path(temporary) / "derived-artifacts"
                )
            }):
                report = scalar_threshold_states_project(project)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["observation_count"], 2)
        self.assertEqual(
            report["trajectory_feature_source_mode"],
            "validated_upstream_report",
        )
        state = report["state_reports"][0]
        self.assertIsNone(state["assignments"])
        self.assertTrue(state["assignments_retained"])
        self.assertEqual(len(state["assignment_artifacts"]), 1)

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

    def test_threshold_reducer_consumes_stream_once(self):
        calls = []
        def source():
            calls.append(1)
            yield from _records([2.8, 3.4, 2.9, 4.0])
        report = analyze_threshold_state(
            [({"system_id": "s", "replica_id": "r", "segment_id": "a"}, source)],
            operator="less_than_or_equal", threshold=3.2,
            sensitivity_thresholds=[3.0, 3.2, 3.5],
            meets_threshold_label="bound",
            does_not_meet_threshold_label="unbound",
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["observation_count"], 4)
        self.assertEqual(
            report["reducer_mode"],
            "single_pass_streaming_state_and_sensitivity_reducers",
        )


if __name__ == "__main__":
    unittest.main()
