import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.pald_community import (
    pald_community_analysis_project,
    regular_strided_sample,
)


def _metadata(group_frames=10):
    rows = []
    for replica_id in ("r1", "r2"):
        for member_id in ("member-1", "member-2"):
            for frame in range(group_frames):
                rows.append({
                    "system_id": "lesion",
                    "replica_id": replica_id,
                    "segment_id": "production",
                    "member_id": member_id,
                    "source_frame_index": frame,
                    "time": float(frame),
                    "time_unit": "ps",
                })
    return rows


class PaLDCommunityTests(unittest.TestCase):
    def test_regular_sampling_uses_one_stride_and_preserves_member_boundaries(self):
        rows = _metadata()
        selected, report = regular_strided_sample(rows, 12)
        self.assertEqual(report["source_frame_stride"], 4)
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {row["selected_observation_count"] for row in report["trajectory_groups"]},
            {2},
        )
        self.assertEqual(
            {row["member_id"] for row in report["trajectory_groups"]},
            {"member-1", "member-2"},
        )

    def test_project_reports_depth_strong_ties_communities_and_separate_msm(self):
        metadata = _metadata(group_frames=1)
        vectors = [(-2.0, -0.1), (-1.0, 0.1), (1.0, -0.1), (2.0, 0.1)]
        feature_report = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
        }
        cohesion = [
            [0.4, 0.3, 0.0, 0.0],
            [0.3, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.3],
            [0.0, 0.0, 0.3, 0.4],
        ]
        project = {"definitions": {
            "pald_community_analysis": {
                "feature_source": "tica",
                "component_indices": [1, 2],
                "standardize_features": True,
                "maximum_observations": 4,
                "community_msm_enabled": True,
                "maximum_reported_intercommunity_ties": 10,
            },
            "markov_state_models": {},
        }}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.pald_community.load_feature_matrix",
                return_value=(feature_report, metadata, vectors, {"columns": []}),
            ), patch(
                "salsbury_md_analysis.pald_community.partitioned_local_depths",
                return_value={"cohesion_matrix": cohesion},
            ), patch(
                "salsbury_md_analysis.pald_community._community_msm",
                return_value={"status": "complete", "model": {"state_count": 2}},
            ):
                report = pald_community_analysis_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertAlmostEqual(report["strong_tie_threshold"], 0.2)
        self.assertEqual(report["strong_tie_count"], 2)
        self.assertEqual(report["community_count"], 2)
        self.assertEqual(
            [row["sampled_population"] for row in report["communities"]],
            [2, 2],
        )
        self.assertEqual(
            [row["community_id"] for row in report["sampled_observations"]],
            [1, 1, 2, 2],
        )
        self.assertEqual(report["community_msm"]["status"], "complete")
        self.assertEqual(report["observation_accounting"], {
            "source_physical_frame_count": 2,
            "source_member_observation_count": 4,
            "selected_physical_frame_count": 2,
            "symmetry_expanded_observation_count": 4,
            "member_observations_are_independent_replicas": False,
        })


if __name__ == "__main__":
    unittest.main()
