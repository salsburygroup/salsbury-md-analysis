import unittest

from salsbury_md_analysis.information_dynamics import _pca_segments
from salsbury_md_analysis.state_populations import summarize_state_populations


class OligomerLineageTests(unittest.TestCase):
    def test_state_population_comparison_scales_to_twenty_variants(self):
        rows = [
            {
                "system_id": f"variant-{index:02d}", "replica_id": "r1",
                "segment_id": "seg", "source_frame_index": 0,
                "cluster_id": 1 + index % 2,
            }
            for index in range(20)
        ]
        report = summarize_state_populations(rows, "cluster_id")
        self.assertEqual(len(report["system_populations"]), 20)
        self.assertEqual(len(report["pairwise_system_differences"]), 190)

    def test_time_lagged_features_split_members(self):
        projections = []
        for frame in range(3):
            for member, offset in (("a", 0.0), ("b", 10.0)):
                projections.append({
                    "source_frame_index": frame,
                    "member_id": member,
                    "scores_angstrom": [frame + offset, 1.0],
                })
        report = {"systems": [{
            "system_id": "s", "replicas": [{
                "replica_id": "r", "segments": [{
                    "segment_id": "seg", "projections": projections,
                }],
            }],
        }]}
        segments, identities = _pca_segments(report, [1, 2])
        self.assertEqual(len(segments), 2)
        self.assertEqual([row[0][0] for row in segments], [0.0, 10.0])
        self.assertEqual([row["member_id"] for row in identities], ["a", "b"])

    def test_state_coupling_uses_paired_physical_frames(self):
        rows = []
        for frame, states in enumerate(((1, 1), (1, 2), (2, 2), (2, 2))):
            for member, state in zip(("a", "b"), states):
                rows.append({
                    "system_id": "s", "replica_id": "r", "segment_id": "seg",
                    "source_frame_index": frame, "member_id": member,
                    "cluster_id": state,
                })
        report = summarize_state_populations(rows, "cluster_id")
        pair = report["paired_member_state_coupling"]["pair_reports"][0]
        self.assertEqual(pair["paired_physical_frame_count"], 4)
        self.assertEqual(pair["same_state_fraction"], 0.75)
        self.assertEqual(len(report["member_populations"]), 2)


if __name__ == "__main__":
    unittest.main()
