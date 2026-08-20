import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.grouped_ml import (
    fit_decision_tree,
    grouped_ml_project,
    predict_tree,
)


class GroupedMLTests(unittest.TestCase):
    def test_decision_tree_is_deterministic_and_separates_classes(self):
        vectors = [(-2.0, 0.0), (-1.0, 0.1), (1.0, 0.0), (2.0, -0.1)]
        labels = [1, 1, 2, 2]
        first = fit_decision_tree(vectors, labels, 3, 1, 100)
        second = fit_decision_tree(vectors, labels, 3, 1, 100)
        self.assertEqual(first, second)
        tree, importances = first
        self.assertEqual([predict_tree(tree, vector) for vector in vectors], labels)
        self.assertAlmostEqual(sum(importances), 1.0)
        self.assertGreater(importances[0], importances[1])

    def test_project_uses_unit_independent_kmeans_feature_values(self):
        rows = []
        for replica_index in range(4):
            for frame_index, (value, label) in enumerate(((-1.0, 1), (1.0, 2))):
                rows.append({
                    "system_id": "sys",
                    "replica_id": f"r{replica_index + 1}",
                    "segment_id": "production",
                    "source_frame_index": frame_index,
                    "feature_values": [value, value / 2.0],
                    "cluster_id": label,
                })
        clustering = {
            "assignments": rows,
            "feature_contract": {"source": "tica", "feature_count": 2},
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
        }
        project = {"definitions": {"grouped_ml": {
            "feature_source": "clustering_kmeans_features",
            "target_source": "clustering_kmeans_assignments",
            "group_strategy": "segment_time_blocks",
            "group_block_size_frames": 10,
            "estimator": "decision_tree",
            "maximum_depth": 3,
            "minimum_leaf_size": 1,
            "maximum_thresholds_per_feature": 10,
            "permutation_repeats": 2,
            "random_seed": 7,
            "minimum_groups": 2,
            "maximum_observations": 100,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.grouped_ml.clustering_kmeans_project",
                return_value=clustering,
            ):
                report = grouped_ml_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["feature_contract"]["source"], "tica")
        self.assertEqual(report["observation_count"], len(rows))


if __name__ == "__main__":
    unittest.main()
