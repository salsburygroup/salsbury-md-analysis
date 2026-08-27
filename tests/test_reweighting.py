import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.reweighting import (
    ReweightingError,
    normalize_log_weights,
    trajectory_reweighting_project,
    trajectory_reweighting_project_safe,
    weighted_moments,
)


class ReweightingTests(unittest.TestCase):
    def test_uniform_log_weights_have_full_effective_sample_size(self):
        report = normalize_log_weights(
            [1000.0, 1000.0, 1000.0, 1000.0],
            minimum_kish_effective_sample_size=4.0,
            minimum_kish_ratio=1.0,
            maximum_single_frame_weight=0.25,
        )
        np.testing.assert_allclose(report["normalized_weights"], [0.25] * 4)
        self.assertAlmostEqual(report["kish_effective_sample_size"], 4.0)
        self.assertAlmostEqual(report["entropy_effective_sample_size"], 4.0)
        self.assertEqual(report["reweighting_validity_status"], "passed")

    def test_concentrated_weights_fail_declared_reliability_gates(self):
        report = normalize_log_weights(
            [0.0, -20.0, -20.0, -20.0],
            minimum_kish_effective_sample_size=2.0,
            minimum_kish_ratio=0.5,
            maximum_single_frame_weight=0.8,
        )
        self.assertEqual(report["reweighting_validity_status"], "failed")
        self.assertGreater(report["maximum_single_frame_weight"], 0.99)
        self.assertFalse(report["gate_results"]["minimum_kish_ratio"]["passed"])

    def test_weighted_moments_use_normalized_population_weights(self):
        report = weighted_moments([[0.0], [10.0]], [0.75, 0.25])
        self.assertAlmostEqual(report["weighted_mean"][0], 2.5)
        self.assertAlmostEqual(report["weighted_population_covariance"][0][0], 18.75)
        with self.assertRaises(ReweightingError):
            weighted_moments([[0.0], [10.0]], [1.0, 1.0])

    def test_project_aligns_weights_by_full_frame_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = self._write_project(root)
            with patch(
                "salsbury_md_analysis.reweighting.common_pca_project",
                return_value=self._pca_report(),
            ):
                report = trajectory_reweighting_project(project_path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["reweighting_validity_status"], "passed")
        self.assertEqual(
            [row["source_frame_index"] for row in report["systems"][0]["frame_weights"]],
            [0, 2, 4],
        )
        self.assertAlmostEqual(
            sum(row["normalized_weight"] for row in report["systems"][0]["frame_weights"]),
            1.0,
        )
        self.assertEqual(
            report["observation_accounting"],
            {
                "selected_physical_frame_count": 3,
                "symmetry_expanded_observation_count": 3,
                "accounting_basis": "exact matched common-PCA projection and weight identities",
            },
        )

    def test_missing_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = self._write_project(root)
            payload = json.loads((root / "weights.json").read_text(encoding="utf-8"))
            payload["rows"].pop()
            (root / "weights.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.reweighting.common_pca_project",
                return_value=self._pca_report(),
            ):
                report = trajectory_reweighting_project_safe(project_path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("missing_weights=1", report["issues"][0]["message"])

    @staticmethod
    def _write_project(root):
        rows = [
            {
                "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": frame, "log_weight": log_weight,
            }
            for frame, log_weight in ((4, -0.2), (0, 0.0), (2, -0.1))
        ]
        (root / "weights.json").write_text(json.dumps({
            "weight_schema": "salsbury-frame-log-weights-v1",
            "weight_semantics": "log_unnormalized_target_over_source_probability",
            "rows": rows,
        }), encoding="utf-8")
        project = {
            "definitions": {
                "trajectory_reweighting": {
                    "observable_source": "common_pca",
                    "weights_path": "weights.json",
                    "normalization_scope": "per_system",
                    "minimum_kish_effective_sample_size": 2.0,
                    "minimum_kish_ratio": 0.5,
                    "maximum_single_frame_weight": 0.5,
                }
            }
        }
        path = root / "project.json"
        path.write_text(json.dumps(project), encoding="utf-8")
        return path

    @staticmethod
    def _pca_report():
        return {
            "technical_status": "complete",
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "projection_frame_selection": {"selected_frame_count": 3},
            "systems": [{
                "system_id": "sys",
                "replicas": [{
                    "replica_id": "r1",
                    "segments": [{
                        "segment_id": "s1",
                        "projections": [
                            {"source_frame_index": 0, "scores_angstrom": [0.0, 1.0]},
                            {"source_frame_index": 2, "scores_angstrom": [1.0, 2.0]},
                            {"source_frame_index": 4, "scores_angstrom": [2.0, 4.0]},
                        ],
                    }],
                }],
            }],
        }


if __name__ == "__main__":
    unittest.main()
