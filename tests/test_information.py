import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.information import (
    generalized_correlation_and_information_project,
    mutual_information_matrices,
)
from tests.test_tica import _write_project


class InformationTests(unittest.TestCase):
    def test_nonlinear_dependence_is_detected_when_pearson_is_near_zero(self):
        x = np.linspace(-1.0, 1.0, 2001)
        y = x * x
        self.assertLess(abs(float(np.corrcoef(x, y)[0, 1])), 1.0e-12)
        report = mutual_information_matrices(
            np.column_stack((x, y)).tolist(),
            bin_count=10,
            minimum_observations=100,
        )
        self.assertGreater(report["mutual_information_nats"][0][1], 0.5)
        self.assertGreater(report["generalized_correlation"][0][1], 0.7)

    def test_independence_is_lower_than_exact_dependence(self):
        generator = np.random.default_rng(42)
        x = generator.normal(size=5000)
        independent = generator.normal(size=5000)
        exact = mutual_information_matrices(
            np.column_stack((x, x)).tolist(), bin_count=12, minimum_observations=100
        )
        null = mutual_information_matrices(
            np.column_stack((x, independent)).tolist(),
            bin_count=12,
            minimum_observations=100,
        )
        self.assertGreater(
            exact["normalized_mutual_information"][0][1],
            null["normalized_mutual_information"][0][1],
        )

    def test_constant_feature_is_null_not_zero(self):
        report = mutual_information_matrices(
            [[float(index), 1.0] for index in range(100)],
            bin_count=5,
            minimum_observations=20,
        )
        self.assertIsNone(report["normalized_mutual_information"][0][1])
        self.assertIsNone(report["generalized_correlation"][0][1])

    def test_cli_reports_common_pca_feature_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["generalized_correlation_and_information"] = {
                "feature_source": "common_pca",
                "component_indices": [1, 2],
                "bin_count": 4,
                "minimum_observations_per_replica": 20,
                "maximum_features": 10,
            }
            project["requested_modules"].append("generalized_correlation_and_information")
            path.write_text(json.dumps(project), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["information-correlation", str(path)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0, report)
        self.assertEqual(report["module_id"], "generalized_correlation_and_information")
        self.assertEqual(report["feature_lineage"]["module_id"], "common_pca")
        self.assertEqual(report["replicas"][0]["observation_count"], 24)

    def test_short_replica_is_skipped_but_retained_in_pooled_system_estimate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps({
                "definitions": {"generalized_correlation_and_information": {
                    "feature_source": "common_pca",
                    "component_indices": [1, 2],
                    "bin_count": 4,
                    "minimum_observations_per_replica": 20,
                    "maximum_features": 10,
                }},
            }), encoding="utf-8")

            def replica(replica_id, count):
                return {
                    "replica_id": replica_id,
                    "segments": [{"projections": [
                        {"scores_angstrom": [float(index), float(index % 3)]}
                        for index in range(count)
                    ]}],
                }

            pca_report = {
                "technical_status": "complete",
                "project_manifest_sha256": "project-hash",
                "system_manifest_path": "/system.json",
                "system_manifest_sha256": "system-hash",
                "contract_signature_sha256": "contract-hash",
                "input_content_signature_sha256": "input-hash",
                "settings": {"component_count": 2},
                "systems": [{
                    "system_id": "conditional-system",
                    "replicas": [replica("short", 10), replica("long", 30)],
                }],
                "issues": [],
            }
            with patch(
                "salsbury_md_analysis.information.common_pca_project",
                return_value=pca_report,
            ):
                report = generalized_correlation_and_information_project(path)

        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["replicas"][0]["technical_status"], "skipped")
        self.assertEqual(report["replicas"][0]["observation_count"], 10)
        self.assertEqual(report["replicas"][1]["technical_status"], "complete")
        self.assertEqual(report["systems"][0]["technical_status"], "complete")
        self.assertEqual(report["systems"][0]["observation_count"], 40)
        self.assertTrue(any(
            issue["code"] == "REPLICA_INFORMATION_ESTIMATE_SKIPPED"
            for issue in report["issues"]
        ))


if __name__ == "__main__":
    unittest.main()
