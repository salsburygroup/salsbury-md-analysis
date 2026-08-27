import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.perturbation_response import (
    PerturbationResponseError,
    perturbation_response_dynamics_project,
    perturbation_response_indices,
)


class PerturbationResponseTests(unittest.TestCase):
    def test_isotropic_independent_nodes_have_uniform_dfi_and_unit_dci(self):
        report = perturbation_response_indices(
            np.eye(9).tolist(),
            [0, 2],
            random_force_directions=32,
            random_seed=17,
            maximum_nodes=3,
        )
        self.assertEqual(report["matrix_orientation"], "row target node; column perturbed source node")
        np.testing.assert_allclose(report["dfi"], [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(report["dci"], [1.5, 0.0, 1.5])
        self.assertAlmostEqual(sum(report["dfi"]), 1.0)

    def test_functional_site_coupling_is_target_specific(self):
        covariance = np.eye(9)
        covariance[3:6, 0:3] = 0.8 * np.eye(3)
        covariance[0:3, 3:6] = 0.8 * np.eye(3)
        report = perturbation_response_indices(
            covariance.tolist(),
            [0],
            random_force_directions=64,
            random_seed=9,
            maximum_nodes=3,
        )
        self.assertGreater(report["dci"][1], report["dci"][2])
        self.assertGreater(report["dfi"][1], report["dfi"][2])

    def test_contract_rejects_invalid_functional_site(self):
        with self.assertRaises(PerturbationResponseError):
            perturbation_response_indices(np.eye(6).tolist(), [2])

    def test_project_uses_per_system_score_covariance_in_common_basis(self):
        pca_report = {
            "technical_status": "complete",
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "reference_system_id": "reference",
            "projection_frame_selection": {"selected_frame_count": 6},
            "basis": {
                "pca": {
                    "mean_structure": [{"analysis_index": 0}, {"analysis_index": 1}],
                    "components": [
                    {
                        "cumulative_explained_variance_fraction": 0.8,
                        "loadings": [
                            {"loading_x": 1.0, "loading_y": 0.0, "loading_z": 0.0},
                            {"loading_x": 0.0, "loading_y": 0.0, "loading_z": 0.0},
                        ],
                    },
                    {
                        "cumulative_explained_variance_fraction": 0.95,
                        "loadings": [
                            {"loading_x": 0.0, "loading_y": 0.0, "loading_z": 0.0},
                            {"loading_x": 1.0, "loading_y": 0.0, "loading_z": 0.0},
                        ],
                    },
                    ],
                },
            },
            "systems": [
                self._system("reference", [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
                self._system("variant", [[-2.0, 1.0], [0.0, 0.0], [2.0, -1.0]]),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps({
                "definitions": {
                    "perturbation_response_dynamics": {
                        "feature_source": "common_pca",
                        "functional_site_node_indices": [0],
                        "random_force_directions": 32,
                        "random_seed": 4,
                        "maximum_nodes": 10,
                        "minimum_observations_per_system": 3,
                        "minimum_cumulative_explained_variance": 0.9,
                        "include_self_perturbations": True,
                    }
                }
            }), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.perturbation_response.common_pca_project",
                return_value=pca_report,
            ):
                report = perturbation_response_dynamics_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["common_pca_component_count"], 2)
        self.assertEqual(
            report["observation_accounting"]["selected_physical_frame_count"], 6
        )
        self.assertEqual([row["observation_count"] for row in report["systems"]], [3, 3])
        variant = report["systems"][1]
        self.assertTrue(any(abs(value) > 1.0e-6 for value in variant["difference_from_reference"]["dfi"]))

    @staticmethod
    def _system(system_id, scores):
        return {
            "system_id": system_id,
            "replicas": [{
                "segments": [{
                    "projections": [
                        {"scores_angstrom": row} for row in scores
                    ]
                }]
            }],
        }


if __name__ == "__main__":
    unittest.main()
