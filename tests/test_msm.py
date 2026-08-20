import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.msm import (
    _clustering_candidates,
    cross_validated_vamp_report,
    estimate_transition_model,
    markov_state_models_project,
)


class MSMTests(unittest.TestCase):
    @staticmethod
    def _rows(state_key="cluster_id"):
        rows = []
        states = ([1] * 5 + [2] * 5) * 4
        for replica in ("r1", "r2"):
            for index, state in enumerate(states):
                rows.append({
                    "system_id": "sys", "replica_id": replica,
                    "segment_id": "seg1", "source_frame_index": index,
                    "time": index * 2.0, "time_unit": "ps",
                    state_key: state,
                })
        return rows

    def test_reversible_model_is_normalized_and_segment_safe(self):
        trajectories = [[0, 0, 1, 1], [1, 0]]
        model = estimate_transition_model(
            trajectories, state_count=2, lag_frames=1,
            estimator="reversible_symmetrized", interval=10.0, time_unit="ps",
        )
        self.assertEqual(model["transition_count"], 4)
        self.assertEqual(model["count_matrix"], [[1, 1], [1, 1]])
        self.assertTrue(model["connected"])
        for row in model["transition_matrix"]:
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertAlmostEqual(sum(model["stationary_distribution"]), 1.0)

    def test_project_reports_unpassed_validation_on_short_fixture(self):
        rows = []
        for index, state in enumerate([1, 1, 2, 2, 1, 2, 2, 1]):
            rows.append({
                "system_id": "sys", "replica_id": "r1", "segment_id": "seg1",
                "source_frame_index": index, "time": (index + 1) * 10.0,
                "time_unit": "ps", "cluster_id": state,
            })
        clustering = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "selected_model": {"k": 2},
            "assignments": rows,
            "issues": [],
        }
        project = {"definitions": {"markov_state_models": {
            "assignment_source": "clustering_kmeans",
            "lag_frames": [1, 2],
            "estimators": ["reversible_symmetrized", "nonreversible_mle"],
            "minimum_transition_count": 100,
            "maximum_states": 10,
            "ck_multiples": [2],
            "maximum_ck_rmse": 0.5,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch("salsbury_md_analysis.msm.clustering_kmeans_project", return_value=clustering):
                report = markov_state_models_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        self.assertEqual(report["kinetic_validation_status"], "not passed")
        self.assertEqual(len(report["models"]), 4)
        self.assertEqual(len(report["chapman_kolmogorov_tests"]), 2)

    def test_time_blocked_vamp_scores_are_finite(self):
        trajectories = [[0] * 5 + [1] * 5] * 4
        report = cross_validated_vamp_report(
            trajectories, state_count=2, lag_frames=1,
            fold_count=2, regularization=1.0e-8,
        )
        self.assertEqual(report["status"], "complete")
        self.assertTrue(math.isfinite(report["mean_training_vamp2"]))
        self.assertTrue(math.isfinite(report["mean_heldout_vamp_e"]))
        self.assertIn("without leaving out an entire replica", report["fold_assignment"])

    def test_multi_source_report_keeps_best_clustering_and_fes_separate(self):
        cluster_rows = self._rows("cluster_id")
        fes_rows = self._rows("basin_id")
        clustering = {
            "module_id": "clustering_kmeans",
            "selected_model": {"k": 2, "silhouette": 0.7},
            "assignments": cluster_rows,
        }
        fes = {
            "module_id": "pca_fes_basins",
            "technical_status": "complete",
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "basin_silhouette": {"score": 0.5},
            "frame_assignments": fes_rows,
        }
        project = {
            "definitions": {
                "clustering_kmeans": {},
                "pca_fes_basins": {},
                "markov_state_models": {
                    "assignment_sources": ["best_clustering", "pca_fes_basins"],
                    "lag_frames": [1, 2, 4],
                    "estimators": ["reversible_symmetrized", "nonreversible_mle"],
                    "minimum_transition_count": 1,
                    "maximum_states": 10,
                    "ck_multiples": [2],
                    "maximum_ck_rmse": 1.0,
                    "vamp_cross_validation_folds": 2,
                    "vamp_regularization": 1.0e-8,
                    "maximum_implied_timescale_relative_range": 10.0,
                    "bootstrap_repeats": 4,
                    "bootstrap_block_length_frames": 8,
                    "bootstrap_confidence_level": 0.9,
                    "random_seed": 7,
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.msm.clustering_kmeans_project",
                return_value=clustering,
            ), patch(
                "salsbury_md_analysis.msm.pca_fes_basins_project",
                return_value=fes,
            ):
                report = markov_state_models_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(
            report["best_clustering_state_model"]["candidate_id"], "kmeans"
        )
        self.assertEqual(report["fes_state_model"]["candidate_id"], "pca_fes_basins")
        self.assertIn("best_clustering_state_model_details", report)
        self.assertIn("fes_state_model_details", report)
        self.assertEqual(report["observation_count"], len(fes_rows))
        self.assertEqual(
            report["observation_accounting"],
            {
                "source_physical_frame_count": len(fes_rows),
                "symmetry_expanded_observation_count": len(fes_rows),
                "member_observations_are_independent_replicas": None,
            },
        )

    def test_overfragmented_clustering_is_reported_but_skipped_for_msm(self):
        cluster_rows = self._rows("cluster_id")
        affinity_rows = []
        for index, row in enumerate(cluster_rows):
            affinity_rows.append({**row, "cluster_id": index + 1})
        kmeans = {
            "module_id": "clustering_kmeans",
            "selected_model": {"k": 2, "silhouette": 0.6},
            "assignments": cluster_rows,
        }
        alternative = {
            "module_id": "alternative_clustering",
            "observation_count": len(affinity_rows),
            "algorithm_results": [{
                "requested_algorithm": "affinity_propagation",
                "parameters": {},
                "silhouette": 0.9,
                "retained_fraction": 1.0,
                "full_retained_fraction": 1.0,
                "fit_observation_count": len(affinity_rows),
                "frame_assignments": affinity_rows,
                "full_partition_silhouette_evaluation": {"score": 0.9},
                "assignment_extension": {"scope": "all source observations"},
            }],
        }
        fes = {
            "module_id": "pca_fes_basins",
            "technical_status": "complete",
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "basin_silhouette": {"score": 0.5},
            "frame_assignments": self._rows("basin_id"),
        }
        project = {"definitions": {
            "clustering_kmeans": {},
            "alternative_clustering": {},
            "pca_fes_basins": {},
            "markov_state_models": {
                "assignment_sources": ["best_clustering", "pca_fes_basins"],
                "lag_frames": [1, 2],
                "estimators": ["reversible_symmetrized"],
                "minimum_transition_count": 1,
                "maximum_states": 10,
                "ck_multiples": [2],
                "maximum_ck_rmse": 1.0,
                "vamp_cross_validation_folds": 2,
                "vamp_regularization": 1.0e-8,
                "maximum_implied_timescale_relative_range": 10.0,
                "bootstrap_repeats": 4,
                "bootstrap_block_length_frames": 8,
                "bootstrap_confidence_level": 0.9,
                "random_seed": 7,
            },
        }}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.msm.clustering_kmeans_project",
                return_value=kmeans,
            ), patch(
                "salsbury_md_analysis.msm.alternative_clustering_project",
                return_value=alternative,
            ), patch(
                "salsbury_md_analysis.msm.pca_fes_basins_project",
                return_value=fes,
            ):
                report = markov_state_models_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(
            report["best_geometric_clustering"]["candidate_id"],
            "affinity_propagation",
        )
        self.assertEqual(
            report["best_clustering_state_model"]["candidate_id"], "kmeans"
        )
        inventory = {
            row["candidate_id"]: row
            for row in report["clustering_method_inventory"]
        }
        self.assertFalse(inventory["affinity_propagation"]["msm_eligible"])
        self.assertEqual(inventory["affinity_propagation"]["msm_role"], "skipped")
        self.assertIn(
            "MSM_STATE_COUNT_EXCEEDS_LIMIT",
            {issue["code"] for issue in report["issues"]},
        )

    def test_hdbscan_core_msm_is_noise_censored_sensitivity_not_primary(self):
        cluster_rows = self._rows("cluster_id")
        hdbscan_rows = []
        for row in cluster_rows:
            copied = dict(row)
            if int(copied["source_frame_index"]) % 5 == 4:
                copied["cluster_id"] = None
                copied["is_noise"] = True
            else:
                copied["is_noise"] = False
            hdbscan_rows.append(copied)
        kmeans = {
            "module_id": "clustering_kmeans",
            "selected_model": {"k": 2, "silhouette": 0.7},
            "assignments": cluster_rows,
        }
        hdbscan = {
            "module_id": "clustering_hdbscan",
            "selected_model": {
                "cluster_count": 2,
                "retained_fraction": 0.8,
                "retained_only_silhouette": 0.95,
            },
            "assignments": hdbscan_rows,
        }
        fes = {
            "module_id": "pca_fes_basins",
            "technical_status": "complete",
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "basin_silhouette": {"score": 0.5},
            "frame_assignments": self._rows("basin_id"),
        }
        project = {"definitions": {
            "clustering_kmeans": {},
            "clustering_hdbscan": {},
            "pca_fes_basins": {},
            "markov_state_models": {
                "assignment_sources": ["best_clustering", "pca_fes_basins"],
                "lag_frames": [1, 2],
                "estimators": ["reversible_symmetrized"],
                "minimum_transition_count": 1,
                "maximum_states": 10,
                "ck_multiples": [2],
                "maximum_ck_rmse": 1.0,
                "vamp_cross_validation_folds": 2,
                "vamp_regularization": 1.0e-8,
                "maximum_implied_timescale_relative_range": 10.0,
                "bootstrap_repeats": 4,
                "bootstrap_block_length_frames": 8,
                "bootstrap_confidence_level": 0.9,
                "random_seed": 7,
            },
        }}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.msm.clustering_kmeans_project",
                return_value=kmeans,
            ), patch(
                "salsbury_md_analysis.msm.clustering_hdbscan_project",
                return_value=hdbscan,
            ), patch(
                "salsbury_md_analysis.msm.pca_fes_basins_project",
                return_value=fes,
            ):
                report = markov_state_models_project(path)
        self.assertEqual(
            report["best_geometric_clustering"]["candidate_id"], "hdbscan"
        )
        self.assertEqual(
            report["best_clustering_state_model"]["candidate_id"], "kmeans"
        )
        sensitivity = report[
            "sampled_clustering_state_model_sensitivities"
        ]
        self.assertEqual([row["candidate_id"] for row in sensitivity], ["hdbscan"])
        self.assertEqual(sensitivity[0]["msm_role"], "sampled_sensitivity")
        inventory = {
            row["candidate_id"]: row
            for row in report["clustering_method_inventory"]
        }
        self.assertTrue(inventory["hdbscan"]["msm_eligible"])
        self.assertFalse(
            inventory["hdbscan"]["primary_msm_selection_eligible"]
        )
        diagnostics = inventory["hdbscan"]["msm_assignment_diagnostics"]
        self.assertFalse(diagnostics["noise_gaps_crossed"])
        self.assertGreater(diagnostics["noise_observation_count"], 0)
        self.assertLess(inventory["hdbscan"]["msm_coverage_fraction"], 1.0)

    def test_full_observation_ward_partition_is_primary_eligible(self):
        rows = self._rows("cluster_id")
        alternative = {
            "module_id": "alternative_clustering",
            "observation_count": len(rows),
            "algorithm_results": [{
                "requested_algorithm": "ward",
                "parameters": {"k": 2},
                "silhouette": 0.7,
                "retained_fraction": 1.0,
                "fit_observation_count": len(rows),
                "fit_frame_assignments": rows,
                "frame_assignments": rows,
                "assignment_extension": {
                    "scope": "all source observations",
                },
            }],
        }
        project = {"definitions": {"alternative_clustering": {}}}
        with patch(
            "salsbury_md_analysis.msm.alternative_clustering_project",
            return_value=alternative,
        ):
            candidates, _ = _clustering_candidates(
                Path("/tmp/project.json"), project, False
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_id"], "ward")
        self.assertTrue(candidates[0]["primary_msm_selection_eligible"])
        self.assertEqual(candidates[0]["msm_role"], "primary_candidate")
        self.assertEqual(
            candidates[0]["msm_assignment_scope"],
            "complete_exact_fitted_partition",
        )


if __name__ == "__main__":
    unittest.main()
