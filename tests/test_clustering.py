import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.clustering import (
    adjusted_rand_index,
    clustering_hdbscan_project_safe,
    clustering_imwkmeans_project,
    clustering_kmeans_project,
    nani_complementary_msd,
    run_imwkmeans,
    run_kmeans,
    silhouette_score,
    silhouette_score_report,
    silhouette_sampling_stability_report,
)


class ClusteringTests(unittest.TestCase):
    def test_nani_complementary_msd_matches_published_reference_example(self):
        vectors = [(1.0, 2.0), (2.0, 2.0), (2.0, 3.0), (8.0, 7.0), (8.0, 8.0)]
        self.assertEqual(
            nani_complementary_msd(vectors),
            [31.0, 34.375, 36.75, 27.75, 23.875],
        )

    def test_stratified_nani_kmeans_is_deterministic_without_random_seed(self):
        vectors = [
            (-2.0 + index * 0.01, -2.0 - index * 0.01)
            for index in range(20)
        ] + [
            (2.0 + index * 0.01, 2.0 - index * 0.01)
            for index in range(20)
        ]
        for method in ("nani_strat_all", "nani_strat_reduced"):
            first = run_kmeans(
                vectors, 2, None, 100, 1.0e-10,
                initialization_method=method, nani_percentage=25,
            )
            second = run_kmeans(
                vectors, 2, None, 100, 1.0e-10,
                initialization_method=method, nani_percentage=25,
            )
            self.assertTrue(first["valid"])
            self.assertEqual(first, second)
            self.assertIsNone(first["seed"])
            self.assertFalse(first["initialization"]["random_seed_used"])
            self.assertEqual(len(first["initialization"]["initial_center_indices"]), 2)

    def test_pooled_clustering_assigns_one_model_across_replicas(self):
        replica_points = {
            "r1": [(-2.0, -2.0), (-1.9, -2.1), (2.0, 2.0)],
            "r2": [(-2.1, -1.9), (1.9, 2.1), (2.1, 1.9)],
        }
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{
                "system_id": "pooled",
                "replicas": [
                    {
                        "replica_id": replica_id,
                        "segments": [{
                            "segment_id": "samples",
                            "projections": [
                                {
                                    "source_frame_index": index,
                                    "sample_index": index,
                                    "scores_angstrom": [x, y],
                                }
                                for index, (x, y) in enumerate(points)
                            ],
                        }],
                    }
                    for replica_id, points in replica_points.items()
                ],
            }],
        }
        project = {"definitions": {"clustering_kmeans": {
            "feature_source": "common_pca", "component_indices": [1, 2],
            "standardize_features": True, "k_values": [2],
            "random_seeds": [3, 7], "maximum_iterations": 100,
            "center_tolerance": 1.0e-10, "minimum_cluster_size": 2,
            "maximum_silhouette_observations": 100,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                report = clustering_kmeans_project(path)
        members_by_cluster = {}
        for row in report["assignments"]:
            members_by_cluster.setdefault(row["cluster_id"], set()).add(
                row["replica_id"]
            )
        self.assertEqual(len(report["assignments"]), 6)
        self.assertEqual(set(members_by_cluster), {1, 2})
        self.assertTrue(all(value == {"r1", "r2"} for value in members_by_cluster.values()))

    def test_hdbscan_dependency_absence_fails_closed(self):
        project = {"definitions": {"clustering_hdbscan": {
            "feature_source": "common_pca", "component_indices": [1, 2],
            "standardize_features": True, "minimum_cluster_sizes": [2],
            "minimum_samples_values": [1], "cluster_selection_method": "eom",
            "allow_single_cluster": False, "minimum_retained_fraction": 0.5,
            "maximum_silhouette_observations": 100,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch("salsbury_md_analysis.clustering.importlib.import_module", side_effect=ImportError("missing")):
                report = clustering_hdbscan_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("optional dependency hdbscan", report["issues"][0]["message"])

    def test_imwkmeans_is_deterministic_complete_and_weighted(self):
        vectors = [(-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (2.0, 2.0), (2.1, 1.9), (1.9, 2.1)]
        first = run_imwkmeans(vectors, 2, 2.0, 0, 100, 1.0e-10, 1.0e-12)
        second = run_imwkmeans(vectors, 2, 2.0, 0, 100, 1.0e-10, 1.0e-12)
        self.assertTrue(first["valid"])
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_sizes"], [3, 3])
        for weights in first["feature_weights"]:
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_kmeans_is_seeded_complete_and_label_canonical(self):
        vectors = [(-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (2.0, 2.0), (2.1, 1.9), (1.9, 2.1)]
        first = run_kmeans(vectors, 2, 7, 100, 1.0e-10)
        second = run_kmeans(vectors, 2, 7, 100, 1.0e-10)
        self.assertTrue(first["valid"])
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_sizes"], [3, 3])
        self.assertGreater(silhouette_score(vectors, first["assignments"]), 0.9)
        self.assertEqual(
            adjusted_rand_index([0, 0, 1, 1], [5, 5, 2, 2]),
            1.0,
        )

    def test_silhouette_estimate_is_seeded_and_uses_full_partition_neighbors(self):
        vectors = [
            (float(index), 0.0) for index in range(20)
        ] + [
            (float(index), 20.0) for index in range(20)
        ]
        labels = [0] * 20 + [1] * 20
        first = silhouette_score_report(vectors, labels, 9, random_seed=17)
        second = silhouette_score_report(vectors, labels, 9, random_seed=17)
        self.assertEqual(first, second)
        self.assertTrue(first["estimated"])
        self.assertEqual(first["evaluated_observation_count"], 9)
        self.assertEqual(first["total_observation_count"], 40)
        self.assertIn("against_full_partition", first["method"])

    def test_sampled_silhouette_records_several_prespecified_seeds(self):
        vectors = [
            (float(index), 0.0) for index in range(20)
        ] + [
            (float(index), 20.0) for index in range(20)
        ]
        labels = [0] * 20 + [1] * 20
        report = silhouette_sampling_stability_report(
            vectors, labels, 9, [0, 7, 19, 41]
        )
        self.assertTrue(report["estimated"])
        self.assertEqual(report["evaluated_random_seeds"], [0, 7, 19, 41])
        self.assertEqual(report["sampling_replicate_count"], 4)
        self.assertEqual(len(report["replicates"]), 4)
        self.assertAlmostEqual(report["score"], report["score_mean"])

    def test_project_grid_selects_complete_two_cluster_partition(self):
        points = [(-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (2.0, 2.0), (2.1, 1.9), (1.9, 2.1)]
        projections = [
            {
                "source_frame_index": index,
                "sample_index": index,
                "scores_angstrom": [x, y],
            }
            for index, (x, y) in enumerate(points)
        ]
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{
                "system_id": "ai",
                "replicas": [{
                    "replica_id": "r1",
                    "segments": [{"segment_id": "samples", "projections": projections}],
                }],
            }],
        }
        project = {
            "definitions": {
                "clustering_kmeans": {
                    "feature_source": "common_pca",
                    "component_indices": [1, 2],
                    "standardize_features": True,
                    "k_values": [2, 3],
                    "random_seeds": [3, 7, 11],
                    "maximum_iterations": 100,
                    "center_tolerance": 1.0e-10,
                    "minimum_cluster_size": 2,
                    "maximum_silhouette_observations": 100,
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                report = clustering_kmeans_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["selected_model"]["k"], 2)
        self.assertEqual(report["selected_model"]["cluster_sizes"], [3, 3])
        self.assertEqual(len(report["assignments"]), 6)
        self.assertTrue(all(row["cluster_id"] in {1, 2} for row in report["assignments"]))
        self.assertTrue(
            any(
                "not a technical failure" in limitation
                and "discarding assignments" in limitation
                for limitation in report["limitations"]
            )
        )

    def test_project_grid_scans_both_stratified_nani_initializers(self):
        points = [
            (-2.0 + index * 0.01, -2.0 - index * 0.01)
            for index in range(20)
        ] + [
            (2.0 + index * 0.01, 2.0 - index * 0.01)
            for index in range(20)
        ]
        projections = [
            {
                "source_frame_index": index,
                "sample_index": index,
                "scores_angstrom": [x, y],
            }
            for index, (x, y) in enumerate(points)
        ]
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{
                "system_id": "nani",
                "replicas": [{
                    "replica_id": "r1",
                    "segments": [{"segment_id": "samples", "projections": projections}],
                }],
            }],
        }
        project = {"definitions": {"clustering_kmeans": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "k_values": [2],
            "initialization_methods": ["nani_strat_all", "nani_strat_reduced"],
            "nani_percentage": 25,
            "silhouette_random_seeds": [0, 7, 19, 41],
            "maximum_iterations": 100,
            "center_tolerance": 1.0e-10,
            "minimum_cluster_size": 2,
            "maximum_silhouette_observations": 10,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                first = clustering_kmeans_project(path)
                second = clustering_kmeans_project(path)
        self.assertEqual(first, second)
        self.assertIsNone(first["selected_model"]["seed"])
        self.assertIn(
            first["selected_model"]["initialization_method"],
            {"nani_strat_all", "nani_strat_reduced"},
        )
        diagnostic = first["grid_diagnostics"][0]
        self.assertEqual(diagnostic["valid_initialization_count"], 2)
        self.assertEqual(diagnostic["valid_seed_count"], 0)
        self.assertEqual(
            {row["initialization_method"] for row in diagnostic["runs"]},
            {"nani_strat_all", "nani_strat_reduced"},
        )
        gate = first["silhouette_selection_stability"]
        self.assertTrue(gate["gate_applied"])
        self.assertEqual(gate["status"], "passed_unanimous_sampled_winner")
        self.assertEqual(gate["configured_random_seeds"], [0, 7, 19, 41])

    def test_sampled_silhouette_gate_fails_when_seeds_select_different_k(self):
        points = [
            (-3.0 + index * 0.01, -3.0) for index in range(20)
        ] + [
            (0.0 + index * 0.01, 0.0) for index in range(20)
        ] + [
            (3.0 + index * 0.01, 3.0) for index in range(20)
        ]
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{"system_id": "nani", "replicas": [{
                "replica_id": "r1",
                "segments": [{"segment_id": "samples", "projections": [
                    {
                        "source_frame_index": index,
                        "sample_index": index,
                        "scores_angstrom": [x, y],
                    }
                    for index, (x, y) in enumerate(points)
                ]}],
            }]}],
        }
        project = {"definitions": {"clustering_kmeans": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "k_values": [2, 3],
            "initialization_methods": ["nani_strat_all"],
            "nani_percentage": 50,
            "silhouette_random_seeds": [0, 7, 19],
            "maximum_iterations": 100,
            "center_tolerance": 1.0e-10,
            "minimum_cluster_size": 2,
            "maximum_silhouette_observations": 10,
        }}}

        def unstable_report(_vectors, assignments, _maximum, seeds):
            scores = (
                [0.9, 0.1, 0.9]
                if len(set(assignments)) == 2 else [0.1, 0.9, 0.1]
            )
            return {
                "score": sum(scores) / len(scores),
                "estimated": True,
                "replicates": [
                    {"score": score, "random_seed": seed}
                    for score, seed in zip(scores, seeds)
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ), patch(
                "salsbury_md_analysis.clustering.silhouette_sampling_stability_report",
                side_effect=unstable_report,
            ):
                with self.assertRaisesRegex(
                    ValueError, "sampled silhouette winner stability gate failed"
                ):
                    clustering_kmeans_project(path)

    def test_imwkmeans_project_grid_reports_algorithm_contract(self):
        points = [(-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1), (2.0, 2.0), (2.1, 1.9), (1.9, 2.1)]
        projections = [
            {"source_frame_index": index, "sample_index": index, "scores_angstrom": [x, y]}
            for index, (x, y) in enumerate(points)
        ]
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{"system_id": "ai", "replicas": [{
                "replica_id": "r1",
                "segments": [{"segment_id": "samples", "projections": projections}],
            }]}],
        }
        project = {"definitions": {"clustering_imwkmeans": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "k_values": [2, 3],
            "minkowski_p_values": [1.5, 2.0],
            "initialization_ranks": [0, 1, 2],
            "maximum_iterations": 100,
            "objective_tolerance": 1.0e-10,
            "minimum_cluster_size": 2,
            "weight_dispersion_floor": 1.0e-12,
            "maximum_silhouette_observations": 100,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch("salsbury_md_analysis.feature_matrix.common_pca_project", return_value=fake_pca):
                report = clustering_imwkmeans_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["selected_model"]["k"], 2)
        self.assertEqual(report["selected_model"]["cluster_sizes"], [3, 3])
        self.assertIn("feature_weight", report["algorithm_contract"])
        self.assertEqual(len(report["assignments"]), 6)

    def test_imwkmeans_grid_checkpoints_resume_without_refitting(self):
        points = [
            (-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1),
            (2.0, 2.0), (2.1, 1.9), (1.9, 2.1),
        ]
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": [{"system_id": "ai", "replicas": [{
                "replica_id": "r1",
                "segments": [{"segment_id": "samples", "projections": [
                    {
                        "source_frame_index": index,
                        "sample_index": index,
                        "scores_angstrom": [x, y],
                    }
                    for index, (x, y) in enumerate(points)
                ]}],
            }]}],
        }
        project = {"definitions": {"clustering_imwkmeans": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "k_values": [2, 3],
            "minkowski_p_values": [1.5, 2.0],
            "initialization_ranks": [0, 1, 2],
            "maximum_iterations": 100,
            "objective_tolerance": 1.0e-10,
            "minimum_cluster_size": 2,
            "weight_dispersion_floor": 1.0e-12,
            "maximum_silhouette_observations": 100,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            environment = {
                "SALSBURY_MD_ANALYSIS_IMWKMEANS_CHECKPOINT": str(root / "checkpoints")
            }
            with patch.dict(os.environ, environment), patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                first = clustering_imwkmeans_project(path)
            self.assertEqual(first["checkpoint_restart"]["written_candidate_count"], 4)
            self.assertEqual(first["checkpoint_restart"]["restored_candidate_count"], 0)
            self.assertEqual(len(list((root / "checkpoints").glob("*.json"))), 4)
            with patch.dict(os.environ, environment), patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ), patch(
                "salsbury_md_analysis.clustering.run_imwkmeans",
                side_effect=AssertionError("checkpointed candidate was refitted"),
            ):
                resumed = clustering_imwkmeans_project(path)
        self.assertEqual(resumed["checkpoint_restart"]["restored_candidate_count"], 4)
        self.assertEqual(resumed["checkpoint_restart"]["written_candidate_count"], 0)
        self.assertEqual(first["selected_model"], resumed["selected_model"])
        self.assertEqual(first["assignments"], resumed["assignments"])

    def test_kmeans_clusters_declared_ion_distance_columns(self):
        points = [(-2.0, -2.1), (-2.1, -1.9), (-1.9, -2.0), (4.0, 4.1), (4.1, 3.9), (3.9, 4.0)]
        records = [
            {
                "source_frame_index": index,
                "axis_kind": "time",
                "axis_value": float(index),
                "values": [first, second],
            }
            for index, (first, second) in enumerate(points)
        ]
        fake_features = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "segments": [{
                "system_id": "thrombin",
                "replica_id": "r1",
                "segment_id": "production",
                "features": [{
                    "feature_id": "ion_site_distances",
                    "kind": "group_distance_statistics",
                    "dimension": 2,
                    "value_labels": ["site_1_angstrom", "site_2_angstrom"],
                    "records": records,
                }],
            }],
        }
        project = {"definitions": {
            "clustering_kmeans": {
                "feature_source": "trajectory_features",
                "trajectory_feature_columns": [{
                    "feature_id": "ion_site_distances",
                    "value_indices": [1, 2],
                }],
                "standardize_features": True,
                "k_values": [2],
                "random_seeds": [3, 7],
                "maximum_iterations": 100,
                "center_tolerance": 1.0e-10,
                "minimum_cluster_size": 2,
                "maximum_silhouette_observations": 100,
            }
        }}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.trajectory_features_project",
                return_value=fake_features,
            ):
                report = clustering_kmeans_project(path)
        self.assertEqual(report["selected_model"]["cluster_sizes"], [3, 3])
        self.assertEqual(report["feature_contract"]["source"], "trajectory_features")
        self.assertEqual(report["feature_contract"]["feature_count"], 2)
        self.assertTrue(all("feature_values" in row for row in report["assignments"]))
        self.assertTrue(all("features_angstrom" not in row for row in report["assignments"]))


if __name__ == "__main__":
    unittest.main()
