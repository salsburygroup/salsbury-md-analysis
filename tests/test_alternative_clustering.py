import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.alternative_clustering import (
    _distance_matrix,
    _integer_stride_sample,
    _sklearn_partition,
    alternative_clustering_project,
    calinski_harabasz_score,
    davies_bouldin_score,
    partitioned_local_depths,
    run_mwpam,
    run_pam,
    run_quality_threshold,
    run_ward,
)


POINTS = [
    (-2.0, -2.0), (-2.1, -1.9), (-1.9, -2.1),
    (2.0, 2.0), (2.1, 1.9), (1.9, 2.1),
]


class AlternativeClusteringTests(unittest.TestCase):
    def test_integer_fit_stride_continues_across_segment_boundaries(self):
        metadata = [
            {
                "system_id": "system",
                "replica_id": "r1",
                "segment_id": "s1" if index < 3 else "s2",
                "source_frame_index": index % 3,
            }
            for index in range(6)
        ]
        indices, report = _integer_stride_sample(metadata, 2)
        self.assertEqual(indices, [0, 2, 4])
        self.assertEqual(report["source_frame_stride"], 2)
        self.assertEqual(
            report["mode"], "integer_stride_per_replica_member_timeline_v1"
        )

    def test_pam_and_weighted_pam_are_deterministic_complete_partitions(self):
        pam = run_pam(POINTS, 2)
        weighted = run_mwpam(POINTS, 2)
        self.assertTrue(pam["valid"])
        self.assertTrue(weighted["valid"])
        self.assertEqual(pam["cluster_sizes"], [3, 3])
        self.assertEqual(weighted["cluster_sizes"], [3, 3])
        self.assertEqual(pam, run_pam(POINTS, 2))
        for weights in weighted["feature_weights"]:
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_precomputed_quadratic_work_preserves_partitions(self):
        values = np.asarray(POINTS, dtype=float)
        distances = _distance_matrix(values, 2.0)
        self.assertEqual(
            run_pam(POINTS, 2),
            run_pam(POINTS, 2, _distances=distances),
        )
        self.assertEqual(
            run_mwpam(POINTS, 2),
            run_mwpam(POINTS, 2, _initial_distances=distances),
        )
        scipy_hierarchy = __import__(
            "scipy.cluster.hierarchy", fromlist=["hierarchy"]
        )
        linkage = scipy_hierarchy.linkage(
            values, method="ward", metric="euclidean", optimal_ordering=True
        )
        self.assertEqual(
            run_ward(POINTS, 2),
            run_ward(POINTS, 2, _linkage=linkage),
        )

    def test_ward_and_validation_scores_separate_two_compact_groups(self):
        report = run_ward(POINTS, 2)
        self.assertTrue(report["valid"])
        self.assertEqual(report["cluster_sizes"], [3, 3])
        labels = report["assignments"]
        self.assertGreater(calinski_harabasz_score(POINTS, labels), 100.0)
        self.assertLess(davies_bouldin_score(POINTS, labels), 0.1)

    def test_quality_threshold_retains_noise_and_pald_is_finite(self):
        qt = run_quality_threshold(POINTS + [(20.0, 20.0)], cutoff=0.5)
        self.assertEqual(qt["cluster_sizes"], [3, 3])
        self.assertEqual(qt["noise_count"], 1)
        pald = partitioned_local_depths(POINTS[:4], maximum_observations=4)
        matrix = pald["cohesion_matrix"]
        self.assertEqual((len(matrix), len(matrix[0])), (4, 4))
        self.assertTrue(all(math.isfinite(value) for row in matrix for value in row))
        # PaLD local depths are row sums of cohesion and have population mean 1/2.
        self.assertAlmostEqual(
            sum(sum(row) for row in matrix) / len(matrix), 0.5
        )

    def test_variational_mixture_normalizes_single_value_lower_bound(self):
        report = _sklearn_partition(
            "variational_gaussian_mixture", np.asarray(POINTS), 2, 7,
            {"maximum_iterations": 100},
        )
        self.assertTrue(math.isfinite(report["lower_bound"]))

    def test_project_scans_parameter_grids_and_selects_by_silhouette(self):
        projections = [
            {
                "source_frame_index": index,
                "sample_index": index,
                "scores_angstrom": [x, y],
            }
            for index, (x, y) in enumerate(POINTS)
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
        project = {"definitions": {"alternative_clustering": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "algorithms": ["pam", "ward"],
            "k": 2,
            "k_values": [2, 3],
            "random_seed": 7,
            "random_seeds": [7, 11],
            "maximum_iterations": 100,
            "minkowski_p": 2.0,
            "minkowski_p_values": [1.5, 2.0],
            "quality_threshold_cutoff": 0.5,
            "maximum_observations": 100,
            "maximum_exact_silhouette_observations": 4,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                report = alternative_clustering_project(path)
        sweeps = {row["algorithm"]: row for row in report["parameter_sweeps"]}
        self.assertEqual(sweeps["pam"]["run_count"], 4)
        self.assertEqual(sweeps["ward"]["run_count"], 2)
        self.assertEqual(len(report["algorithm_results"]), 2)
        self.assertTrue(all(
            result["silhouette_evaluation"]["estimated"]
            for result in report["algorithm_results"]
        ))
        self.assertTrue(all(
            len(result["frame_assignments"]) == len(POINTS)
            for result in report["algorithm_results"]
        ))

    def test_project_balances_sampled_fits_and_extends_primary_to_all_frames(self):
        points = [
            (-3.0, -3.0), (-2.9, -3.1), (-3.1, -2.9),
            (3.0, 3.0), (2.9, 3.1), (3.1, 2.9),
        ] * 2
        systems = [{"system_id": "lesion", "replicas": []}]
        for replica_index in range(2):
            offset = replica_index * 6
            projections = [{
                "source_frame_index": index,
                "sample_index": index,
                "scores_angstrom": list(points[offset + index]),
            } for index in range(6)]
            systems[0]["replicas"].append({
                "replica_id": f"r{replica_index + 1}",
                "segments": [{"segment_id": "trajectory", "projections": projections}],
            })
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": systems,
        }
        project = {"definitions": {"alternative_clustering": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "algorithms": ["pam", "ward", "gaussian_mixture"],
            "k": 2,
            "k_values": [2],
            "random_seed": 7,
            "random_seeds": [7],
            "maximum_iterations": 100,
            "minkowski_p": 2.0,
            "minkowski_p_values": [2.0],
            "quality_threshold_cutoff": 0.5,
            "maximum_observations": 6,
            "maximum_exact_silhouette_observations": 4,
            "fit_sampling": {
                "mode": "integer_stride_per_replica_member_v1",
                "strides": [3, 2],
                "primary_stride": 2,
            },
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                report = alternative_clustering_project(path)
        self.assertEqual(report["observation_count"], 12)
        self.assertEqual(report["fit_observation_count"], 6)
        self.assertEqual(
            report["fit_sampling"]["selected_source_matrix_indices"],
            [0, 2, 4, 6, 8, 10],
        )
        self.assertEqual(
            report["fit_sampling"]["mode"],
            "integer_stride_per_replica_member_timeline_v1",
        )
        self.assertEqual(report["fit_sampling"]["source_frame_stride"], 2)
        self.assertEqual(len(report["sampling_sensitivity"]), 1)
        self.assertNotIn(
            "ward",
            {
                result["requested_algorithm"]
                for result in report["algorithm_results"]
            },
        )
        self.assertIn(
            "ward",
            {
                row["algorithm"] for row in report["skipped_algorithms"]
            },
        )
        comparisons = report["sampling_sensitivity"][0]["comparisons_to_primary"]
        self.assertTrue(all(
            row["full_partition_adjusted_rand_to_primary"] is not None
            for row in comparisons
        ))
        for result in report["algorithm_results"]:
            self.assertEqual(len(result["fit_frame_assignments"]), 6)
            self.assertEqual(len(result["frame_assignments"]), 12)
            self.assertEqual(sum(result["full_cluster_sizes"]), 12)
            self.assertEqual(
                result["assignment_extension"]["scope"], "all source observations"
            )

    def test_algorithm_specific_fit_strides_are_executed_and_reported(self):
        points = [
            (-3.0, -3.0), (-2.9, -3.1), (-3.1, -2.9),
            (3.0, 3.0), (2.9, 3.1), (3.1, 2.9),
        ] * 2
        systems = [{"system_id": "lesion", "replicas": []}]
        for replica_index in range(2):
            offset = replica_index * 6
            systems[0]["replicas"].append({
                "replica_id": f"r{replica_index + 1}",
                "segments": [{
                    "segment_id": "trajectory",
                    "projections": [{
                        "source_frame_index": index,
                        "sample_index": index,
                        "scores_angstrom": list(points[offset + index]),
                    } for index in range(6)],
                }],
            })
        fake_pca = {
            "project_manifest_sha256": "a" * 64,
            "system_manifest_path": "/tmp/system.json",
            "system_manifest_sha256": "b" * 64,
            "input_content_signature_sha256": "c" * 64,
            "issues": [],
            "systems": systems,
        }
        common = {
            "full_observation_count": 12,
            "calibration_status": "provisional_complexity_profile_v1",
        }
        fit_sampling = {
            "mode": "algorithm_specific_integer_stride_v1",
            "target_wall_hours": 24.0,
            "member_observation_multiplier": 1,
            "source_physical_frames_per_replica": [6, 6],
            "full_observation_count": 12,
            "scientific_boundary": "computational only",
            "algorithm_plans": {
                "pam": {
                    **common,
                    "execution": "run",
                    "mode": "integer_stride_per_replica_member_v1",
                    "strides": [2],
                    "primary_stride": 2,
                    "fit_observation_ceiling": 6,
                    "selected_fit_observations_per_physical_replica": [3, 3],
                    "selected_fit_observation_count": 6,
                    "complexity_class": "quadratic",
                    "time_exponent": 2.0,
                },
                "gaussian_mixture": {
                    **common,
                    "execution": "run",
                    "mode": "integer_stride_per_replica_member_v1",
                    "strides": [1],
                    "primary_stride": 1,
                    "fit_observation_ceiling": 12,
                    "selected_fit_observations_per_physical_replica": [6, 6],
                    "selected_fit_observation_count": 12,
                    "complexity_class": "linear",
                    "time_exponent": 1.0,
                },
                "ward": {
                    **common,
                    "execution": "skip",
                    "skip_reason": "full fit exceeds resource ceiling",
                    "fit_observation_ceiling": 6,
                    "complexity_class": "full_or_skip",
                    "time_exponent": 2.0,
                },
            },
        }
        project = {"definitions": {"alternative_clustering": {
            "feature_source": "common_pca",
            "component_indices": [1, 2],
            "standardize_features": True,
            "algorithms": ["pam", "gaussian_mixture", "ward"],
            "k": 2,
            "k_values": [2],
            "random_seed": 7,
            "random_seeds": [7],
            "maximum_iterations": 100,
            "minkowski_p": 2.0,
            "minkowski_p_values": [2.0],
            "quality_threshold_cutoff": 0.5,
            "maximum_observations": 12,
            "maximum_exact_silhouette_observations": 4,
            "fit_sampling": fit_sampling,
        }}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.feature_matrix.common_pca_project",
                return_value=fake_pca,
            ):
                report = alternative_clustering_project(path)
        self.assertEqual(
            report["algorithm_fit_observation_counts"],
            {"pam": 6, "gaussian_mixture": 12},
        )
        self.assertEqual(
            report["fit_sampling"]["algorithms"]["pam"]
            ["resolved_sampling"]["source_frame_stride"],
            2,
        )
        self.assertEqual(
            report["fit_sampling"]["algorithms"]["gaussian_mixture"]
            ["resolved_sampling"]["source_frame_stride"],
            1,
        )
        self.assertIn("ward", {
            row["algorithm"] for row in report["skipped_algorithms"]
        })


if __name__ == "__main__":
    unittest.main()
