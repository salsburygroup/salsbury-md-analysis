import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.reactive_paths import (
    extract_reactive_path_spans,
    multidimensional_dtw_distance,
    reactive_path_ensembles_project,
    reactive_path_ensembles_project_safe,
)


def _definition(**updates):
    value = {
        "assignment_source": "clustering_kmeans",
        "endpoint_mode": "automatic_recurrent_pair",
        "source_state_ids": [],
        "sink_state_ids": [],
        "feature_indices": [1, 2],
        "feature_scaling": "zscore",
        "minimum_pair_events_for_automatic_selection": 2,
        "sakoe_chiba_fraction": 0.25,
        "maximum_paths_per_direction": 100,
        "maximum_path_frames": 100,
        "maximum_pairwise_dtw_cells": 1_000_000,
        "maximum_path_clusters": 3,
        "minimum_path_cluster_size": 2,
        "minimum_complete_paths_for_comparison": 8,
        "minimum_complete_paths_per_direction": 2,
        "minimum_replicas_with_complete_paths": 2,
        "minimum_complete_paths_for_kinetics": 20,
        "minimum_complete_paths_per_direction_for_kinetics": 5,
        "minimum_replicas_with_complete_paths_for_kinetics": 3,
        "require_validated_msm_for_kinetics": True,
    }
    value.update(updates)
    return value


def _rows(replica_id, states, system_id="system"):
    centers = {1: (-3.0, 0.0), 2: (3.0, 0.0), 3: (0.0, 1.0), 4: (0.0, -1.0)}
    return [{
        "system_id": system_id,
        "replica_id": replica_id,
        "segment_id": "production",
        "source_frame_index": index,
        "time": float(index),
        "time_unit": "ps",
        "cluster_id": state,
        "feature_values": [
            centers[state][0] + 0.01 * index,
            centers[state][1] - 0.01 * index,
        ],
    } for index, state in enumerate(states)]


def _clustering(rows):
    return {
        "module_id": "clustering_kmeans",
        "technical_status": "complete",
        "project_manifest_sha256": "a" * 64,
        "system_manifest_path": "/tmp/system.json",
        "system_manifest_sha256": "b" * 64,
        "input_content_signature_sha256": "c" * 64,
        "assignments": rows,
        "issues": [],
    }


class ReactivePathTests(unittest.TestCase):
    def test_last_exit_first_arrival_spans(self):
        self.assertEqual(
            extract_reactive_path_spans(
                [2, 1, 1, 3, 1, 3, 2, 2, 1, 2], [1], [2]
            ),
            [(4, 6), (8, 9)],
        )

    def test_multidimensional_dtw_is_zero_for_identical_paths(self):
        path = [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
        self.assertEqual(multidimensional_dtw_distance(path, path, 0.0), 0.0)
        self.assertGreater(
            multidimensional_dtw_distance(path, [[0.0, 1.0], [2.0, 2.0]], 0.5),
            0.0,
        )

    def test_automatic_pair_reuses_assignments_and_reports_comparison_readiness(self):
        sequence = [1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1]
        rows = _rows("r1", sequence) + _rows("r2", sequence)
        project = {"definitions": {"reactive_path_ensembles": _definition()}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.reactive_paths.clustering_kmeans_project",
                return_value=_clustering(rows),
            ), patch(
                "salsbury_md_analysis.reactive_paths.load_cached_project_report",
                return_value=None,
            ):
                report = reactive_path_ensembles_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["observation_accounting"], {
            "source_physical_frame_count": 26,
            "symmetry_expanded_observation_count": 26,
            "member_observations_are_independent_replicas": None,
            "reactive_path_count": 12,
        })
        self.assertEqual(
            report["endpoint_selection"]["status"], "selected_recurrent_pair"
        )
        self.assertEqual(report["endpoint_selection"]["source_state_ids"], [1])
        self.assertEqual(report["endpoint_selection"]["sink_state_ids"], [2])
        self.assertEqual(report["complete_path_count_by_direction"], {
            "source_to_sink": 6, "sink_to_source": 6,
        })
        self.assertEqual(
            report["transition_sufficiency_status"], "pathway_comparison_ready"
        )
        self.assertFalse(
            report["kinetics_readiness_gates"]["validated_kmeans_markov_model"]["passed"]
        )
        self.assertEqual(
            report["route_analyses"]["source_to_sink"]["route_clustering_status"],
            "complete",
        )

    def test_explicit_multiple_endpoint_sets_and_insufficient_replica_gate(self):
        rows = _rows("r1", [1, 3, 3, 4, 2, 4, 1])
        definition = _definition(
            endpoint_mode="explicit_state_sets",
            source_state_ids=[1, 3], sink_state_ids=[2, 4],
        )
        project = {"definitions": {"reactive_path_ensembles": definition}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.reactive_paths.clustering_kmeans_project",
                return_value=_clustering(rows),
            ), patch(
                "salsbury_md_analysis.reactive_paths.load_cached_project_report",
                return_value=None,
            ):
                report = reactive_path_ensembles_project(path)
        self.assertEqual(
            report["endpoint_selection"]["source_state_ids"], [1, 3]
        )
        self.assertEqual(report["endpoint_selection"]["sink_state_ids"], [2, 4])
        self.assertEqual(
            report["transition_sufficiency_status"], "observed_but_insufficient"
        )
        self.assertFalse(
            report["comparison_sufficiency_gates"]
            ["physical_replicas_with_paths_per_system"]["passed"]
        )
        self.assertIn(
            "INSUFFICIENT_REACTIVE_TRANSITIONS",
            {issue["code"] for issue in report["issues"]},
        )

    def test_pairwise_dtw_resource_gate_is_nonfatal_and_explicit(self):
        sequence = [1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2]
        rows = _rows("r1", sequence)
        project = {"definitions": {"reactive_path_ensembles": _definition(
            maximum_pairwise_dtw_cells=1,
        )}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.reactive_paths.clustering_kmeans_project",
                return_value=_clustering(rows),
            ), patch(
                "salsbury_md_analysis.reactive_paths.load_cached_project_report",
                return_value=None,
            ):
                report = reactive_path_ensembles_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertIn(
            "resource_gate_blocked",
            {row["route_clustering_status"] for row in report["route_analyses"].values()},
        )
        self.assertIn(
            "DTW_RESOURCE_GATE_BLOCKED",
            {issue["code"] for issue in report["issues"]},
        )
        self.assertFalse(
            report["comparison_sufficiency_gates"]
            ["dtw_route_analysis_completed"]["passed"]
        )

    def test_comparison_gates_do_not_pool_one_replica_per_system(self):
        sequence = [1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1]
        rows = (
            _rows("r1", sequence, "k-retained")
            + _rows("r1", sequence, "k-absent")
        )
        project = {"definitions": {"reactive_path_ensembles": _definition()}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.reactive_paths.clustering_kmeans_project",
                return_value=_clustering(rows),
            ), patch(
                "salsbury_md_analysis.reactive_paths.load_cached_project_report",
                return_value=None,
            ):
                report = reactive_path_ensembles_project(path)
        self.assertEqual(
            report["transition_sufficiency_status"],
            "observed_but_insufficient",
        )
        self.assertEqual(
            report["comparison_sufficiency_gates"]
            ["physical_replicas_with_paths_per_system"]["observed"],
            1,
        )
        self.assertFalse(
            report["comparison_sufficiency_gates"]
            ["physical_replicas_with_paths_per_system"]["passed"]
        )
        self.assertEqual(
            {row["system_id"] for row in report["system_transition_summaries"]},
            {"k-retained", "k-absent"},
        )
        route_counts = report["route_analyses"]["source_to_sink"][
            "route_clusters"
        ][0]["system_path_counts"]
        self.assertEqual(
            {row["system_id"] for row in route_counts},
            {"k-retained", "k-absent"},
        )

    def test_overlapping_explicit_endpoints_fail_safe(self):
        project = {"definitions": {"reactive_path_ensembles": _definition(
            endpoint_mode="explicit_state_sets",
            source_state_ids=[1, 2], sink_state_ids=[2, 3],
        )}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            report = reactive_path_ensembles_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn("disjoint", report["issues"][0]["message"])


if __name__ == "__main__":
    unittest.main()
